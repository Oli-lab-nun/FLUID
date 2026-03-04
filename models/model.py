import torch
import torch.nn as nn
import os
import json
import random
from typing import Optional, List
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.utils import logging
from transformers.cache_utils import DynamicCache
from peft import PeftModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class FLUID(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig 
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def __init__(self, base_model: PreTrainedModel, k_masks=None, **kwargs):
        config = base_model.config
        super().__init__(config)
        
        self.model = base_model
        if hasattr(self.model, "get_output_embeddings"):
            self.lm_head = self.model.get_output_embeddings()
        else:
            self.lm_head = getattr(self.model, "lm_head", None)

        self.mask_token_id = -1 
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = None
        self.k_masks = k_masks
        self.generation_config = self.model.generation_config
        
        # store special token ID
        self.ignored_token_ids = []
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # -------------------  openPangu  -------------------
        name = getattr(tokenizer, "name_or_path", "") or ""
        name = name.lower()
    
        if not name and hasattr(tokenizer, "_tokenizer_config"):
            cfg_name = tokenizer._tokenizer_config.get("name_or_path", "")
            name = cfg_name.lower()
        is_openpangu = ("openpangu" in name)

        if is_openpangu:
            unused32_id = 144208     # select unused32 as mask embedding
            self.mask_token_id = unused32_id
            logger.info(f"[OpenPangu] 使用固定 mask_token_id={self.mask_token_id} (来自 unused32)")

        # ------------------- other models -------------------
        else:
            mask_token = "<mask>"

            if mask_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})

            self.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
            logger.info(f"[Normal] <mask> 正常添加, id={self.mask_token_id}")

        # ------------------- publish ops-------------------
        self.ignored_token_ids = tokenizer.all_special_ids
        self.special_ids_tensor = torch.tensor(self.ignored_token_ids, dtype=torch.long)

        logger.info(f"ignore token IDs: {self.ignored_token_ids}")
        logger.info(f"mask_token_id is {self.mask_token_id}")

    def _prepare_batch_vectorized(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Refactored to meet specific requirements:
        1. No masks in Prompt (labels == -100).
        2. Response: 50% keep original (k=0), 50% insert masks (1-K).
        3. Random Restoration of masks (0-50% ratio).
        4. Attention Mask ensures Prompt doesn't attend to template special tokens.
        """
        # torch.set_printoptions(threshold=(2048))
        device = input_ids.device
        B, N = input_ids.shape
        
        # --- 0. 预处理：构建 Shifted Labels ---
        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        shifted_labels[:, -1] = -100

        # --- 1. 确定 Mask 插入策略 (Requirement 1 & 3) ---
        # 只有在 labels != -100 (即 Response 部分) 才允许插入
        is_response_col = (shifted_labels != -100).all(dim=0)
        
        # 生成基础概率 (N,)
        rand_probs = torch.rand(N, device=device)
        
        # 筛选 Candidates (N,)
        mask_candidates = is_response_col & (rand_probs > 0.5)
        
        # 初始化 k_counts (N,)
        k_counts = torch.zeros(N, dtype=torch.long, device=device)
        
        if self.k_masks is not None and self.k_masks > 0:
            random_k = torch.randint(1, self.k_masks + 1, (N,), device=device)
            # 只有 candidate 位置设为 random_k
            k_counts = torch.where(mask_candidates, random_k, k_counts) 

        # --- 2. 计算扩增索引映射 ---
        repeats = 1 + k_counts # (N,)
        
        col_indices = torch.repeat_interleave(torch.arange(N, device=device), repeats) 
        L_new = col_indices.shape[0]
        
        # 起点
        cumsum_repeats = torch.cumsum(repeats, dim=0)
        block_starts = torch.cat([torch.tensor([0], device=device), cumsum_repeats[:-1]])
        
        range_indices = torch.arange(L_new, device=device)
        offsets = range_indices - block_starts[col_indices] # (L_new,)
        
        is_mask = (offsets > 0) # offsets > 0 是插入的 Mask 位置
        
        # --- 3. 构建 New Input IDs ---
        new_input_ids = torch.full((B, L_new), self.mask_token_id, dtype=input_ids.dtype, device=device)
        
        # 获取原始 Token
        expanded_inputs = input_ids[:, col_indices]
        # 填充 Main Token 
        new_input_ids = torch.where(is_mask.unsqueeze(0), new_input_ids, expanded_inputs)
        
        # --- 4. 构建 New Position IDs 和 Labels ---
        new_position_ids = (col_indices + offsets).unsqueeze(0).expand(B, -1)

        target_indices = col_indices + offsets
        valid_target = (target_indices < N)
        safe_target_indices = target_indices.clamp(max=N-1)
        
        expanded_targets = shifted_labels[:, safe_target_indices]
        new_labels = expanded_targets.masked_fill(~valid_target.unsqueeze(0), -100)

        # --- 5. 构建 Attention Mask ---
        # 准备广播向量
        block_ids = col_indices # (L_new,)
        steps = col_indices + offsets # (L_new,)
        types = is_mask.long() # (L_new,)
        
        # 扩展维度 (Query: col vector, Key: row vector)
        q_block = block_ids.unsqueeze(1)
        k_block = block_ids.unsqueeze(0)
        q_type = types.unsqueeze(1)
        k_type = types.unsqueeze(0)
        q_step = steps.unsqueeze(1)
        k_step = steps.unsqueeze(0)
        
        # 初始化 -inf
        attn_bias = torch.full((L_new, L_new), float("-inf"), device=device)
        
        # [Rule A] 保证 Main Token 之间可见性正常 (标准的 Causal Mask)
        mask_key_main = (k_type == 0) & (k_block <= q_block)
        
        # [Rule B] Key 是 Mask Token (Block 内部)
        # 如果 k=0，types 全为 0，这一项全为 False，不影响结果
        # 如果 k>0，允许 Mask 看到同一 Block 内之前的 Mask
        mask_key_mask = (k_type == 1) & (q_type == 1) & (k_block == q_block) & (k_step <= q_step)
        
        # 组合规则
        valid_connections = mask_key_main | mask_key_mask
        
        attn_bias.masked_fill_(valid_connections, 0.0)
        
        # 扩展到 Batch
        new_attention_mask = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)


        # --- 6. 随机还原 Mask (Requirement 4) ---
        
        # 1. 确定哪些位置是 Mask
        mask_indices_bool = is_mask.unsqueeze(0).expand(B, -1) # (B, L_new)
        
        # 2. 为每个位置生成还原概率
        restore_ratios = torch.rand(N, device=device) * 1.0 # 0 ~ 1.0
        expanded_ratios = restore_ratios[col_indices] # (L_new,)
        
        # 对每个位置生成随机数，如果 < ratio 且是 mask，则还原
        element_probs = torch.rand(B, L_new, device=device)
        should_restore = mask_indices_bool & (element_probs < expanded_ratios.unsqueeze(0))
        
        if should_restore.any():
            # --- [Requirement 4 Enhanced] 随机还原 + 噪声注入 ---
            # 1. 准备真实的未来 Token (Ground Truth)
            # safe_gather_indices: (L_new,)
            safe_gather_indices = target_indices.clamp(max=N-1)
            # real_tokens: (B, L_new)
            real_tokens = input_ids[:, safe_gather_indices]
            
            # 2. 准备随机噪声 Token (Noise)
            # 目标：模拟前序预测出错的情况
            vocab_size = len(self.tokenizer)
            # 生成与 real_tokens 形状一致的随机整数
            noise_tokens = torch.randint(0, vocab_size, real_tokens.shape, device=device, dtype=input_ids.dtype)
            
            # 3. [关键] 过滤特殊 Token (No Special Tokens)
            # 噪声不应该是 EOS/PAD 等特殊功能符，否则会破坏句法结构导致模型“偷懒”或崩溃
            if self.tokenizer is not None:
                # special_ids_tensor = torch.tensor(special_ids_list, device=device)
                if self.special_ids_tensor.device != device:
                    self.special_ids_tensor = self.special_ids_tensor.to(device)
                # 检查哪些噪声命中了特殊 Token
                is_special_noise = torch.isin(noise_tokens, self.special_ids_tensor)
                
                # 如果命中了，简单策略：直接用 real_tokens 覆盖它 (放弃这次噪声注入)
                # 或者：再次随机 (为了性能，这里选择偏移策略或回退到真值)
                # 这里采用回退策略：如果随机出特殊token，为了安全起见，这次就不加噪声了，算作运气好预测对了
                noise_tokens = torch.where(is_special_noise, real_tokens, noise_tokens)

            # 4. 生成噪声掩码 (10% Noise Rate)
            noise_prob = 0.1
            # 在所有需要 restore 的位置中，随机挑 10% 替换为噪声
            # use_noise: (B, L_new)
            use_noise = torch.rand(real_tokens.shape, device=device) < noise_prob
            
            # 5. 组合最终用于还原的 Token
            # 90% 是真值，10% 是噪声
            tokens_to_restore = torch.where(use_noise, noise_tokens, real_tokens)
            
            # 6. 执行还原
            # 只有在 (需要还原 & 目标位置有效) 的地方才填入
            valid_restore = should_restore & valid_target.unsqueeze(0)
            
            # 将 mask_token_id 替换为 (Real or Noise)
            new_input_ids = torch.where(valid_restore, tokens_to_restore, new_input_ids)
        

        return new_input_ids, new_labels, new_position_ids, new_attention_mask
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        if self.mask_token_id == -1:
             raise ValueError("Please call model.set_tokenizer(tokenizer) first.")

        if labels is not None:
            # 训练模式
            (
                new_input_ids, 
                new_labels, 
                new_position_ids, 
                new_attention_mask
            ) = self._prepare_batch_vectorized(input_ids, labels)
            
            outputs = self.model.model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                use_cache=False, 
                return_dict=True
            )
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(hidden_states)
            
            loss = self.criterion(logits.view(-1, logits.size(-1)), new_labels.view(-1))
            
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        else:
            return self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
            
    # ... (Keep other methods unchanged: get_tokenizer, embeddings, etc.)
    def get_tokenizer(self):
        return self.tokenizer
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        logger.warning("set_output_embeddings called, but lm_head is intended to be frozen.")
        self.lm_head = new_embeddings
        self.model.set_output_embeddings(new_embeddings)
    def update_from_adapter(self, adapter_name):
        # 引入必要的底层加载函数
        from peft import PeftModel, PeftConfig
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        import torch
        
        print(f"[Info] Loading adapter from {adapter_name}...")

        # 1. 仅加载配置 
        config = PeftConfig.from_pretrained(adapter_name)
        
        # 2. 初始化 PeftModel 结构，加载到 NPU 上
        peft_model = PeftModel(self, config)
        
        # 3. 强制使用 'cpu' 读取权重文件
        print("[Info] Manually loading weights to CPU RAM...")
        adapter_weights = load_peft_weights(adapter_name, device="cpu")
        
        # 4. 将 CPU 上的权重注入到 NPU 模型中
        print("[Info] Injecting weights into NPU model...")
        set_peft_model_state_dict(peft_model, adapter_weights)
        
        # 5. 合并权重并卸载 Adapter
        print("[Info] Merging and unloading...")
        peft_model.merge_and_unload()
        
        # 6. 清理内存
        del adapter_weights
        del peft_model
        if torch.npu.is_available():
            torch.npu.empty_cache()
            
        print("Loading model done (Manual Method)")
    @torch.no_grad()
    def infer(
        self, 
        eos_token_id, 
        input_ids, 
        max_new_tokens=128, 
        block_size=8,  
        confidence_threshold=0.9,
        remask=False, 
    ):
        if self.mask_token_id == -1:
             raise ValueError("Please call model.set_tokenizer(tokenizer) first.")
        device = input_ids.device
        
        initial_len = input_ids.shape[1]
        current_ids = input_ids.clone()

        # generating 
        while current_ids.shape[1] - initial_len < max_new_tokens:
            
            context_len = current_ids.shape[1]
            
            # --- 1. generate Phase: Iterative Filling ---
            generate_ids = torch.full(
                (current_ids.shape[0], block_size), 
                self.mask_token_id, 
                dtype=current_ids.dtype, 
                device=device
            )
            
            bonus_token = None 
            step = 0
            refinement_count = 0 
            
            # block generating ...
            while step < block_size:
                inp = torch.cat([current_ids, generate_ids], dim=1)
                
                # Forward
                outputs = self.model(input_ids=inp, use_cache=False)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    hidden_states = outputs.last_hidden_state
                    logits = self.lm_head(hidden_states)

                relevant_logits = logits[:, context_len - 1 :, :] 
                probs = torch.softmax(relevant_logits, dim=-1)
                max_probs, best_ids = torch.max(probs, dim=-1)
                
                # A. Main Token 推进
                generate_ids[:, step] = best_ids[:, step]
                
                # B. 并行扫描 + 跳跃
                accepted_jump = 0

                # 逻辑链条是否断裂
                chain_broken = False
                
                for lookahead_idx in range(step + 1, block_size):
                    conf = max_probs[:, lookahead_idx]
                    bid = best_ids[:, lookahead_idx]
                    
                    if (conf > confidence_threshold).all():
                        generate_ids[:, lookahead_idx] = bid
                        if not chain_broken:
                            accepted_jump += 1
                    else:
                        chain_broken = True

                # 检查是否填满
                if (generate_ids != self.mask_token_id).all():
                    last_token_conf = max_probs[:, -1]
                    last_token_id = best_ids[:, -1]
                    if (last_token_conf > confidence_threshold).all():
                        bonus_token = last_token_id.unsqueeze(1)
                    break
                
                step += (1 + accepted_jump)
                refinement_count += 1

            # --- 2. remask Phase ---
            segment_to_remask = generate_ids
            if bonus_token is not None:
                segment_to_remask = torch.cat([generate_ids, bonus_token], dim=1)
            
            # 如果不开启remask，直接生成
            if not remask:
                current_ids = torch.cat([current_ids, segment_to_remask], dim=1)
            
            else:
                # remask
                full_sequence = torch.cat([current_ids, segment_to_remask], dim=1)
                
                # new forward
                new_outputs = self.model(input_ids=full_sequence, use_cache=False)
                
                if hasattr(new_outputs, "logits"):
                    new_logits = new_outputs.logits
                else:
                    new_logits = self.lm_head(new_outputs.last_hidden_state)
                
                segment_len = segment_to_remask.shape[1]
                pred_logits = new_logits[:, context_len - 1 : context_len - 1 + segment_len, :]
                pre_gen_tokens = segment_to_remask
                

                best_tokens = torch.argmax(pred_logits, dim=-1)
                is_correct_mask = (pre_gen_tokens == best_tokens)

                mismatch_indices = (~is_correct_mask).nonzero(as_tuple=True)
                
                if len(mismatch_indices[1]) > 0:
                    first_fail_idx = mismatch_indices[1][0].item()
                    print(f"Mismatch at {first_fail_idx}. remasking...")
                    
                    # remask
                    best_correct_token = torch.argmax(pred_logits[:, first_fail_idx, :], dim=-1).unsqueeze(1)
                    
                    if first_fail_idx > 0:
                        valid_part = segment_to_remask[:, :first_fail_idx]
                        new_extension = torch.cat([valid_part, best_correct_token], dim=1)
                    else:
                        new_extension = best_correct_token
                        
                    current_ids = torch.cat([current_ids, new_extension], dim=1)
                else:
                    current_ids = full_sequence 

            # --- 3. EOS Check ---
            newly_added_start = context_len
            if current_ids.shape[1] > newly_added_start:
                new_part = current_ids[:, newly_added_start:]
                if eos_token_id is not None and (new_part == eos_token_id).any():
                    eos_indices = (new_part == eos_token_id).nonzero(as_tuple=True)
                    if len(eos_indices[1]) > 0:
                        first_eos_local = eos_indices[1].min().item()
                        current_ids = current_ids[:, :newly_added_start + first_eos_local + 1]
                    break
        
        return current_ids 

    @torch.no_grad()
    def infer_kv(
        self, 
        eos_token_id, 
        input_ids, 
        max_new_tokens=128, 
        block_size=8, 
        confidence_threshold=0.9,
        remask=False,
    ):
        if self.mask_token_id == -1:
             raise ValueError("Please call model.set_tokenizer(tokenizer) first.")
        
        # ================= NPU Patch Start =================
        # 动态获取定义了 NPU_ATTN_INFR 的模块 (即 modeling_openpangu_dense)
        # 目的是在生成期间临时关闭 NPU 融合算子，避免 Kernel 崩溃
        base_model_module = sys.modules[self.model.model.__module__]
        original_npu_flag = getattr(base_model_module, "NPU_ATTN_INFR", False)
        
        # 强制关闭 NPU Fused Attention
        # 这会迫使模型走 eager_attention_forward 或 SDPA，它们支持 Block Draft (Q>1 + Cache)
        if original_npu_flag:
            setattr(base_model_module, "NPU_ATTN_INFR", False)
            # logger.info("Temporarily disabled NPU_ATTN_INFR for block generation.")
        # ================= NPU Patch End =================

        try:
            device = input_ids.device
            initial_len = input_ids.shape[1]
            
            # 1. 初始化 KV Cache
            past_key_values = DynamicCache()
            
            # 2. Prefill & 获取 Prefix Logits
            cache_position = torch.arange(initial_len, device=device)
            outputs = self.model(
                input_ids=input_ids, 
                use_cache=True, 
                past_key_values=past_key_values,
                cache_position=cache_position
            )
            
            if hasattr(outputs, "logits"):
                prefix_next_token_logit = outputs.logits[:, -1:, :]
            else:
                hidden_states = outputs.last_hidden_state
                prefix_next_token_logit = self.lm_head(hidden_states[:, -1:, :])

            current_ids = input_ids.clone()
            global_step_counter = 0

            while current_ids.shape[1] - initial_len < max_new_tokens:
                
                prefix_len = past_key_values.get_seq_length()
                
                # init empty Block
                generate_ids = torch.full(
                    (input_ids.shape[0], block_size), 
                    self.mask_token_id, 
                    dtype=input_ids.dtype, 
                    device=device
                )
                
                bonus_token = None 
                step = 0
                refinement_count = 0 
                current_prefix_logit = prefix_next_token_logit

                # --- Generate Phase ---
                while step < block_size:
                    cache_position = torch.arange(prefix_len, prefix_len + block_size, device=device)
                    
                    outputs = self.model(
                        input_ids=generate_ids, 
                        use_cache=True, 
                        past_key_values=past_key_values,
                        cache_position=cache_position
                    )
                    
                    generate_logits = outputs.logits if hasattr(outputs, "logits") else self.lm_head(outputs.last_hidden_state)
                    full_logits = torch.cat([current_prefix_logit, generate_logits], dim=1)

                    probs = torch.softmax(full_logits, dim=-1)
                    max_probs, best_ids = torch.max(probs, dim=-1)
                    
                    generate_ids[:, step] = best_ids[:, step]
                    
                    accepted_jump = 0
                    chain_broken = False
                    for lookahead_idx in range(step + 1, block_size):
                        if (max_probs[:, lookahead_idx] > confidence_threshold).all():
                            generate_ids[:, lookahead_idx] = best_ids[:, lookahead_idx]
                            if not chain_broken: accepted_jump += 1
                        else:
                            chain_broken = True
                    
                    global_step_counter += 1

                    if (generate_ids != self.mask_token_id).all():
                        if (max_probs[:, block_size] > confidence_threshold).all():
                            bonus_token = best_ids[:, block_size].unsqueeze(1)
                        past_key_values.crop(prefix_len)
                        break
                    
                    past_key_values.crop(prefix_len)
                    step += (1 + accepted_jump)
                    refinement_count += 1

                # --- Remask Phase ---
                segment_to_remask = generate_ids
                if bonus_token is not None:
                    segment_to_remask = torch.cat([generate_ids, bonus_token], dim=1)
                
                final_segment = None
                
                if not remask:
                    final_segment = segment_to_remask
                    cache_position = torch.arange(prefix_len, prefix_len + final_segment.shape[1], device=device)
                    outputs = self.model(
                        input_ids=final_segment,
                        use_cache=True,
                        past_key_values=past_key_values,
                        cache_position=cache_position
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else self.lm_head(outputs.last_hidden_state)
                    prefix_next_token_logit = logits[:, -1:, :]
                else:
                    cache_position = torch.arange(prefix_len, prefix_len + segment_to_remask.shape[1], device=device)
                    new_outputs = self.model(
                        input_ids=segment_to_remask,
                        use_cache=True,
                        past_key_values=past_key_values,
                        cache_position=cache_position
                    )
                    new_logits = new_outputs.logits if hasattr(new_outputs, "logits") else self.lm_head(new_outputs.last_hidden_state)
                    
                    full_new_logits = torch.cat([current_prefix_logit, new_logits[:, :-1, :]], dim=1)
                    
                    # remask
                    pred_logits = full_new_logits
                    min_len = min(pred_logits.shape[1], segment_to_remask.shape[1])
                    pred_logits = pred_logits[:, :min_len, :]
                    target_tokens = segment_to_remask[:, :min_len]

                    
                    is_correct_mask = (target_tokens == torch.argmax(pred_logits, dim=-1))

                    mismatch_indices = (~is_correct_mask).nonzero(as_tuple=True)
                    
                    if len(mismatch_indices[1]) > 0:
                        past_key_values.crop(prefix_len) # Rollback
                        first_fail_idx = mismatch_indices[1][0].item()
                        print(f"Mismatch at {first_fail_idx}. Correcting...")
                        best_correct_token = torch.argmax(pred_logits[:, first_fail_idx, :], dim=-1).unsqueeze(1)
                        
                        if first_fail_idx > 0:
                            final_segment = torch.cat([segment_to_remask[:, :first_fail_idx], best_correct_token], dim=1)
                        else:
                            final_segment = best_correct_token
                        
                        # commit
                        cache_position = torch.arange(prefix_len, prefix_len + final_segment.shape[1], device=device)
                        outputs = self.model(
                            input_ids=final_segment,
                            use_cache=True,
                            past_key_values=past_key_values,
                            cache_position=cache_position
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else self.lm_head(outputs.last_hidden_state)
                        prefix_next_token_logit = logits[:, -1:, :]
                    else:
                        final_segment = segment_to_remask
                        prefix_next_token_logit = new_logits[:, -1:, :]

                # --- Update current_ids & EOS Check ---
                current_ids = torch.cat([current_ids, final_segment], dim=1)

                if eos_token_id is not None:
                    is_eos = (final_segment == eos_token_id)
                    if is_eos.any():
                        first_eos_idx = is_eos.nonzero(as_tuple=True)[1].min().item()
                        tokens_to_remove = final_segment.shape[1] - (first_eos_idx + 1)
                        if tokens_to_remove > 0:
                            current_ids = current_ids[:, :-tokens_to_remove]
                        break

            return current_ids
            
        finally:
            # ================= Restore NPU Flag =================
            # 无论生成是否成功，都要把标志位还原，以免影响其他代码的正常运行
            if original_npu_flag:
                setattr(base_model_module, "NPU_ATTN_INFR", True)
                # logger.info("Restored NPU_ATTN_INFR.")
    def load_model_for_inference(model, head_checkpoint_path):
        
        # 3. 加载你刚刚单独保存的 Head
        head_weight_path = os.path.join(head_checkpoint_path, "model_head.bin")
        
        if os.path.exists(head_weight_path):
            # 加载权重
            state_dict = torch.load(head_weight_path, map_location="cpu")
            
            # [关键] 使用 strict=False
            # 因为 state_dict 里只有 diffusion_k_head 的权重，
            # 而 model 里还有整个 base model，strict=True 会报错说缺少 key。
            # strict=False 会忽略缺失的 base model 权重（反正它们已经在内存里了），
            # 只更新匹配到的 diffusion_k_head。
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            print(f"Head 加载成功! 忽略了 {len(missing_keys)} 个 Base Model 的 key。")
        else:
            print("警告: 未找到 Head 权重文件，使用随机初始化 (仅用于测试流程)。")
            
        return model