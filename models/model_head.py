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

        # 扩散步长预测头 (Diffusion K-Head)
        # 回归模式预测一个标量 K (1.0 ~ max_k)
        self.max_k = k_masks if k_masks else 0
        self.diffusion_k_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, self.max_k)
        )        
        

        # # 语义熵阈值，平均 Loss 超过此值则认为扩散受阻
        self.loss_threshold = 2.8  
        self.diffusion_k_head_train = True

    def set_tokenizer(self, tokenizer):

        self.tokenizer = tokenizer
        # ------------------- openPangu -------------------
        name = getattr(tokenizer, "name_or_path", "") or ""
        name = name.lower()
        # 如果是本地 tokenizer，name 为空，则从 config 补充
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
                # 对普通模型，如果你想真正使用这个 token，才需要 resize
                # self.base_model.resize_token_embeddings(len(tokenizer))

            self.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
            logger.info(f"[Normal] <mask> 正常添加, id={self.mask_token_id}")

        # ------------------- publish ops -------------------
        self.ignored_token_ids = tokenizer.all_special_ids
        self.special_ids_tensor = torch.tensor(self.ignored_token_ids, dtype=torch.long)

        logger.info(f"忽略 token IDs: {self.ignored_token_ids}")
        logger.info(f"最终 mask_token_id = {self.mask_token_id}")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        if self.mask_token_id == -1:
             raise ValueError("Please call model.set_tokenizer(tokenizer) first.")
        return self.train_head_forward(input_ids, labels)
    def train_head_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        B, N = input_ids.shape
        # 仅在训练 Head 时使用新逻辑
        if labels is not None:
            # 1. 强制插入 Max K Mask
            (
                new_input_ids, new_labels, new_position_ids, new_attention_mask, is_mask
            ) = self._prepare_batch_for_head_training(input_ids, labels)
            
            # 2. 冻结的主模型 Forward
            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    position_ids=new_position_ids,
                    use_cache=False
                )
                hidden_states = outputs.last_hidden_state
                lm_logits = self.lm_head(hidden_states)
                
                # 3. 计算每个 Mask 的独立 Loss 
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                token_losses = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), 
                    new_labels.view(-1)
                ) 
                # 恢复形状 (Batch, New_Length)
                token_losses = token_losses.view(new_labels.shape)

                # 4. 生成 Head 的标签 (Target K)
                is_mask_bool = is_mask.unsqueeze(0).expand(B, -1)
                
                shifted_mask = torch.roll(is_mask_bool, shifts=-1, dims=1)
                shifted_mask[:, -1] = False
                is_trigger = (~is_mask_bool) & shifted_mask # 找到 Block 的发射点

                # 计算最佳 K
                # 这里传入 max_k=self.max_k，逻辑是：
                # 检查 Trigger 后面跟着的 max_k 个 loss，找到第一个"太难"的地方截断
                target_ks = self._calculate_cutoff_k(
                    token_losses, is_trigger, is_mask_bool, 
                    max_k=self.max_k, 
                    threshold=self.loss_threshold
                )

            # 5. 训练 Head (高斯软标签分类模式)
            if is_trigger.any():
                trigger_states = hidden_states[is_trigger]
                
                # 1. 获取预测 Logits (M, 16)
                pred_logits = self.diffusion_k_head(trigger_states)
                
                # 2. 准备真实标签索引 (M,)
                target_indices = (target_ks.long() - 1).clamp(min=0, max=self.max_k - 1)
                
                # ================= 生成高斯软标签 =================
                # 我们不直接用 CrossEntropy，而是构造一个目标分布
                M = target_indices.size(0)
                num_classes = self.max_k
                
                # 生成所有类别的索引 [0, 1, ..., 15]
                class_indices = torch.arange(num_classes, device=trigger_states.device).unsqueeze(0).expand(M, -1)
                
                # 目标中心 (M, 1)
                target_centers = target_indices.unsqueeze(1)
                
                # 计算距离平方 (M, Num_Classes)
                dist_sq = (class_indices - target_centers) ** 2
                
                # 生成高斯分布: exp(-dist^2 / (2 * sigma^2))
                # sigma 控制容忍度：sigma=1.0 表示容忍 ±1~2 的误差，sigma=2.0 容忍更大
                sigma = 1.5 
                soft_targets = torch.exp(-dist_sq / (2 * sigma ** 2))
                
                # 归一化，使概率和为 1
                soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
                
                # 3. 计算 Loss (KL 散度 或 Soft CrossEntropy)
                log_probs = nn.LogSoftmax(dim=1)(pred_logits)
                loss_head = -(soft_targets * log_probs).sum(dim=1).mean()
                
            else:
                # Dummy 分支 (保持维度一致)
                dummy_input = hidden_states.view(-1, hidden_states.size(-1))[0].unsqueeze(0)
                dummy_pred = self.diffusion_k_head(dummy_input)
                loss_head = dummy_pred.sum() * 0.0
            
            return CausalLMOutputWithPast(loss=loss_head, logits=None)

    def _calculate_cutoff_k(self, token_losses, is_trigger, is_mask, max_k, threshold):
        """
        计算截断点：找到连续的 Mask，其平均 Loss 保持在 threshold 以下的最大长度。
        """
        B, L = token_losses.shape
        device = token_losses.device
        
        # 默认 K=1 (至少预测一步)
        optimal_ks = torch.ones_like(token_losses, dtype=torch.float)
        
        cumulative_loss = torch.zeros_like(token_losses)
        
        # 循环 1 到 8
        for k in range(1, max_k + 1):
            # 把第 k 个 Mask 的 loss 移到 Trigger 这一格
            shifted_loss = torch.roll(token_losses, shifts=-k, dims=1)
            
            # 累积 Loss (计算前 k 个的平均值)
            cumulative_loss += shifted_loss
            avg_loss = cumulative_loss / k
            
            # 判断条件：平均 loss < 阈值
            is_good = (avg_loss < threshold)
            
            update_mask = is_trigger & is_good
            
            # 注意：必须保证第 k 个位置确实是 Mask (防止越界读到了别的句子)
            shifted_is_mask = torch.roll(is_mask, shifts=-k, dims=1)
            update_mask = update_mask & shifted_is_mask
            
            # 更新值
            optimal_ks = torch.where(update_mask, torch.tensor(float(k), device=device), optimal_ks)
            
        # 只返回 Trigger 处算出来的 K
        return optimal_ks[is_trigger]
    def _prepare_batch_for_head_training(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        专门用于训练 Diffusion Head 的数据准备。
        区别在于：在随机选中的位置，强制插入 max_k 个 Mask，以便探测模型能力的边界。
        """
        device = input_ids.device
        B, N = input_ids.shape
        self.max_k = self.k_masks if self.k_masks else 8
        
        # --- 1. 确定插入位置 (保持你的随机逻辑) ---
        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        shifted_labels[:, -1] = -100
        is_response_col = (shifted_labels != -100).all(dim=0)
        
        rand_probs = torch.rand(N, device=device)
        # 只选25%的部分，为了更长的上下文
        mask_candidates = is_response_col & (rand_probs > 0.5) # 选中这些位置
        
        k_counts = torch.zeros(N, dtype=torch.long, device=device)
        
        # [关键修改]：只要选中，就强制插入 self.max_k，而不是随机 random_k
        # 这样我们可以看到模型在"极限"情况下的表现，从而计算出最佳截断点
        k_counts = torch.where(mask_candidates, torch.tensor(self.max_k, device=device), k_counts) 

        # --- 2. 后续逻辑保持不变 (构建 New Input IDs) ---
        repeats = 1 + k_counts 
        col_indices = torch.repeat_interleave(torch.arange(N, device=device), repeats) 
        L_new = col_indices.shape[0]
        
        cumsum_repeats = torch.cumsum(repeats, dim=0)
        block_starts = torch.cat([torch.tensor([0], device=device), cumsum_repeats[:-1]])
        range_indices = torch.arange(L_new, device=device)
        offsets = range_indices - block_starts[col_indices]
        
        is_mask = (offsets > 0) # 偏移量 >0 的都是 Mask
        
        # 构建 new input
        new_input_ids = torch.full((B, L_new), self.mask_token_id, dtype=input_ids.dtype, device=device)
        expanded_inputs = input_ids[:, col_indices]
        new_input_ids = torch.where(is_mask.unsqueeze(0), new_input_ids, expanded_inputs)
        
        # 构建 new labels
        target_indices = col_indices + offsets
        valid_target = (target_indices < N)
        safe_target_indices = target_indices.clamp(max=N-1)
        expanded_targets = shifted_labels[:, safe_target_indices]
        new_labels = expanded_targets.masked_fill(~valid_target.unsqueeze(0), -100)
        
        # 构建 Position IDs
        new_position_ids = (col_indices + offsets).unsqueeze(0).expand(B, -1)
        
        # 构建 Attention Mask
        
        # 简写复用原逻辑:
        block_ids = col_indices
        steps = col_indices + offsets
        types = is_mask.long()
        
        q_block = block_ids.unsqueeze(1)
        k_block = block_ids.unsqueeze(0)
        q_type = types.unsqueeze(1)
        k_type = types.unsqueeze(0)
        q_step = steps.unsqueeze(1)
        k_step = steps.unsqueeze(0)
        
        attn_bias = torch.full((L_new, L_new), float("-inf"), device=device)
        mask_key_main = (k_type == 0) & (k_block <= q_block)
        mask_key_mask = (k_type == 1) & (q_type == 1) & (k_block == q_block) & (k_step <= q_step)
        attn_bias.masked_fill_(mask_key_main | mask_key_mask, 0.0)
        new_attention_mask = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        return new_input_ids, new_labels, new_position_ids, new_attention_mask, is_mask     
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

        # 1. 仅加载配置 (不涉及权重，不会报错)
        config = PeftConfig.from_pretrained(adapter_name)
        
        # 2. 初始化 PeftModel 结构
        # 这一步会在你的模型中创建 Adapter 层，但初始化为随机权重/空权重
        # 此时模型结构已改变，且都在 NPU 上
        peft_model = PeftModel(self, config)
        
        # 3. 【核心修复】强制使用 'cpu' 读取权重文件
        # 直接调用底层函数，显式指定 device="cpu"，彻底避开 NPU 的 IO 问题
        print("[Info] Manually loading weights to CPU RAM...")
        adapter_weights = load_peft_weights(adapter_name, device="cpu")
        
        # 4. 将 CPU 上的权重注入到 NPU 模型中
        # set_peft_model_state_dict 会自动处理 dtype 和 device 的转换 (CPU -> NPU)
        print("[Info] Injecting weights into NPU model...")
        set_peft_model_state_dict(peft_model, adapter_weights)
        
        # 5. 合并权重并卸载 Adapter
        print("[Info] Merging and unloading...")
        # merge_and_unload 会将 Adapter 的权重加到 Base Model 上，并移除 Adapter 层
        # 这里的 self 已经被修改了
        peft_model.merge_and_unload()
        
        # 6. 清理内存
        del adapter_weights
        del peft_model
        if torch.npu.is_available():
            torch.npu.empty_cache()
            
        print("Loading model done (Manual Method)")
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        重写 save_pretrained。
        当 Trainer 调用此函数时，我们只保存 diffusion_k_head 的权重。
        """
        import os
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. 保存 Config (保留配置是一个好习惯，不占空间)
        # 这样加载时框架能识别这是个合法的模型目录
        self.config.save_pretrained(save_directory)

        # 2. 提取 diffusion_k_head 的权重
        # 我们手动构建一个只包含 head 的 state_dict
        # 注意：为了方便后续直接 load_state_dict(..., strict=False)，
        # 我们保留完整的 key 前缀 "diffusion_k_head."
        head_state_dict = {}
        for k, v in self.diffusion_k_head.state_dict().items():
            # k 此时是 "weight" 或 "bias"
            # 我们把它变回完整路径 "diffusion_k_head.weight"
            head_state_dict[f"diffusion_k_head.{k}"] = v.cpu()

        # 3. 保存为 pytorch_model.bin
        # 框架通常通过文件名识别权重，保存为标准名称可以让 Trainer 认为保存成功了
        output_file = os.path.join(save_directory, "model_head.bin")
        torch.save(head_state_dict, output_file)
        
        logger.info(f"[SmartSave] 已拦截保存请求，仅保存 Head 权重至: {output_file} (大小: {len(head_state_dict)} keys)")
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