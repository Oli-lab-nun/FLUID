# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, tokenizer, model_args)
    except Exception as e:
        logger.debug(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                
                if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                    model = model.thinker  # use part of Omni model
        
        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)
    
    # import sys
    # print("\n" + "#"*60)
    # print(">>> [验收关键信息验证 / Acceptance Verification]")
    
    # try:
        # # 1. 动态获取模型实际加载的路径或名称 (从 Config 中读取)
        # real_model_path = getattr(model.config, "_name_or_path", "Unknown")
        # if real_model_path == "Unknown":
        #      real_model_path = getattr(model.config, "name_or_path", "Unknown")
             
    #     print(f">>> 1. Loaded Model Path/Name: {real_model_path}")

    #     # 2. 动态获取模型的类名
    #     model_class_name = model.__class__.__name__
    #     print(f">>> 2. Model Architecture Class: {model_class_name}")

    #     # 3. 验证是否为 OpenPangu 系列 (模糊匹配)
    #     if "pangu" in str(real_model_path).lower() or "pangu" in model_class_name.lower():
    #         print(">>> 3. Status: ✅ Verified 'openPangu' series model loaded.")
    #     else:
    #         print(f">>> 3. Status: ⚠️ Model name/class does not explicitly contain 'pangu' ({model_class_name})")

    #     # 4. 打印设备信息 (验证昇腾 NPU)
    #     # LLaMA-Factory 基于 PyTorch/Ascend，打印第一个参数的设备
    #     try:
    #         # 获取模型当前所在的设备对象
    #         device_obj = next(model.parameters()).device
    #         device_info = str(device_obj)
            
    #         print(f">>> 4. Running Device Address: {device_info}")

    #         # 判断是否为 NPU 设备
    #         if "npu" in device_info:
    #             # 尝试导入 torch_npu 以使用特定 API
    #             try:
    #                 import torch_npu
    #                 # 获取设备索引 (比如 npu:0 中的 0)
    #                 dev_idx = device_obj.index if device_obj.index is not None else 0
                    
    #                 # 【核心代码】获取具体芯片型号 (例如: Ascend910B)
    #                 npu_model_name = torch.npu.get_device_name(dev_idx)
                    
    #                 print(f">>> 5. Hardware Check: ✅ Running on Huawei Ascend NPU.")
    #                 print(f"       Detected Chip Model: {npu_model_name}") # 这里会打印出 910B
                    
    #             except Exception as e_npu:
    #                 # 如果获取详细型号失败，至少确认是在 NPU 上
    #                 print(f">>> 5. Hardware Check: ✅ Running on NPU (Details unavailable: {e_npu})")
            
    #         elif "xla" in device_info:
    #             print(">>> 5. Hardware Check: ✅ Running on TPU/XLA Device.")
    #         else:
    #             print(f">>> 5. Hardware Check: ⚠️ Current device is {device_info} (Not NPU).")

    #     except Exception as e:
    #         print(f">>> 4. Running Device: Could not detect device directly. Error: {e}")

    # except Exception as e:
    #     print(f">>> [!] Verification Logic Error: {e}")
        
    # print("#"*60 + "\n")
    # sys.stdout.flush() # 强制刷新缓冲区，确保写入日志

    if model_args.IS_MDModel:
        import sys
        sys.path.append(os.path.join("/data/mxy/workspace/L-MTP-main"))
        from models import get_md_model
        model = get_md_model(
            base_model=model,
            lora_adapter_path=model_args.lora_adapter_path,
            sampler_state_dict=model_args.sampler_state_dict,
            corrector_state_dict = model_args.corrector_state_dict,
            k_masks = model_args.k_masks,
            is_trainable = is_trainable
        )
        model.set_tokenizer(tokenizer)
    if hasattr(model, "language_model"):
        model = model.language_model
    if hasattr(config, "text_config"):
        config = config.text_config
    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # MSAK MODEL 更新lm_head embedding
    # if model_args.IS_MDModel and is_trainable:
    #     model.lm_head.requires_grad_(True)
    #     model.model.model.base_model.embed_tokens.requires_grad_(True)
    
        
    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
