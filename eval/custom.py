from typing import List, Optional, Dict, Any
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from models.model import FLUID


# 1. 使用register_model_api注册模型
@register_model_api(name='my_custom_model')
class MyCustomModel(ModelAPI):
    """自定义模型实现"""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        gsm8k_path: Optional[str] = None,
        _device: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)

        BASE_MODEL_PATH = "/.cache/modelscope/hub/models/FreedomIntelligence/openPangu-Embedded-7B"
        DEVICE = _device


        base_model_skeleton = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,trust_remote_code=True)
        model = FLUID(base_model=base_model_skeleton, k_masks=16).to(device=DEVICE, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,trust_remote_code=True)
        print("2. Updating Adapter...")
        model.update_from_adapter("/saves/huawei/adapter-1")

        model.update_from_adapter("/saves/huawei/adapter-2")

        print("3. Updating Head...")
        model.load_model_for_inference("/saves/huawei/head/2000")

        model.set_tokenizer(tokenizer)
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 3. 实现模型推理逻辑

        # 3.1 处理输入消息
        input_text = self._process_messages(input)
        
        # 3.2 调用您的模型
        response = self._call_model(input_text, config)
        
        # 3.3 返回标准化输出
        return ModelOutput.from_content(
            model=self.model_name,
            content=response
        )

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """将聊天消息转换为文本"""
        # text_parts = []
        # for message in messages:
        #     role = getattr(message, 'role', 'user')
        #     content = getattr(message, 'content', str(message))
        #     text_parts.append(f"{role}: {content}")
        # return '\n'.join(text_parts)
        prompt = getattr(messages[0], 'content', str(messages[0]))
        prompt = [{"role": "user", "content": prompt}]
        return prompt

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        # 2. 应用模板
        text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        """调用您的模型进行推理"""

        generated_ids = self.model.generate_dynamic_kv(
            input_ids=model_inputs.input_ids,
            max_new_tokens=7000,
            block_size=16,              # 最大加速倍数
            confidence_threshold=0.9, # 核心参数
            eos_token_id=self.tokenizer.eos_token_id,
            end_think_token_id = 45982,
            remask=True,
        )
        
        # generated_ids = self.model.generate_dynamic_kv(
        #     input_ids=model_inputs.input_ids,
        #     max_new_tokens=4096,
        #     block_size=16,              # 最大加速倍数
        #     confidence_threshold=0.90, # 核心参数
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     end_think_token_id = 45982,
        #     remask=True,
        # )
        
        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:].tolist()
        final_content = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        return final_content
