import torch
from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from models.model import FLUID

BASE_MODEL_PATH = "/workspace/saves/FLUID"
DEVICE = "cuda:0"

print("1. Loading Skeleton...")
base_model_skeleton = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,trust_remote_code=True)

model = FLUID(base_model=base_model_skeleton, k_masks=16).to(device=DEVICE, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,trust_remote_code=True)

print("3. Updating Head...")
model.load_model_for_inference("/workspace/saves/head/2000")

model.set_tokenizer(tokenizer)
tokenizer = model.tokenizer



# print(tokenizer.pad_token)
print(model.mask_token_id)
# # prepare the model input
prompt = "The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?"
# prompt = "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
# prompt = "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?"
# prompt = "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?"

messages = [
    {"role": "user", "content": prompt} 
]

# # 2. 应用模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

# 3. 编码输入
# 注意：return_tensors="pt" 返回的是字典，包含 input_ids 和 attention_mask
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f"Input Prompt: {text}")
print("Start generating...")

# 4. 执行并行生成
# 获取 EOS ID (不同模型可能不同，如 tokenizer.eos_token_id 或 model.config.eos_token_id)
eos_id = tokenizer.eos_token_id


generated_ids = model.generate_dynamic_kv(
    input_ids=model_inputs.input_ids,
    max_new_tokens=2048,
    block_size=16,              # 最大加速倍数
    # max_refinement_steps=16, 
    confidence_threshold=0.9, # 核心参数
    eos_token_id=eos_id,
    end_think_token_id = 45982,
    # end_think_token_id = 151668,
    remask=True,
)


# 5. 提取新生成的 token
input_len = model_inputs.input_ids.shape[1]
output_ids = generated_ids[0][input_len:].tolist() 

# 6. 解析内容 (处理 Thinking 标签，如果适用)
try:
    think_end_token = 45982 
    # think_end_token = 151668 
    if think_end_token in output_ids:
        # 找到最后一个出现的索引
        index = len(output_ids) - output_ids[::-1].index(think_end_token)
    else:
        index = 0
except ValueError:
    index = 0

# 7. 解码并打印
# 如果 index > 0，说明有 thinking content，可以分别打印
final_content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip()
think_content = tokenizer.decode(output_ids[:index], skip_special_tokens=False).strip()
print("-" * 40)
print("Think Content:\n", think_content)
print("-" * 40)
print("Generated Content:\n", final_content)
print("-" * 40)

