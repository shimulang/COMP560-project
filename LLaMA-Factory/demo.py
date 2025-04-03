import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, Accelerator

model_id = "/root/vln/LLaMA-Factory/saves/llama3-8b/lora/sft-megred-model/"

# 清空并回收 GPU 内存
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 初始化空模型权重
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_id)

# 推断设备映射，以跨 GPU 分配模型
device_map = infer_auto_device_map(model, max_memory={i: "15GB" for i in range(torch.cuda.device_count())})

# 使用推断的设备映射和 bfloat16 精度重新加载模型
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch.bfloat16)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 初始化 Accelerator
accelerator = Accelerator()

# 准备模型和分词器
model, tokenizer = accelerator.prepare(model, tokenizer)

# 创建示例提示
PLANNER_PROMPT = (
    f"""
    Instructions::
        1) Clarity and Self-containment: Each instruction should be self-contained, providing clear directions...
    Please generate new instructions based on the descriptions provided。
    Begin!
    Generating New Instructions:
    Output:
    """
)

# 将输入提示转换为模型的输入张量
input_ids = tokenizer(PLANNER_PROMPT, return_tensors="pt").input_ids

# 使用 Accelerator 将输入数据移动到适当的设备
input_ids = input_ids.to(accelerator.device)

# 显示设备信息，确保一致性
print("Model device:", next(model.parameters()).device)  # 检查模型参数的设备
print("Input IDs device:", input_ids.device)  # 检查输入数据的设备

# 使用模型的 generate 方法生成文本
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的文本
print(generated_text[len(PLANNER_PROMPT):])
