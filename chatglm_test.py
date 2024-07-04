from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/home/myt/Models/InternLM2-7B-chat", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/myt/Models/InternLM2-7B-chat", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("/home/myt/Models/Qwen1.5-7B-chat", device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True).eval()
# tokenizer = AutoTokenizer.from_pretrained("/home/myt/Models/Qwen1.5-7B-chat", trust_remote_code=True)

# prompt = "Give me a short introduction to large language model."

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# print(text)
# print("-------------------------------")
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)