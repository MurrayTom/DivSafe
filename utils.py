# from modelscope import snapshot_download
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from modelscope import GenerationConfig
# import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# Model names: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"
#tokenizer = AutoTokenizer.from_pretrained("/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0322/7b_llama", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("/home/myt/Models/Qwen-7B-chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
#model = AutoModelForCausalLM.from_pretrained("/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0322/7b_llama", device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=None).eval()
#model.generation_config = GenerationConfig.from_pretrained("/home/myt/Models/Qwen-7B-chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

#api_key = "sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986"
#api_base = "https://open.api.gu28.top/v1"
api_key = "sk-B0WJgJmOVihkw3sk206eFc9c2118498bB79926CcC1178bAe"
api_base = "https://openkey.cloud/v1"
client = OpenAI(api_key=api_key, base_url=api_base)

def read_json_tolist(filename=""):
    with open(filename, 'r') as file:
        data = json.load(file)
    print(len(data))

    return data


def write_to_json(data, data2, filename="./AdvBench_davinci003_new2.json"):
    i = 0
    my_list = []
    for item in data:
        item["target_responses"].append(data2[i]["target_responses"][0])
        my_list.append(item)
        i+=1
    with open(filename, 'a') as json_file:
        json.dump(my_list, json_file, indent=4)
    return my_list

data = read_json_tolist(filename="/home/myt/Datasets/Eval_benchmark/HarmfulQA-SelfEval/HarmfulQA244-SelfEval-test.json")
print(len(data))
print("------------------------------")

my_list_new = []
i = 0
for item in data:
    if i < 244:
        i+=1
        with open("./HarmfulQA244-SelfEval-test_v2.json", 'a') as json_file:
            json.dump(item, json_file, indent=4)
    else:
        item["target_responses"][0] = item["reference_responses"][0]  + item["target_responses"][0]
        item["target_responses"][1] = item["reference_responses"][0]  + item["target_responses"][1]
        item["target_responses"][2] = item["reference_responses"][0]  + item["target_responses"][2]
        with open("./HarmfulQA244-SelfEval-test_v2.json", 'a') as json_file:
            json.dump(item, json_file, indent=4)