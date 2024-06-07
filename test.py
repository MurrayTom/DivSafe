from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.ICA_wei_2023 import ICA
from easyjailbreak.attacker.Jailbroken_wei_2023 import Jailbroken
from easyjailbreak.attacker.Multilingual_Deng_2023 import Multilingual
from easyjailbreak.attacker.MCQ import MultipleChioceQuestion, MultipleChioceQuestion_evaluation
from easyjailbreak.attacker.SelfEval import selfevalcheck, selfevalcheck_evaluation, claude3_check_dataset
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.huggingface_model import from_pretrained

model_list = {
    "black_box":[],
    "open_source":[],
    "ours":["llama2-13b-sft","llama2-13b-sft-dpo-DDAPO","mistral-7b-sft-dpo-DDAPO","llama2-7b-sft","llama2-7b-sft-kto", "llama2-13b-sft","mistral-7b-sft","mistral-7b-sft-dpo","mistral-7b-sft-dpo_v2","llama2-7b-sft-dpo","llama2-7b-sft-dpo_epoch2","llama2-7b-sft-dpo_continue","llama2-7b-sft-dpo_continue_v2","llama2-13b-sft-dpo","llama2-13b-sft-dpo_safe","llama2-13b-sft-dpo_v2","llama2-13b-sft-dpo_v3","llama2-13b-sft-dpo_epoch2","mistral-7b-sft-dpo-safe","llama2-7b-sft-dpo-ICC-pipeline","llama2-13b-sft-dpo-ICC-pipeline","mistral-7b-sft-dpo-ICC-pipeline","mistral-7b-instruct-dpo","mistral-7b-instruct-dpo-ICC-pipeline","llama2-7b-chat-dpo"]
}
dataset_list = {
    "AdvBench520": ["AdvBench", "/home/myt/Datasets/Eval_benchmark/AdvBench-MCQ/AdvBench-mcq-test.json", "/home/myt/Datasets/Eval_benchmark/AdvBench-SelfEval/AdvBench-SelfEval-test.json", 520],
    "HarmfulQA244": ["HarmfulQA", "/home/myt/Datasets/Eval_benchmark/HarmfulQA-MCQ/HarmfulQA244-MCQ-test.json", "/home/myt/Datasets/Eval_benchmark/HarmfulQA-SelfEval/HarmfulQA244-SelfEval-test.json", 244],
    "Beaver428": ["Beaver-eval", "/home/myt/Datasets/Eval_benchmark/Beaver-eval-MCQ/Beaver428-MCQ-test.json", "/home/myt/Datasets/Eval_benchmark/Beaver-eval-SelfEval/Beaver428-SelfEval-test.json", 428],
    "SaladBench": ["SaladBench", "/home/myt/Datasets/Eval_benchmark/SaladBench-MCQ/SaladBench-MCQ-test.json", "/home/myt/Datasets/Eval_benchmark/SaladBench-SelfEval/SaladBench-selfeval.json",250],
    "DivSafe": ["DivSafe", "/root/Datasets/Eval_benchmark/DivSafe-MCQ/mcq_test.json", "/root/Datasets/Eval_benchmark/DivSafe-Judge/judge_test.json", 1442]
}

#model_name = "llama2-13b-sft-dpo-DDAPO"
model_name = "llama2-7b-chat"
#model_name = "mistral-7b-sft-dpo_v2"
#model_name = "mistral-7b-sft-dpo-safe"
#model_name = "mistral-7b-sft-dpo-DDAPO"
eval_by_gpt4 = True
dataset_name = "DivSafe"
eval_set = "JB" # Non-adv, JB, MCQ, TFQ
MODE = "attack" # attack, evaluation
prompt_mode = "ToP-ZS"
scene_mode = "single"

if MODE == "attack":
    if "mistral" in model_name or "llama2" in model_name or "LLAMA-Guard" in model_name:
        from easyjailbreak.models.huggingface_model import from_pretrained
    elif "qwen-" in model_name:
        from easyjailbreak.models.modelscope_model import from_pretrained
    elif "chatglm3" in model_name or "internlm2" in model_name:
        from easyjailbreak.models.glm_model import from_pretrained
    elif "qwen1.5" in model_name:
        from easyjailbreak.models.qwen2_model import from_pretrained
else:
    from easyjailbreak.models.huggingface_model import from_pretrained

# First, prepare models and datasets.
if MODE == "attack":
    print(model_name)
    if model_name == "":
        target_model = from_pretrained(model_name_or_path='/home/myt/Models/LLAMA-Guard-7B',
                                   model_name='llama-2')
    elif model_name == "chatgpt":
        # attack_model = OpenaiModel(model_name='gpt-3.5-turbo',
        #                     api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
        target_model = OpenaiModel(model_name='gpt-3.5-turbo',
                            api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
    elif model_name == "text-davinci-003":
        # attack_model = OpenaiModel(model_name='gpt-3.5-turbo',
        #                     api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
        target_model = OpenaiModel(model_name='gpt-3.5-turbo-instruct',
                            api_keys='sk-B0WJgJmOVihkw3sk206eFc9c2118498bB79926CcC1178bAe')
    elif model_name == "gpt4":
        # attack_model = OpenaiModel(model_name='gpt-4-turbo-2024-04-09',
        #                     api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
        target_model = OpenaiModel(model_name='gpt-4-turbo-preview',
                            api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
    elif model_name == "claude3":
        # attack_model = OpenaiModel(model_name='claude-3-haiku-20240307',
        #                     api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
        target_model = OpenaiModel(model_name='claude-3-haiku-20240307',
                            api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
    elif model_name == "llama2-7b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/LLAMA2-7B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/LLAMA2-7B-chat',
                                    model_name='llama-2')
    elif model_name == "llama2-13b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/LLAMA2-13B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/LLAMA2-13B-chat',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-instruct":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/Mistral-7B-instruct',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/Mistral-7B-instruct',
                                    model_name='llama-2')
    elif model_name == "chatglm3-6b":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/ChatGLM-6B',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/chatglm3-6b',
                                    model_name='llama-2')
    elif model_name == "internlm2-7b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/InternLM2-7B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/internlm2-7b-chat',
                                    model_name='llama-2')
    elif model_name == "qwen1.5-7b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/Qwen1.5-7B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/qwen1.5-7b-chat',
                                    model_name='llama-2')
    elif model_name == "qwen1.5-14b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/Qwen1.5-14B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/qwen1.5-14b-chat',
                                    model_name='llama-2')
    elif model_name == "qwen-7b-chat":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/Models/Qwen-7B-chat',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/root/Models/qwen-7b-chat',
                                    model_name='llama-2')

    #--------------------------------------------------------------------------

    elif model_name == "llama2-7b-sft":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0322/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0322/7b_llama',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-sft":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "llama2-7b-sft-dpo":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0325_dpo/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0325_dpo/7b_llama',
                                    model_name='llama-2')
        # target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_dpo/7b_llama',
        #                                model_name='llama-2')
    elif model_name == "llama2-7b-sft-dpo_epoch2":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0325_dpo/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_dpo/7b_llama',
                                    model_name='llama-2')
        # target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_dpo/7b_llama',
        #                                model_name='llama-2')
    elif model_name == "llama2-7b-sft-dpo_continue":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0325_dpo/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0511_dpo_continue/7b_llama',
                                    model_name='llama-2')
        # target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_dpo/7b_llama',
        #                                model_name='llama-2')
    elif model_name == "llama2-7b-sft-dpo_continue_v2":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0325_dpo/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0512_dpo_continue/7b_llama',
                                    model_name='llama-2')
        # target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_dpo/7b_llama',
        #                                model_name='llama-2')
    elif model_name == "mistral-7b-sft-dpo":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-sft-dpo_v2":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0512_dpo_epoch2/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-sft-dpo-safe":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0512_dpo_epoch2_safe/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "llama2-7b-sft-dpo-ICC-pipeline":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0412_dpo_ICC-pipeline/7b_llama',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0412_dpo_ICC-pipeline/7b_llama',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-sft-dpo-ICC-pipeline":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo_ICC-pipeline/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0506_dpo_ICC-pipeline/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "mistral-7b-instruct-dpo":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501_mistral-7B-instruct-dpo/7b_mistral',
        #                            model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501_mistral-7B-instruct-dpo/7b_mistral',
                                model_name='llama-2')
    elif model_name == "mistral-7b-instruct-dpo-ICC-pipeline":
        # attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501_mistral-7B-instruct-dpo-ICC-pipeline/7b_mistral',
        #                                model_name='llama-2')
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0501_mistral-7B-instruct-dpo-ICC-pipeline/7b_mistral',
                                    model_name='llama-2')
    elif model_name == "llama2-7b-chat-dpo":
        target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0503_llama2-7B-chat-dpo/7b_llama',
                                model_name='llama-2')
    else:
        print("model does not implement yet.....")
        exit()
        # target_model = OpenaiModel(model_name='gpt-3.5-turbo',
        #                         api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
elif MODE == "evaluation":
    target_model = OpenaiModel(model_name='gpt-3.5-turbo',
                                api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
elif MODE == "check":
    target_model = OpenaiModel(model_name='gpt-3.5-turbo',
                                api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
else:
    print("The mode is not supported ....")
    exit()


# attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0403_dpo_ICC/7b_llama',
#                                model_name='llama-2')
# target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0403_dpo_ICC/7b_llama',
#                                model_name='llama-2')

# attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0427_dpo_ICC/7b_mistral',
#                                model_name='llama-2')
# target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0427_dpo_ICC/7b_mistral',
#                                model_name='llama-2')

# attack_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_kto/7b_llama',
#                                model_name='llama-2')
# target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0327_kto/7b_llama',
#                                model_name='llama-2')


# target_model = from_pretrained(model_name_or_path='/home/myt/OpenRLHF-main/examples/scripts/ckpt_full_0503_llama2-7B-chat-dpo/7b_llama',
#                                model_name='llama-2')

attack_model = OpenaiModel(model_name='gpt-3.5-turbo',
                         api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
# target_model = OpenaiModel(model_name='gpt-3.5-turbo',
#                          api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
if eval_by_gpt4:
    eval_model = OpenaiModel(model_name='gpt-4-turbo-preview',
                         api_keys='sk-viv3lQLoCNWGY6gj4cE78bF1D077491f9221A0E11d2f2986')
else:
    eval_model = from_pretrained(model_name_or_path='/root/Models/llamaguard-7b',
                                   model_name='llama-2')

dataset = JailbreakDataset('/root/Datasets/Eval_benchmark/'+dataset_list[dataset_name][0])
print(len(dataset))
print(dataset[10])

if eval_set == "Non-adv":
    attacker = Multilingual(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
    if model_name in model_list["ours"]:
        save_path_template = "./outputs_results_benchmark/llm_response/{dataset_name}_ours/{dataset_name}-{model_name}.json"
        eval_path_template = "./outputs_results_benchmark/eval_results/{dataset_name}_ours/{dataset_name}-{model_name}_eval.json"
    else:
        if scene_mode == "single":
            save_path_template = "./outputs_results_benchmark/llm_response/{dataset_name}/{prompt_mode}/{dataset_name}-{model_name}.json"
            eval_path_template = "./outputs_results_benchmark/eval_results/{dataset_name}/{prompt_mode}/{dataset_name}-{model_name}_eval.json"
        else:
            save_path_template = "./outputs_results_benchmark/llm_response/{dataset_name}_{scene_mode}/{prompt_mode}/{dataset_name}-{model_name}.json"
            eval_path_template = "./outputs_results_benchmark/eval_results/{dataset_name}_{scene_mode}/{prompt_mode}/{dataset_name}-{model_name}_eval.json"

    if MODE == "attack":
        save_path = save_path_template.format(dataset_name=dataset_name, scene_mode=scene_mode, prompt_mode=prompt_mode, model_name=model_name)
        attacker.attack(save_path, prompt_mode, scene_mode)
    elif MODE == "evaluation":
        save_path = save_path_template.format(dataset_name=dataset_name, scene_mode=scene_mode, prompt_mode=prompt_mode, model_name=model_name)
        eval_path = eval_path_template.format(dataset_name=dataset_name, scene_mode=scene_mode, prompt_mode=prompt_mode, model_name=model_name)
        attacker.evaluation(input_path = save_path, eval_output_path = eval_path)

elif eval_set == "JB":
# Then instantiate the recipe.
    attacker = Jailbroken(attack_model=attack_model,
                    target_model=target_model,
                    eval_model=eval_model,
                    jailbreak_datasets=dataset)
    if model_name in model_list["ours"]:
        save_path_template = "./outputs_results_benchmark/llm_response_jailbreak/{dataset_name}_ours/{dataset_name}_jailbreak-{model_name}.json"
        eval_path_template = "./outputs_results_benchmark/eval_results_jailbreak/{dataset_name}_ours/{dataset_name}_jailbreak-{model_name}_eval.json"
    else:
        if scene_mode == "single":
            save_path_template = "./outputs_results_benchmark/llm_response_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}.json"
            eval_path_template = "./outputs_results_benchmark/eval_results_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}_eval.json"
        else:
            save_path_template = "./outputs_results_benchmark/llm_response_jailbreak/{dataset_name}_{scene_mode}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}.json"
            eval_path_template = "./outputs_results_benchmark/eval_results_jailbreak/{dataset_name}_{scene_mode}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}_eval.json"

        # save_path_template = "./outputs_results_benchmark/llm_response_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}.json"
        # eval_path_template = "./outputs_results_benchmark/eval_results_jailbreak/{dataset_name}/{prompt_mode}/{dataset_name}_jailbreak-{model_name}_eval.json"

    if MODE == "attack":
        save_path = save_path_template.format(dataset_name=dataset_name, scene_mode=scene_mode, prompt_mode=prompt_mode, model_name=model_name)
        attacker.attack(save_path, prompt_mode, scene_mode, model_name)
    elif MODE == "evaluation":
        save_path = save_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        eval_path = eval_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        print(save_path)
        print(eval_path)
        ASR = attacker.evaluation(input_path = save_path, eval_output_path = eval_path)
        print(dataset_name+"/"+model_name+"/"+eval_set)
        print(str(ASR))

elif eval_set == "MCQ":
    if model_name in model_list["ours"]:
        save_path_template = "./outputs_results_benchmark/llm_response_mcq/{dataset_name}_ours/{dataset_name}_MCQ-{model_name}.json"
        eval_path_template_1 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}_ours/{dataset_name}_MCQ-{model_name}_eval.json"
        eval_path_template_2 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}_ours/{dataset_name}_MCQ-{model_name}_eval-2.json"
    else:
        save_path_template = "./outputs_results_benchmark/llm_response_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}.json"
        eval_path_template_1 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}_eval.json"
        eval_path_template_2 = "./outputs_results_benchmark/eval_results_mcq/{dataset_name}/{prompt_mode}/{dataset_name}_MCQ-{model_name}_eval-2.json"
    if MODE == "attack":
        save_path = save_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        MultipleChioceQuestion(target_model=target_model, test_file=dataset_list[dataset_name][1], save_path=save_path, prompt_mode=prompt_mode)
    elif MODE == "evaluation":
        save_path = save_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        eval_path_1 = eval_path_template_1.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        eval_path_2 = eval_path_template_2.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1 = MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_1)
        false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, format_mode=False)
        print(dataset_name+"/"+model_name+"/"+eval_set)
        print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print(false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1)
        print("--------------------------------------")
        print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print(false_rate_non_format, false_rate_format, false_format_nums, error_format_nums)


elif eval_set == "TFQ":
    if model_name in model_list["ours"]:
        save_path_template = "./outputs_results_benchmark/llm_response_selfeval/{dataset_name}_ours/{dataset_name}_SelfEval-{model_name}.json"
        eval_path_template_1 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}_ours/{dataset_name}_SelfEval-{model_name}_eval.json"
        eval_path_template_2 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}_ours/{dataset_name}_SelfEval-{model_name}_eval-2.json"
    else:
        save_path_template = "./outputs_results_benchmark/llm_response_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}.json"
        eval_path_template_1 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}_eval.json"
        eval_path_template_2 = "./outputs_results_benchmark/eval_results_selfeval/{dataset_name}/{prompt_mode}/{dataset_name}_SelfEval-{model_name}_eval-2.json"
    if MODE == "attack":
        save_path = save_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        selfevalcheck(target_model=target_model, test_file=dataset_list[dataset_name][2], save_path=save_path, prompt_mode=prompt_mode)
    elif MODE == "evaluation":
        save_path = save_path_template.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        eval_path_1 = eval_path_template_1.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        eval_path_2 = eval_path_template_2.format(dataset_name=dataset_name, prompt_mode=prompt_mode, model_name=model_name)
        false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1 = selfevalcheck_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_1, Ts=dataset_list[dataset_name][3])
        false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = selfevalcheck_evaluation(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, Ts=dataset_list[dataset_name][3], format_mode=False)
        print(dataset_name+"/"+model_name+"/"+eval_set)
        print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print(false_rate_non_format_1, false_rate_format_1, false_format_nums_1, error_format_nums_1)
        print("--------------------------------------")
        print("false_rate_non_format / false_rate_format / false_format_nums / error_format_nums")
        print(false_rate_non_format, false_rate_format, false_format_nums, error_format_nums)
    elif MODE == "check":
        save_path = save_path_template.format(dataset_name=dataset_name, model_name=model_name)
        eval_path_1 = eval_path_template_1.format(dataset_name=dataset_name, model_name=model_name)
        eval_path_2 = eval_path_template_2.format(dataset_name=dataset_name, model_name=model_name)
        false_rate_non_format, false_rate_format, false_format_nums, error_format_nums = claude3_check_dataset(eval_model=eval_model, results_file=save_path, save_path=eval_path_2, Ts=dataset_list[dataset_name][3], format_mode=False)



#MultipleChioceQuestion(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/AdvBench-MCQ/AdvBench-mcq-test.json", save_path="./outputs_results_benchmark/llm_response_mcq/AdvBench520/AdvBench_MCQ-chatgpt.json")
#MultipleChioceQuestion(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/HarmfulQA-MCQ/HarmfulQA244-MCQ-test.json", save_path="./outputs_results_benchmark/llm_response_mcq/HarmfulQA244_ours/HarmfulQA244_MCQ-"+model_name+".json")
#MultipleChioceQuestion(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/Beaver-eval-MCQ/Beaver428-MCQ-test.json", save_path="./outputs_results_benchmark/llm_response_mcq/Beaver428/Beaver428_MCQ-chatgpt.json")
# MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file="outputs_results_benchmark/llm_response_mcq/AdvBench520_ours/AdvBench_MCQ-mistral-7b-sft-dpo-ICC-pipeline_v2.json", save_path="./outputs_results_benchmark/eval_results_mcq/AdvBench520_ours/AdvBench_MCQ-mistral-7b-sft-dpo-ICC-pipeline_v2_eval.json")
# MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file="outputs_results_benchmark/llm_response_mcq/AdvBench520_ours/AdvBench_MCQ-mistral-7b-sft-dpo-ICC-pipeline_v2.json", save_path="./outputs_results_benchmark/eval_results_mcq/AdvBench520_ours/AdvBench_MCQ-mistral-7b-sft-dpo-ICC-pipeline_v2_eval-2.json", format_mode=False)
#MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_mcq/HarmfulQA244_ours/HarmfulQA244_MCQ-llama2-7b-chat-dpo.json", save_path="./outputs_results_benchmark/eval_results_mcq/HarmfulQA244_ours/HarmfulQA244_MCQ-llama2-7b-chat-dpo_eval.json")
#MultipleChioceQuestion_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_mcq/HarmfulQA244_ours/HarmfulQA244_MCQ-llama2-7b-chat-dpo.json", save_path="./outputs_results_benchmark/eval_results_mcq/HarmfulQA244_ours/HarmfulQA244_MCQ-llama2-7b-chat-dpo_eval-2.json", format_mode=False)

#selfevalcheck(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/AdvBench-SelfEval/AdvBench-SelfEval-test.json", save_path="./outputs_results_benchmark/llm_response_selfeval/AdvBench520/AdvBench_SelfEval-chatgpt.json")
#selfevalcheck(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/HarmfulQA-SelfEval/HarmfulQA244-SelfEval-test.json", save_path="./outputs_results_benchmark/llm_response_selfeval/HarmfulQA244_ours/HarmfulQA244_SelfEval-"+model_name+".json")
#selfevalcheck(target_model=target_model, test_file="/home/myt/Datasets/Eval_benchmark/Beaver-eval-SelfEval/Beaver428-SelfEval-test.json", save_path="./outputs_results_benchmark/llm_response_selfeval/Beaver428/Beaver428_SelfEval-chatgpt.json")
#selfevalcheck_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_selfeval/AdvBench520/AdvBench_SelfEval-claude3.json", save_path="./outputs_results_benchmark/eval_results_selfeval/AdvBench520/AdvBench_SelfEval-claude3_eval.json")
#selfevalcheck_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_selfeval/AdvBench520/AdvBench_SelfEval-claude3.json", save_path="./outputs_results_benchmark/eval_results_selfeval/AdvBench520/AdvBench_SelfEval-claude3_eval-2.json", format_mode=False)
#selfevalcheck_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_selfeval/HarmfulQA244_ours/HarmfulQA244_SelfEval-llama2-7b-chat-dpo.json", save_path="./outputs_results_benchmark/eval_results_selfeval/HarmfulQA244_ours/HarmfulQA244_SelfEval-llama2-7b-chat-dpo_eval.json")
#selfevalcheck_evaluation(eval_model=eval_model, results_file="./outputs_results_benchmark/llm_response_selfeval/HarmfulQA244_ours/HarmfulQA244_SelfEval-llama2-7b-chat-dpo.json", save_path="./outputs_results_benchmark/eval_results_selfeval/HarmfulQA244_ours/HarmfulQA244_SelfEval-llama2-7b-chat-dpo_eval-2.json", format_mode=False)

# Finally, start jailbreaking.
#attacker.attack()
#attacker.evaluation()