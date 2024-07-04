CUDA_VISIBLE_DEVICES=4 python main_test.py \
    --model_name llama3-8b-instruct \
    --model_path /home/myt/Models/LLAMA3-8B-Instruct \
    --dataset_name DivSafe \
    --eval_task jailbreak_attack \
    --evaluation