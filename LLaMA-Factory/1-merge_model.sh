llamafactory-cli export \
    --model_name_or_path /root/vln/LLaMA-Factory/models/LLM-Research/Meta-Llama-3-70B-Instruct \
    --adapter_name_or_path /root/vln/LLaMA-Factory/saves/llama3-70b/lora/sft  \
    --template llama3 \
    --finetuning_type lora \
    --export_dir /root/vln/LLaMA-Factory/saves/llama3-70b/lora/sft-megred-model \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False \

