---
base_model: /root/vln/LLaMA-Factory/models/LLM-Research/Meta-Llama-3-70B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/root/vln/LLaMA-Factory/models/LLM-Research/Meta-Llama-3-70B-Instruct](https://huggingface.co//root/vln/LLaMA-Factory/models/LLM-Research/Meta-Llama-3-70B-Instruct) on the r2r_vln_fineturn dataset.
It achieves the following results on the evaluation set:
- Loss: 2.4855

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 16
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 512
- total_eval_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 30
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss |
|:-------------:|:-------:|:----:|:---------------:|
| 1.2702        | 14.5455 | 120  | 1.4376          |
| 0.2911        | 29.0909 | 240  | 2.4855          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.41.2
- Pytorch 2.2.2
- Datasets 2.18.0
- Tokenizers 0.19.1