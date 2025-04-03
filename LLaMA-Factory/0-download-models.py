# pip install modelscope
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-70B-Instruct', cache_dir='/root/vln/LLaMA-Factory/models', revision='master')
