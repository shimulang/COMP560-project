# 🔍 COMP560-project Enhancing VLN-DUET with Multimodal LLMs for Vision-and-Language Navigation (R2R)

This project is part of UNC's **COMP560: Artificial Intelligence** course. We aim to improve performance on the **Vision-and-Language Navigation (VLN)** task by integrating and finetuning a **multimodal LLM (LLAMA3-Vision)** using the R2R dataset.

## 🌐 Project Overview

We build on **VLN-DUET**, a strong baseline for the **Room-to-Room (R2R)** navigation benchmark. Our core idea is to perform **data augmentation using a finetuned LLAMA3-Vision model** to generate high-quality, structured sub-instructions that better align with visual observations.

### 🔧 Frameworks Used
- 🤖 VLN-DUET (baseline navigation model)
- 🔎 **LLAMA3-Vision**, finetuned via **[LLAMA-FACTORY](https://github.com/hiyouga/LLaMA-Factory)**
- 🔁 Designed prompting for sub-instruction decomposition
- 📚 Room-to-Room (R2R) dataset for instruction-following navigation

## Key Contributions
- ✅ Used a **finetuned multimodal LLAMA3** model to perform high-quality sub-instruction generation  
- ✅ Applied **data augmentation** to the R2R dataset with LLM-enhanced and vision-aware instructions  
- ✅ Incorporated augmented data into the training pipeline of **VLN-DUET**, resulting in improved generalization  
- ✅ Achieved **consistent performance gain** on `val_unseen` split, showing better instruction understanding and grounding