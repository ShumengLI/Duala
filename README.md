# Duala

### Introduction

This is the implementation of《Duala: Dual-Level Alignment of Subjects and Stimuli for Cross-Subject fMRI Decoding》

### Usage

To train the model, run
```
python train_duala.py --wandb_log  --model_name=subj01_duala --no-multi_subject --subj=1 --num_sessions=1 --multisubject_ckpt=xxx/train_logs/final_multisubject_subj01
```

The pretrained models (final_multisubject_subj0x) are adopted from [MindEye2](https://huggingface.co/datasets/pscotti/mindeyev2).

### Model

The fine-tuned Duala model is available at [Duala](https://huggingface.co/ShumengLI/Duala/tree/main).

### Acknowledgement

Part of the code is adapted from [MindEye2](https://github.com/MedARC-AI/MindEyeV2). 

