# Duala

### Introduction

This is the Implementation of《Duala: Dual-Level Alignment of Subjects and Stimuli for Cross-Subject fMRI Decoding》

### Usage

Train the model
```
python train_duala.py --wandb_log  --model_name=subj01_duala --no-multi_subject --subj=1 --num_sessions=1 --multisubject_ckpt=xxx/train_logs/final_multisubject_subj01
```

### Acknowledgement

Part of the code is origin from [MindEye2](https://github.com/MedARC-AI/MindEyeV2). 
