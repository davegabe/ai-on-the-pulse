#!/bin/bash

# Change the current directory to the directory of the script
cd "$(dirname "$0")"

# Download the pre-trained model
d_model=128
wget https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x${d_model}_pretrain_checkpoint.pth -O units_x${d_model}_pretrain_checkpoint.pth
mv units_x${d_model}_pretrain_checkpoint.pth pretrain_checkpoint.pth

# For each dataset
for dataset_name in "wesad" "dreamer" "hci"
do
  # Fine-tuning (with pre-trained model)
  torchrun --nnodes 1 --nproc-per-node=1  --master_port 4444  run.py \
    --is_training 1 \
    --model_id UniTS_sup_ \
    --model UniTS \
    --task_name anomaly_detection \
    --prompt_num 10 \
    --patch_len 16 \
    --stride 16 \
    --e_layers 3 \
    --d_model ${d_model} \
    --des 'Exp' \
    --itr 1 \
    --lradj finetune_anl \
    --weight_decay 5e-6 \
    --train_epochs 20 \
    --batch_size 64 \
    --acc_it 32 \
    --dropout 0 \
    --debug online \
    --dataset_name ${dataset_name} \
    --project_name ${dataset_name}_units_pretrained_d${d_model}_kfold \
    --pretrained_weight pretrain_checkpoint.pth \
    --clip_grad 100 \
    --task_data_config_path ${dataset_name}.yaml \
    --anomaly_ratio 7 # 7% anomaly ratio
done