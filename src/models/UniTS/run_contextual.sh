#!/bin/bash

# Change the current directory to the directory of the script
cd "$(dirname "$0")"

# Check if a dataset name is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

# Assign the first argument to the dataset_name variable
dataset_name="$1"
d_model=128

# # Download the pre-trained model
# wget https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x${d_model}_pretrain_checkpoint.pth -O units_x${d_model}_pretrain_checkpoint.pth
# mv units_x${d_model}_pretrain_checkpoint.pth pretrain_checkpoint.pth


# Fine-tuning (with pre-trained model)
torchrun --nnodes 1 --nproc-per-node=1  --master_port 4444  run.py \
  --is_training 1 \
  --model_id UniTS_sup_ \
  --model UniTS \
  --task_name contextual_anomaly_detection \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model ${d_model} \
  --des 'Exp' \
  --itr 1 \
  --lradj finetune_anl \
  --weight_decay 5e-6 \
  --train_epochs 2 \
  --batch_size 128 \
  --acc_it 32 \
  --dropout 0 \
  --debug online \
  --dataset_name ${dataset_name} \
  --project_name ${dataset_name}_units_pret_spec_d${d_model}_kfold \
  --pretrained_weight pretrain_checkpoint.pth \
  --clip_grad 100 \
  --task_data_config_path ${dataset_name}.yaml \
  --anomaly_ratio 15 # 15% anomaly ratio
