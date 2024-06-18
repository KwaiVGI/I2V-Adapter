#!/usr/bin/env bash

ROOT=./
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-0}
job_name=${2-i2v_infer}
pretrain_weight=${3-"pretrained_models/stable-diffusion-v1-5"} # path to pretrained SD1.5
first_frame_path=${4-"assets/test.png"}
output_dir=${5-"results/i2v_infer"}
i2v_module_path=${6-"pretrained_models/I2V-Adapter/i2v_module.pth"} # path to pretrained i2v module
infer_config=${7-"configs/inference/infer.yaml"}
pretrained_image_encoder_path=${8-"pretrained_models/IP-Adapter/models/image_encoder"} # path to pretrained IP-Adapter
pretrained_ipadapter_path=${9-"pretrained_models/IP-Adapter/models/ip-adapter-plus_sd15.bin"} # path to pretrained IP-Adapter
################

echo 'start job:' ${job_name}

now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}

export CUDA_VISIBLE_DEVICES=${gpus}
python -W ignore -u inference.py \
    --output ${output_dir} \
    --pretrain_weight ${pretrain_weight} \
    --first_frame_path ${first_frame_path} \
    --prompt "an anime girl with long brown hair hugging a white cat" \
    --height 512 --width 512 \
    --cfg 7.5 --infer_config ${infer_config} \
    --i2v_module_path ${i2v_module_path} \
    --pretrained_image_encoder_path ${pretrained_image_encoder_path} \
    --pretrained_ipadapter_path ${pretrained_ipadapter_path} \
    --neg_prompt "worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, watermark, moles" \
    2>&1 | tee ${LOG_FILE}
