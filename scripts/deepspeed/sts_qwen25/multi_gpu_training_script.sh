#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]
then
    SEQ_LENGTH=4096
fi

timestamp="$2"
if [ -z "$timestamp" ]
then
    timestamp=`date +'%Y%m%d_%H'`0000
fi

export ROOT_PATH=/mnt/
export CODE_PATH=${ROOT_PATH}/Zen_S2S/
export LOCAL_ROOT_PATH=/mnt/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/Zen_S2S/

cd ${LOCAL_CODE_PATH}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export DISTRIBUTED_BACKEND="nccl"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NPROC_PER_NODE=2
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34567

OUTPUT_DIR=${ROOT_PATH}/output/debt_collection_training/${timestamp}/
mkdir -p ${OUTPUT_DIR}
cp $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME/"
mkdir -p ${HF_HOME}
export TRITON_CACHE_DIR=${LOCAL_CODE_PATH}
export PYTHONPATH=$PYTHONPATH:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

LOG=${OUTPUT_DIR}/log_training.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
echo ${@}

DATA_PATH=/mnt/Zen_S2S/configs/debt_collection_finetune.yaml
MODEL_NAME_OR_PATH=${LOCAL_CODE_PATH}/models/VITA-MLLM/VITA-Audio-Boost/
AUDIO_TOKENIZER_PATH=${LOCAL_CODE_PATH}/models/THUDM/glm-4-voice-tokenizer

cp ${DATA_PATH} ${OUTPUT_DIR}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS tools/finetune_sts_v4_48_3.py --log_level "info" --do_train --overwrite_output_dir --config_name ${MODEL_NAME_OR_PATH} --tokenizer_name $MODEL_NAME_OR_PATH --model_name_or_path $MODEL_NAME_OR_PATH --audio_tokenizer_path $AUDIO_TOKENIZER_PATH --audio_tokenizer_type "glm4voice" --dataset_name $DATA_PATH --bf16 True --tf32 True --torch_dtype bfloat16 --output_dir $OUTPUT_DIR --num_train_epochs 1 --max_steps 2000 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 32 --save_strategy "steps" --save_steps 500 --save_total_limit 2 --learning_rate 2e-5 --max_grad_norm 1.0 --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 10 --report_to "tensorboard" --model_max_length ${SEQ_LENGTH} --gradient_checkpointing True --deepspeed ${LOCAL_CODE_PATH}/scripts/deepspeed/ds_config_zero3_cpu_offload.json --trust_remote_code False --ddp_timeout 7200 --ddp_backend ${DISTRIBUTED_BACKEND} --attn_implementation flash_attention_2 --seed 42 --data_seed 42 --reset_attention_mask --reset_position_ids --create_attention_mask false --create_attention_mask_2d false --dataloader_num_workers 1 --mtp_model_lr_mult 1.00e1 --text-audio-interval-ratio 1 10 4 10

set +x
