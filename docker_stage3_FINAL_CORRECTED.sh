#!/bin/bash
set -e
set -x

# VITA-Audio-Boost Stage 3 Training Script - FINAL CORRECTED VERSION
# Multiple MCTP Modules Training - Continue from Stage 2 checkpoint
# Includes ALL missing parameters from original VITA-Audio scripts

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]; then
    SEQ_LENGTH=4096  # Balanced for VITA-Audio-Boost training
fi

timestamp="$2"
if [ -z "$timestamp" ]; then
    timestamp=`date +'%Y%m%d_%H%M%S'`
fi

#################################################################################
# DOCKER CONTAINER PATHS
#################################################################################

export ROOT_PATH="/workspace"
export LOCAL_ROOT_PATH="/workspace/temp"
export CODE_PATH="/workspace/VITA-Audio"
export LOCAL_CODE_PATH="/workspace/temp"

# Audio tokenizer path
AUDIO_TOKENIZER_PATH="/workspace/VITA-Audio/models/THUDM/glm-4-voice-tokenizer"

#################################################################################
# Find Stage 2 Checkpoint
#################################################################################

echo "🔍 Finding Stage 2 checkpoint..."

# Try to find from temp file first
if [ -f "/tmp/vita_audio_stage2_checkpoint.txt" ]; then
    STAGE2_CHECKPOINT=$(cat /tmp/vita_audio_stage2_checkpoint.txt)
    echo "Found Stage 2 checkpoint from temp file: $STAGE2_CHECKPOINT"
elif [ -f "/workspace/checkpoints/stage2_vita_boost_mtp1_*/stage_info.txt" ]; then
    STAGE2_CHECKPOINT=$(grep "STAGE2_CHECKPOINT_PATH=" /workspace/checkpoints/stage2_vita_boost_mtp1_*/stage_info.txt | cut -d'=' -f2)
    echo "Found Stage 2 checkpoint from stage_info: $STAGE2_CHECKPOINT"
else
    # Find latest Stage 2 checkpoint directory
    STAGE2_CHECKPOINT=$(find /workspace/checkpoints -name "stage2_vita_boost_mtp1_*" -type d | sort -V | tail -1)
    if [ -n "$STAGE2_CHECKPOINT" ]; then
        STAGE2_CHECKPOINT=$(find $STAGE2_CHECKPOINT -name "checkpoint-*" -type d | sort -V | tail -1)
        echo "Found Stage 2 checkpoint by search: $STAGE2_CHECKPOINT"
    fi
fi

if [ -z "$STAGE2_CHECKPOINT" ] || [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found!"
    echo "Available checkpoints:"
    find /workspace/checkpoints -name "checkpoint-*" -type d | head -10
    exit 1
fi

echo "✅ Using Stage 2 checkpoint: $STAGE2_CHECKPOINT"
MODEL_NAME_OR_PATH="$STAGE2_CHECKPOINT"

#################################################################################
# Memory Optimization (Same as previous stages)
#################################################################################

echo "🧹 GPU Memory Cleanup..."

# Kill all existing processes
pkill -f python || true
pkill -f torchrun || true
sleep 5

# Clear GPU memory
python3 -c "
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f'Cleared GPU {i}')
gc.collect()
"

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=4

#################################################################################
# Environment Setup
#################################################################################

echo "=============================================="
echo "🚀 VITA-Audio-Boost Stage 3 (Multiple MCTP)"
echo "=============================================="
echo "Input Model: Stage 2 checkpoint"
echo "Target: Multiple MCTP Modules Training"
echo "Variant: VITA-Audio-Boost (1 10 4 10 ratios)"
echo "Sequence Length: $SEQ_LENGTH"
echo "Stage: 3 (Multiple MCTP Modules)"
echo "=============================================="

mkdir -p $LOCAL_ROOT_PATH

apt update
apt install -y rsync

echo "Syncing code..."
rsync -avh $CODE_PATH/ $LOCAL_ROOT_PATH/
cd $LOCAL_ROOT_PATH

rm -fr datasets
ln -s $ROOT_PATH/data datasets

# Install exact version from requirements
pip3 install transformers==4.48.3

#################################################################################
# Training Configuration
#################################################################################

DATA_PATH="$LOCAL_ROOT_PATH/configs/custom_training_stage1.yaml"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Configuration file not found: $DATA_PATH"
    exit 1
fi

STAGE3_OUTPUT_DIR="$ROOT_PATH/checkpoints/stage3_vita_boost_mtp10_$timestamp"
mkdir -p $STAGE3_OUTPUT_DIR

cp $DATA_PATH $STAGE3_OUTPUT_DIR/

export HF_HOME="$ROOT_PATH/data/HF_HOME/"
mkdir -p $HF_HOME
export TRITON_CACHE_DIR=$LOCAL_ROOT_PATH
export PYTHONPATH=$PYTHONPATH:$LOCAL_ROOT_PATH/third_party/GLM-4-Voice:$LOCAL_ROOT_PATH/third_party/GLM-4-Voice

#################################################################################
# Use Optimized DeepSpeed Configuration
#################################################################################

cat > $LOCAL_ROOT_PATH/ds_config_zero2_optimized.json << 'EOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": false
}
EOF

echo "✅ Created optimized DeepSpeed ZeRO-2 configuration"

#################################################################################
# Pre-training Verification
#################################################################################

echo "=============================================="
echo "🔍 Pre-training Verification"
echo "=============================================="

# Verify Stage 2 checkpoint exists
if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found: $MODEL_NAME_OR_PATH"
    exit 1
fi

echo "✅ Stage 2 checkpoint found: $MODEL_NAME_OR_PATH"

# Check current GPU memory
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
    cached_gb = torch.cuda.memory_reserved(i) / 1024**3
    print(f'GPU {i}: {props.name}')
    print(f'  Total: {memory_gb:.1f} GB')
    print(f'  Allocated: {allocated_gb:.1f} GB')
    print(f'  Cached: {cached_gb:.1f} GB')
    print(f'  Free: {memory_gb - cached_gb:.1f} GB')
"

# Verify all paths
for path in "$MODEL_NAME_OR_PATH" "$AUDIO_TOKENIZER_PATH" "$CODE_PATH/clear_data/ASR" "$CODE_PATH/clear_data/TTS" "$CODE_PATH/clear_data/speech_qna"; do
    if [ ! -d "$path" ]; then
        echo "❌ ERROR: Path not found: $path"
        exit 1
    fi
done

echo "✅ All pre-training checks passed!"

#################################################################################
# Training Configuration
#################################################################################

NPROC_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29502  # Different port from previous stages

# Distributed backend
DISTRIBUTED_BACKEND="nccl"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

#################################################################################
# CORRECTED VITA-Audio-Boost Stage 3 Training
#################################################################################

echo "=============================================="
echo "🚀 Stage 3: Multiple MCTP Modules Training"
echo "=============================================="
echo "Input Model: Stage 2 VITA-Audio-Boost checkpoint"
echo "Target: Multiple MCTP Modules (MTP10)"
echo "Interval Ratios: 1 10 4 10 (VITA-Audio-Boost)"
echo "Flash Attention: Enabled"
echo "Stage: 3 (Multiple MCTP Modules)"
echo "=============================================="

LOG_FILE="$STAGE3_OUTPUT_DIR/training.log"
exec > >(tee -a $LOG_FILE)
exec 2>&1

echo "Starting VITA-Audio-Boost Stage 3 training at $(date)"

# CORRECTED: Stage 3 training with ALL original parameters
torchrun $DISTRIBUTED_ARGS tools/finetune_sts_v4_48_3.py \
    --log_level "info" \
    --do_train \
    --overwrite_output_dir \
    --config_name ${MODEL_NAME_OR_PATH} \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_tokenizer_path $AUDIO_TOKENIZER_PATH \
    --audio_tokenizer_type "glm4voice" \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --output_dir $STAGE3_OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 8000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 2 \
    --learning_rate 6.00e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length ${SEQ_LENGTH} \
    --gradient_checkpointing True \
    --deepspeed $LOCAL_ROOT_PATH/ds_config_zero2_optimized.json \
    --trust_remote_code False \
    --ddp_timeout 7200 \
    --ddp_backend ${DISTRIBUTED_BACKEND} \
    --attn_implementation flash_attention_2 \
    --seed 42 \
    --data_seed 42 \
    --reset_attention_mask \
    --reset_position_ids \
    --create_attention_mask false \
    --create_attention_mask_2d false \
    --dataloader_num_workers 8 \
    --text-audio-interval-ratio 1 10 4 10 \
    --remove_unused_columns False

#################################################################################
# Post-Training Model Saving
#################################################################################

echo "=============================================="
echo "VITA-Audio-Boost Stage 3 completed at $(date)"
echo "=============================================="

FINAL_CHECKPOINT=$(find $STAGE3_OUTPUT_DIR -name "checkpoint-*" -type d | sort -V | tail -1)

if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $STAGE3_OUTPUT_DIR"
    exit 1
fi

echo "Final checkpoint: $FINAL_CHECKPOINT"

# Create symlink for easy access
ln -sf $FINAL_CHECKPOINT $STAGE3_OUTPUT_DIR/final_checkpoint

# Save stage information
echo "STAGE3_CHECKPOINT_PATH=$FINAL_CHECKPOINT" > $STAGE3_OUTPUT_DIR/stage_info.txt
echo "STAGE3_TIMESTAMP=$timestamp" >> $STAGE3_OUTPUT_DIR/stage_info.txt
echo "STAGE3_COMPLETED=$(date)" >> $STAGE3_OUTPUT_DIR/stage_info.txt
echo "INPUT_MODEL=$MODEL_NAME_OR_PATH" >> $STAGE3_OUTPUT_DIR/stage_info.txt
echo "TARGET_MODEL=VITA-Audio-Boost-MTP10" >> $STAGE3_OUTPUT_DIR/stage_info.txt
echo "INTERVAL_RATIOS=1_10_4_10" >> $STAGE3_OUTPUT_DIR/stage_info.txt
echo "TRAINING_TYPE=MULTIPLE_MCTP" >> $STAGE3_OUTPUT_DIR/stage_info.txt

echo "Checkpoint contents:"
ls -la $FINAL_CHECKPOINT/

echo "=============================================="
echo "🎉 VITA-Audio-Boost Stage 3 COMPLETED!"
echo "=============================================="
echo "✅ Stage 2 → Stage 3: Multiple MCTP Modules"
echo "✅ Multiple MCTP Training: Complete"
echo "✅ Interval Ratios: 1 10 4 10 (VITA-Audio-Boost)"
echo "✅ Flash Attention: Enabled"
echo "✅ All Original Parameters: Included"
echo "Checkpoint: $FINAL_CHECKPOINT"
echo "=============================================="

# Save checkpoint path for next stage
echo $FINAL_CHECKPOINT > /tmp/vita_audio_stage3_checkpoint.txt

echo "🎯 Ready for Stage 4: Supervised Fine-tuning (FINAL)!"

