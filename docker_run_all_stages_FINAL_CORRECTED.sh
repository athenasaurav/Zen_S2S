#!/bin/bash
set -e
set -x

# VITA-Audio-Boost Complete Training Pipeline - FINAL CORRECTED VERSION
# Runs all 4 stages automatically with proper checkpoint chaining
# Creates VITA-Audio-Boost model from Qwen2.5-7B-Instruct FROM SCRATCH

echo "=============================================="
echo "🚀 VITA-Audio-Boost Complete Training Pipeline"
echo "=============================================="
echo "Base Model: Qwen2.5-7B-Instruct"
echo "Target: VITA-Audio-Boost (FROM SCRATCH)"
echo "Stages: 4 (Audio-Text → Single MCTP → Multiple MCTP → SFT)"
echo "Architecture: VITA-Audio-Boost (1 10 4 10)"
echo "Memory Strategy: Optimized for 2x H100 80GB"
echo "=============================================="

# Create master log directory
MASTER_LOG_DIR="/workspace/training_logs/complete_pipeline_$(date +'%Y%m%d_%H%M%S')"
mkdir -p $MASTER_LOG_DIR

# Master log file
MASTER_LOG="$MASTER_LOG_DIR/complete_training.log"
exec > >(tee -a $MASTER_LOG)
exec 2>&1

echo "Starting complete VITA-Audio-Boost training pipeline at $(date)"

#################################################################################
# Stage 1: Audio-Text Alignment
#################################################################################

echo ""
echo "=============================================="
echo "🎯 STAGE 1: Audio-Text Alignment"
echo "=============================================="
echo "Starting Stage 1 at $(date)"

if [ ! -f "./docker_stage1_FINAL_CORRECTED.sh" ]; then
    echo "❌ ERROR: docker_stage1_FINAL_CORRECTED.sh not found!"
    exit 1
fi

chmod +x ./docker_stage1_FINAL_CORRECTED.sh

# Run Stage 1
./docker_stage1_FINAL_CORRECTED.sh 4096

# Verify Stage 1 completion
if [ ! -f "/tmp/vita_audio_stage1_checkpoint.txt" ]; then
    echo "❌ ERROR: Stage 1 did not complete successfully!"
    exit 1
fi

STAGE1_CHECKPOINT=$(cat /tmp/vita_audio_stage1_checkpoint.txt)
echo "✅ Stage 1 completed successfully!"
echo "✅ Stage 1 checkpoint: $STAGE1_CHECKPOINT"

#################################################################################
# Stage 2: Single MCTP Module Training
#################################################################################

echo ""
echo "=============================================="
echo "🎯 STAGE 2: Single MCTP Module Training"
echo "=============================================="
echo "Starting Stage 2 at $(date)"

if [ ! -f "./docker_stage2_FINAL_CORRECTED.sh" ]; then
    echo "❌ ERROR: docker_stage2_FINAL_CORRECTED.sh not found!"
    exit 1
fi

chmod +x ./docker_stage2_FINAL_CORRECTED.sh

# Run Stage 2
./docker_stage2_FINAL_CORRECTED.sh 4096

# Verify Stage 2 completion
if [ ! -f "/tmp/vita_audio_stage2_checkpoint.txt" ]; then
    echo "❌ ERROR: Stage 2 did not complete successfully!"
    exit 1
fi

STAGE2_CHECKPOINT=$(cat /tmp/vita_audio_stage2_checkpoint.txt)
echo "✅ Stage 2 completed successfully!"
echo "✅ Stage 2 checkpoint: $STAGE2_CHECKPOINT"

#################################################################################
# Stage 3: Multiple MCTP Modules Training
#################################################################################

echo ""
echo "=============================================="
echo "🎯 STAGE 3: Multiple MCTP Modules Training"
echo "=============================================="
echo "Starting Stage 3 at $(date)"

if [ ! -f "./docker_stage3_FINAL_CORRECTED.sh" ]; then
    echo "❌ ERROR: docker_stage3_FINAL_CORRECTED.sh not found!"
    exit 1
fi

chmod +x ./docker_stage3_FINAL_CORRECTED.sh

# Run Stage 3
./docker_stage3_FINAL_CORRECTED.sh 4096

# Verify Stage 3 completion
if [ ! -f "/tmp/vita_audio_stage3_checkpoint.txt" ]; then
    echo "❌ ERROR: Stage 3 did not complete successfully!"
    exit 1
fi

STAGE3_CHECKPOINT=$(cat /tmp/vita_audio_stage3_checkpoint.txt)
echo "✅ Stage 3 completed successfully!"
echo "✅ Stage 3 checkpoint: $STAGE3_CHECKPOINT"

#################################################################################
# Stage 4: Supervised Fine-tuning (FINAL)
#################################################################################

echo ""
echo "=============================================="
echo "🎯 STAGE 4: Supervised Fine-tuning (FINAL)"
echo "=============================================="
echo "Starting Stage 4 at $(date)"

if [ ! -f "./docker_stage4_FINAL_CORRECTED.sh" ]; then
    echo "❌ ERROR: docker_stage4_FINAL_CORRECTED.sh not found!"
    exit 1
fi

chmod +x ./docker_stage4_FINAL_CORRECTED.sh

# Run Stage 4
./docker_stage4_FINAL_CORRECTED.sh 4096

# Verify Stage 4 completion
if [ ! -f "/tmp/vita_audio_final_model.txt" ]; then
    echo "❌ ERROR: Stage 4 did not complete successfully!"
    exit 1
fi

FINAL_MODEL_PATH=$(cat /tmp/vita_audio_final_model.txt)
echo "✅ Stage 4 completed successfully!"
echo "✅ Final model: $FINAL_MODEL_PATH"

#################################################################################
# Training Pipeline Completion
#################################################################################

echo ""
echo "=============================================="
echo "🎉🎉🎉 COMPLETE TRAINING PIPELINE FINISHED! 🎉🎉🎉"
echo "=============================================="
echo "Training completed at $(date)"
echo ""
echo "🎯 TRAINING SUMMARY:"
echo "   Base Model: Qwen2.5-7B-Instruct"
echo "   Final Model: VITA-Audio-Boost"
echo "   Training Type: FROM SCRATCH"
echo "   Total Stages: 4"
echo "   Architecture: VITA-Audio-Boost (1 10 4 10)"
echo ""
echo "📊 STAGE PROGRESSION:"
echo "   Stage 1: $STAGE1_CHECKPOINT"
echo "   Stage 2: $STAGE2_CHECKPOINT"
echo "   Stage 3: $STAGE3_CHECKPOINT"
echo "   Stage 4: $FINAL_MODEL_PATH"
echo ""
echo "🎯 FINAL MODEL LOCATION:"
echo "   $FINAL_MODEL_PATH"
echo ""
echo "📋 MODEL CAPABILITIES:"
echo "   ✅ Audio-Text Alignment"
echo "   ✅ Single MCTP Module"
echo "   ✅ Multiple MCTP Modules"
echo "   ✅ Supervised Fine-tuning"
echo "   ✅ VITA-Audio-Boost Architecture"
echo "   ✅ Flash Attention Optimized"
echo "   ✅ Cross-modal Token Generation"
echo ""
echo "🎊 YOUR VITA-Audio-Boost MODEL IS READY!"
echo "=============================================="

# Create final summary file
SUMMARY_FILE="$MASTER_LOG_DIR/training_summary.txt"
cat > $SUMMARY_FILE << EOF
VITA-Audio-Boost Training Summary
=================================

Training Completed: $(date)
Base Model: Qwen2.5-7B-Instruct
Final Model: VITA-Audio-Boost
Training Type: FROM SCRATCH
Total Stages: 4
Architecture: VITA-Audio-Boost (1 10 4 10)

Stage Progression:
- Stage 1 (Audio-Text Alignment): $STAGE1_CHECKPOINT
- Stage 2 (Single MCTP Module): $STAGE2_CHECKPOINT  
- Stage 3 (Multiple MCTP Modules): $STAGE3_CHECKPOINT
- Stage 4 (Supervised Fine-tuning): $FINAL_MODEL_PATH

Final Model Location: $FINAL_MODEL_PATH

Model Capabilities:
✅ Audio-Text Alignment
✅ Single MCTP Module
✅ Multiple MCTP Modules  
✅ Supervised Fine-tuning
✅ VITA-Audio-Boost Architecture
✅ Flash Attention Optimized
✅ Cross-modal Token Generation

Training Logs: $MASTER_LOG_DIR
EOF

echo "📄 Training summary saved to: $SUMMARY_FILE"
echo ""
echo "🎉 VITA-Audio-Boost training pipeline completed successfully!"
echo "🎯 Your model is ready for inference and deployment!"

