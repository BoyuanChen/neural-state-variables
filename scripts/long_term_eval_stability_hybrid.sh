#!/bin/bash

dataset=$1
gpu=$2
step=$3

echo "============================================================================================"
echo "========= Evaluating stability of long term prediction on: $dataset (gpu id: $gpu) ========="
echo "============================================================================================"

screen -S eval-"$dataset"-long-term-stab -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_latent-prediction_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/prediction_long_term/hybrid_rollout_$step/ 60; \
                                                      CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_latent-prediction_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/prediction_long_term/hybrid_rollout_$step/ 60; \
                                                      CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_latent-prediction_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/prediction_long_term/hybrid_rollout_$step/ 60; \
                                                      exec sh";