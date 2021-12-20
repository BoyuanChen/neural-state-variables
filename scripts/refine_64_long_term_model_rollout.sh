#!/bin/bash

dataset=$1
gpu=$2

echo "================================================================================================="
echo "============== Long-term model rollout refine-64 model on: $dataset (gpu id: $gpu) =============="
echo "================================================================================================="

screen -S eval-"$dataset"-longterm-modelrollout -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints NA 60; \
                                                             CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints NA 60; \
                                                             CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints NA 60; \
                                                             exec sh";