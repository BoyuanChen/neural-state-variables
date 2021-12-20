#!/bin/bash

dataset=$1
gpu=$2

echo "======================================================================================================="
echo "============== Long-term model rollout encoder-decoder model on: $dataset (gpu id: $gpu) =============="
echo "======================================================================================================="

screen -S eval-"$dataset"-longterm-modelrollout -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config1.yaml ./logs_"$dataset"_encoder-decoder_1/lightning_logs/checkpoints NA NA 60; \
                                                             CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config2.yaml ./logs_"$dataset"_encoder-decoder_2/lightning_logs/checkpoints NA NA 60; \
                                                             CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config3.yaml ./logs_"$dataset"_encoder-decoder_3/lightning_logs/checkpoints NA NA 60; \
                                                             exec sh";