#!/bin/bash

dataset=$1
gpu=$2
step=$3

echo "================================================================================================"
echo "======= Long-term hybrid-$step model rollout refine-64 model on: $dataset (gpu id: $gpu) ======="
echo "================================================================================================"

screen -S eval-"$dataset"-longterm-hybridrollout -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py hybrid-$step ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints NA 60; \
                                                              CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py hybrid-$step ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints NA 60; \
                                                              CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py hybrid-$step ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints NA 60; \
                                                              exec sh";