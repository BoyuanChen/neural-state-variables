#!/bin/bash

dataset=$1
gpu=$2

echo "========================================================================================================="
echo "============== Evaluating (Gathering) encoder-decoder-64 model on: $dataset (gpu id: $gpu) =============="
echo "========================================================================================================="

screen -S eval-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA eval-train NA; \
                                          CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA eval-train NA; \
                                          CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA eval-train NA; \
                                          exec sh";