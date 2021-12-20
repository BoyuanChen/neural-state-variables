#!/bin/bash

dataset=$1
gpu=$2

echo "================================================================================================"
echo "============== Evaluating (Gathering) refine-64 model on: $dataset (gpu id: $gpu) =============="
echo "================================================================================================"

screen -S eval-"$dataset"-refine-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints eval-refine-train NA; \
                                                 CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints eval-refine-train NA; \
                                                 CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints eval-refine-train NA; \
                                                 exec sh";