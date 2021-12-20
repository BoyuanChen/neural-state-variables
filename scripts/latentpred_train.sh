#!/bin/bash

dataset=$1
gpu=$2

echo "========================================================================================"
echo "================ Training latentpred model on: $dataset (gpu id: $gpu) ================="
echo "========================================================================================"

screen -S train-"$dataset" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/; \
                                        exec sh";