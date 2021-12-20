#!/bin/bash

dataset=$1
gpu=$2

echo "=========================================================================================="
echo "============== Evaluating encoder-decoder model on: $dataset (gpu id: $gpu) =============="
echo "=========================================================================================="

screen -S eval-"$dataset" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config1.yaml ./logs_"$dataset"_encoder-decoder_1/lightning_logs/checkpoints NA eval-eval NA; \
                                       CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config2.yaml ./logs_"$dataset"_encoder-decoder_2/lightning_logs/checkpoints NA eval-eval NA; \
                                       CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config3.yaml ./logs_"$dataset"_encoder-decoder_3/lightning_logs/checkpoints NA eval-eval NA; \
                                       exec sh";