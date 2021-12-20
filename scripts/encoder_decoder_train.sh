#!/bin/bash

dataset=$1
gpu=$2

echo "========================================================================================"
echo "============== Training encoder-decoder model on: $dataset (gpu id: $gpu) =============="
echo "========================================================================================"

screen -S train-"$dataset" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config1.yaml; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config2.yaml; \
                                        CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config3.yaml; \
                                        exec sh";