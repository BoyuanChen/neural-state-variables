#!/bin/bash

dataset=$1

echo "============================================================================================"
echo "========== Estimating intrinsic dimension from encoder-decoder model on: $dataset =========="
echo "============================================================================================"

screen -S eval-"$dataset"-dimension -dm bash -c "OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config1.yaml model-latent NA; \
                                                 OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config2.yaml model-latent NA; \
                                                 OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config3.yaml model-latent NA; \
                                                 exec sh";