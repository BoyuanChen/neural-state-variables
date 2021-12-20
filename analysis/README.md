
## Physics Evaluation on Known Systems

We provide physics evaluation algorithms extracting physical variables including positions, velocities, and energies from ground truth and predicted frames for the three known systems (single pendulum, rigid double pendulum, and elastic double pendulum).
You can use these algorithms to evaluate the physical accuracy of predictions.

1. Evaluate physical variables from ground truth data.
    ```
    python eval_phys_data.py dataset_name data_filepath
    ```
    The ```dataset_name``` can be single_pendulum, double_pendulum, or elastic_pendulum. The ```data_filepath``` is the respective data filepath, e.g., ```data/single_pendulum``` where ```data``` is your customized data folder. The results will be saved in the ```phys_vars.npy``` file under ```data_filepath```.

2. Evaluate physical variables from long-term predictions via model rollouts and compute physical errors between ground truth data and the predictions.
    ```
    python eval_phys_long_term_pred.py config_filepath pred_save_path
    ```
    The ```config_filepath``` is the model configuration filepath, e.g., ```../configs/single_pendulum/model/config1.yaml```. The ```pred_save_path``` is the path to the predicted images, e.g., ```../scripts/logs_single_pendulum_encoder-decoder_1/prediction_long_term/model_rollout/```. The results will be saved in the ```phys_vars.npy```, ```phys_error.npy```, and ```pixel_error.npy``` files under ```pred_save_path```.

**Note**: ```eval_phys_single_pendulum```,  ```eval_phys_double_pendulum```, and ```eval_phys_elastic_pendulum``` are helper packages for evaluating physical variables for the above command lines.


## Intrinsic Dimension Estimation

1. Navigate to the scripts folder
    ```
    cd ../scripts
    ```
    which is the default directory saving all models' log folders.

2. Estimate intrinsic dimension from model latent vectors using the Levina-Bickel method
    ```
    python ../analysis/eval_intrinsic_dimension.py {config_filepath} model-latent NA
    ```
    or using all methods (Levina-Bickel, MiND-ML, MiND-KL, Hein, CD)
    ```
    python ../analysis/eval_intrinsic_dimension.py {config_filepath} model-latent all-methods
    ```
    The ```config_filepath``` is the model configuration filepath, e.g., ```../configs/single_pendulum/model/config1.yaml```. The results will be saved in the ```intrinsic_dimension.npy``` file (using the Levina-Bickel method) or the ```intrinsic_dimension_all_methods.npy``` file (using all methods) in the ```variables``` subfolder under the model's log folder.

3. Estimate intrinsic dimension from raw data (image pairs) using the Levina-Bickel method
    ```
    python ../analysis/eval_intrinsic_dimension.py {config_filepath} data-image NA
    ```
    or using all methods (Levina-Bickel, MiND-ML, MiND-KL, Hein, CD)
    ```
    python ../analysis/eval_intrinsic_dimension.py {config_filepath} data-image all-methods
    ```
    The ```config_filepath``` is the model configuration filepath, e.g., ```../configs/single_pendulum/model/config1.yaml```. The results will be saved in the ```intrinsic_dimension_image.npy``` file (using the Levina-Bickel method) or the ```intrinsic_dimension_image_all_methods.npy``` file (using all methods) in the ```variables``` subfolder under the model's log folder. Here the model configuration only provides data filepath and test video ids.

**Note**: ```intrinsic_dimension_estimation``` are helper packages providing various intrinsic dimension estimation methods. When choosing the ```all-methods``` option, the MATLAB and MATLAB Engine API for Python should be installed on the machine, see instructions [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

References of MATLAB codes:
```
[1] Gabriele Lombardi (2021). Intrinsic dimensionality estimation techniques (https://www.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques), MATLAB Central File Exchange. Retrieved October 21, 2021.
[2] M. Hein and J.-Y. Audibert, Intrinsic dimensionality estimation of submanifolds in Euclidean space, Proceedings of the 22nd Internatical Conference on Machine Learning (https://www.ml.uni-saarland.de/code/IntDim/IntDim.htm).
```


## Latent Space Regression on Known Systems

1. Navigate to the scripts folder
    ```
    cd ../scripts
    ```
    which is the default directory saving all models' log folders.

2. Regress physical variables from the model latent vectors
    ```
    python ../analysis/eval_regression.py config_filepath NA
    ```
    or from the first ```num_components``` principal components of the model latent vectors
    ```
    python ../analysis/eval_regression.py config_filepath num_components
    ```
    The ```config_filepath``` is the model configuration filepath, e.g., ```../configs/single_pendulum/model/config1.yaml```. The results will be saved in the ```regression_results.npy``` file (with model latent vectors) or the ```regression_results_pca_{num_components}.npy``` file (with first few principal components of model latent vectors) in the ```variables``` subfolder under the model's log folder.