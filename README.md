# Discovering State Variables Hidden in Experimental Data

[Boyuan Chen](http://boyuanchen.com/),
[Kuang Huang](https://cm3.apam.columbia.edu/people/kuang-huang),
[Sunand Raghupathi](https://www.linkedin.com/in/sunand-raghupathi?trk=public_profile_browsemap),
[Ishaan Chandratreya](https://www.linkedin.com/in/ishaan-chandratreya-546aa6141),
[Qiang Du](https://www.engineering.columbia.edu/faculty/qiang-du),
[Hod Lipson](https://www.hodlipson.com/)
<br>
Columbia University
<br>

### [Project Website](https://www.cs.columbia.edu/~bchen/neural-state-variables) | [Video](https://youtu.be/KWHXchlJzSw) | [Paper](http://arxiv.org/abs/2112.10755)

## Overview
This repo contains the PyTorch implementation for paper "Discovering State Variables Hidden in Experimental Data".

![teaser](figures/teaser.gif)

## Citation

If you find our paper or codebase helpful, please consider citing:

```
@article{chen2021discover,
  title={Discovering State Variables Hidden in Experimental Data},
  author={Chen, Boyuan and Huang, Kuang and Raghupathi, Sunand and Chandratreya, Ishaan and Du, Qiang and Lipson, Hod},
  journal={arXiv preprint arXiv:2112.10755},
  year={2021}
}
```

## Content

- [Installation](#installation)
- [Logging](#logging)
- [Data Preparation](#data-preparation)
- [Training and Testing](#training-and-testing)
- [Intrinsic Dimension Estimation](#intrinsic-dimension-estimation)
- [Long-term Prediction and Stability Evaluation](#long-term-prediction-and-stability-evaluation)
- [Evaluation and Analysis](#evaluation-analysis)
- [License](#license)

## Installation

The installation has been test on Ubuntu 18.04 with CUDA 11.0. All the experiments are performed on one GeForce RTX 2080 Ti Nvidia GPU.

Create a python virtual environment and install the dependencies.
```
virtualenv -p /usr/bin/python3.6 env3.6
source env3.6/bin/activate
pip install -r requirements.txt
```
**Note**: You may need to install Matlab on your computer to use some of the data collectors and intrinsic dimension estimation algorithms.

## Logingg

We first introduce the naming convention of the saved files so that it is clear what will be saved and where they will be saved.

1. Log folder naming convention:
    ```
    logs_{dataset}_{model_name}_{seed}
    ```
2. Inside the logs folder, the structure and contents are:
    ```
    \logs_{dataset}_{model_name}_{seed}
        \lightning_logs
            \checkpoints               [saved checkpoint]
            \version_0                 [training stats]
            \version_1                 [testing stats]
        \predictions                   [testing predicted images]
        \prediction_long_term          [long term predicted images]
        \variables                     [file id and latent vectors on testing data]
        \variables_train               [file id and latent vectors on training data]
        \variables_val                 [file id and latent vectors on validation data]
    ```

## Data Preparation

We provide nine datasets with their own download links below.

- [circular_motion](https://drive.google.com/file/d/19DFYqh08B2L-YX_bTmpmxYu2jpARqoUA/view?usp=sharing) (circular motion system)
- [reaction_diffusion](https://drive.google.com/file/d/1LOqpB0l86lELEegX5sV9NwHkGKeHoaTN/view?usp=sharing) (reaction diffusion system)
- [single_pendulum](https://drive.google.com/file/d/1rw6vrVcV3KCIaVVwKMMhoZ9Jph5u-Zdf/view?usp=sharing) (single pendulum system)
- [double_pendulum](https://drive.google.com/file/d/1QEtk4JjnRysIEjtkZIKBjACu_IiTAdX6/view?usp=sharing) (rigid double pendulum system)
- [elastic_pendulum](https://drive.google.com/file/d/1Y8vzawQZhzPHp6cpLFuyM2LHuhgLjH9H/view?usp=sharing) (elastic double pendulum system)
- [swingstick_non_magnetic](https://drive.google.com/file/d/1BfeGW4XTFyGdyBO0G_YnSnRyJGu2WRnc/view?usp=sharing) (swing stick system)
- [air_dancer](https://drive.google.com/file/d/163KvwevY1fnDnI6WiWpc5Zaz-WWwPaAS/view?usp=sharing) (air dancer system)
- [lava_lamp](https://drive.google.com/file/d/1R-I2CZaJLe2D4H818-n25l54R0pbKQfo/view?usp=sharing) (lava lamp system)
- [fire](https://drive.google.com/file/d/1OIH6SalwPyD_lkXBLZq6bzmKfrp-xhrI/view?usp=sharing) (fire system)

Save the downloaded dataset as ```data/{dataset_name}```, where ```data``` is your customized dataset folder. **Please make sure that ```data``` is an absolute path and you need to change the ```data_filepath``` item in the ```config.yaml``` files in ```configs``` to specify your customized dataset folder**.

Please refer to the [datainfo](datainfo) folder for more details about data structure and dataset collection process.


## Training and Testing

Our approach involves three models:
- dynamics predictive model (encoder-decoder / encoder-decoder-64)
- latent reconstruction model (refine-64)
- neural latent dynamics model (latentpred)

1. Navigate to the scripts folder
    ```
    cd scripts
    ```

2. Train the dynamics predictive model (encoder-decoder and encoder-decoder-64) and then save the high-dimensional latent vectors from the testing data.
    ```
    ./encoder_decoder_64_train.sh {dataset_name} {gpu no.}
    ./encoder_decoder_train.sh {dataset_name} {gpu no.}
    ./encoder_decoder_64_eval.sh {dataset_name} {gpu no.}
    ./encoder_decoder_eval.sh {dataset_name} {gpu no.}
    ```

3. Run forward pass on the trained encoder-deocer model and encoder-decoder-64 model to save the high-dimensional latent vectors from the training and validation data. The saved latent vectors will be used as the training and validataion data for training and validating the latent reconstruction model.
    ```
    ./encoder_decoder_eval_gather.sh {dataset_name} {gpu no.}
    ./encoder_decoder_64_eval_gather.sh {dataset_name} {gpu no.}
    ```

4. Before you proceed this step, please refer to the next section to obtain the system's intrinsic dimension and then come back to this step. 

    Train the latent reconstruction model (refine-64) with the saved 64-dim latent vectors from previous steps. Then save the obtained Neural State Variables from both training and testing data.
    ```
    ./refine_64_train.sh {dataset_name} {gpu no.}
    ./refine_64_eval.sh {dataset_name} {gpu no.}
    ./refine_64_eval_gather.sh {dataset_name} {gpu no.}
    ```

5. Train the neural latent dynamics model (latentpred) with the trained models from previous steps.
    ```
    ./latentpred_train.sh single_pendulum {dataset_name} {gpu no.}
    ```

## Intrinsic Dimension Estimation

With the trained dynamics predictive model, our approach provides subroutines to estimate the system's intrinsic dimension (ID) using manifold learning algorithms. The estimated intrinsic dimension will be used to decide the number of Neural State Variables and to design the latent reconstruction model. Only after this step you can proceed to train the latent reconstruction model to obtain Neural State Variables.

1. Navigate to the scripts folder
    ```
    cd scripts
    ```
    which is the default directory saving all models' log folders.

2. Estimate the intrinsic dimension from the saved latent vectors of the encoder-decoder models for all random seeds. 
    ```
    ./encoder_decoder_estimate_dimension.sh {dataset_name}
    ```

3. Calculate the final intrinsic dimension estimated values (mean and standard deviation).
    ```
    python ../utils/dimension.py {dataset_name}
    ```


## Long-term Prediction and Stability Evaluation

With all above trained models, our approach offers system long-term predictions through model rollouts as well as stability evaluation of the long-term predictions.

1. Navigate to the scripts folder
    ```
    cd scripts
    ```

2. Long-term prediction with single model rollouts.
    ```
    ./encoder_decoder_long_term_model_rollout.sh {dataset_name} {gpu no.}
    ./encoder_decoder_64_long_term_model_rollout.sh {dataset_name} {gpu no.}
    ./refine_64_long_term_model_rollout.sh {dataset_name} {gpu no.}
    ```
    The predictions will be saved in the ```prediction_long_term``` subfolder under the model's log folder.

3. Long-term prediction with hybrid model rollouts.
    ```
    ./refine_64_long_term_hybrid_rollout.sh {dataset_name} {gpu no.} {step}
    ```
    where ```step``` is the number of model rollouts via 64-dim latent vectors before a model rollout via Neural State Variables.

    The predictions will be saved in the ```prediction_long_term``` subfolder under the refine-64 model's log folder.

4. Long-term prediction with single model rollouts from perturbed initial frames.
    ```
    ./encoder_decoder_long_term_model_rollout_perturb_all.sh {dataset_name} {gpu no.}
    ./encoder_decoder_64_long_term_model_rollout_perturb_all.sh {dataset_name} {gpu no.}
    ./refine_64_long_term_model_rollout_perturb_all.sh {dataset_name} {gpu no.}
    ```
    
    The predictions will be saved in the ```prediction_long_term``` subfolder under the model's log folder.

3. Stability evaluation on long-term predictions with single model rollouts
    ```
    ./long_term_eval_stability.sh {dataset_name} {gpu no.}
    ```
    and on long-term predictions with hybrid model rollouts
    ```
    ./long_term_eval_stability_hybrid.sh {dataset_name} {gpu no.} {step}
    ```
    where ```step``` is as mentioned above for hybrid model rollouts.
    
    The evaluated latent space errors measuring the prediction stability will be saved in the ```stability.npy``` file under the respective ```prediction_long_term``` folders.

**Note**: The default long-term prediction length is 60 (in frames). You will need to modify the scripts if you want to use a different prediction length.


## Evaluation and Analysis

Please refer to the [analysis](analysis) folder for detailed instructions for physical evaluation and analysis.

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
