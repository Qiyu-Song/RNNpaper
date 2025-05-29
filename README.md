# RNNpaper

This repo contains code and data for the paper draft "Physically Interpretable Emulation of a Moist Convecting Atmosphere with a Recurrent Neural Network" by Qiyu Song and Zhiming Kuang. Following the steps below, one should be able to reproduce all results included in the paper.

## Step 1: Generate Training Data

We use The System for Atmospheric Modeling (SAM) model to generate our data. The source code of the model with our modifications is in [SAM_v6.11.7/SRC_noisywave/](SAM_v6.11.7/SRC_noisywave/). We ran several groups of experiments with different configurations.

#### A. Random forcing experiments

For the 3 experiments to identify a linear model, we use an ensemble size of 1024. User should use [prm.spinup_1024](SAM_v6.11.7/RCE_randmultsine/prm.spinup_1024) as spinup configuration and these prm files [1024_4_1](SAM_v6.11.7/RCE_randmultsine/prm.run_1024_msinefx4_1), [1024_4_2](SAM_v6.11.7/RCE_randmultsine/prm.run_1024_msinefx4_2), [1024_4_3](SAM_v6.11.7/RCE_randmultsine/prm.run_1024_msinefx4_3) as experimental configurations, and modify [SAM_v6.11.7/SRC_noisywave/domain.f90](SAM_v6.11.7/SRC_noisywave/domain.f90) as:

```fortran
    integer, parameter :: nx_gl = 1024 ! Number of grid points in X
    integer, parameter :: ny_gl = 1024 ! Number of grid points in Y
    integer, parameter :: nsubdomains_x  = 32 ! No of subdomains in x
    integer, parameter :: nsubdomains_y  = 32 ! No of subdomains in y
```

For the majority of experiments, we use an ensemble size of 256. Users can refer to prm files [prm.spinup_256_0](SAM_v6.11.7/RCE_randmultsine/prm.spinup_256_0_), [prm.spinup_256_1](SAM_v6.11.7/RCE_randmultsine/prm.spinup_256_1_) for spinup and `SAM_v6.11.7/RCE_randmultsine/prm.run_256_msinefx*` for experimental configurations. The domain setup should be:

```fortran
    integer, parameter :: nx_gl = 512 ! Number of grid points in X
    integer, parameter :: ny_gl = 512 ! Number of grid points in Y
    integer, parameter :: nsubdomains_x  = 16 ! No of subdomains in x
    integer, parameter :: nsubdomains_y  = 16 ! No of subdomains in y
```

For all these and following SAM experiments, first compile the source code for an executable file and use the [resub.ens](SAM_v6.11.7/RCE_randmultsine/resub.ens) file in the case directory to submit a job to a cluster (current version only reflect the setup on Harvard Cannon cluster).

#### B. Coupled wave forcing experiments

First, go to [SAM_v6.11.7/RCE_noisywave/](SAM_v6.11.7/RCE_noisywave/) and run the spinup experiment. Then use [run_batch_noisywave.sh](SAM_v6.11.7/run_batch_noisywave.sh) to generate case folders for different wavenumbers. We used two different values (1 and 2) in line 5 of that file, therefore having 2 experiments for each wavenumber differed by initial random seeds. Submit all experiments using [submit_exps.sh](SAM_v6.11.7/submit_exps.sh).

After running the experiments, convert .stat files to .nc files using `stat2nc`, which can be compiled in [SAM_v6.11.7/UTIL/](SAM_v6.11.7/UTIL/). Then use [RNN_train_test/extract_data.ipynb](RNN_train_test/extract_data.ipynb) to extract the data for training.

## Step 2: Train the Model

### 2.1 Identify a linear model

```sh
cd linear_model_paper
sbatch identification.run_4x
```

The identified linear model should be in [linear_model_paper/model/](linear_model_paper/model/). Load the model and write the variables **A,B,C,K,NoisePattern** in the `sys` variable to a .txt [file](RNN_train_test/allMtq_addp_4x_64.txt), which will be used in next steps. 

(optional) For a benchmark comparison, a linear response model without memory can be calculated as in [linear_model_paper/get_linear_response_matrix.m](linear_model_paper/get_linear_response_matrix.m), which will write the linear response matrix to another .txt [file](linear_model_paper/linear_response_matrix_star.txt).

### 2.2 Train the RNN model

We perform a 2-stage training for the model. In each stage, first train for `h0` and then for all parameters. Make modifications to [model script](RNN_train_test/ultimaternn_addp_includewave_lightning.py) (line 184-187) and [training script](RNN_train_test/train_ultimaternn_includewave_addp_lightning) (line 23-36) accordingly. Then submit the training by

```sh
cd RNN_train_test
sbatch train_ultimaternn_includewave_addp_lightning
```

The learned model will be a checkpoint file with the lowest validation error. The authors mostly used [this model](RNN_train_test/logs/rand_4_to_50_seqlen_192_skip_48_noisywave1.0_6to20_damp2_seqlen_192_skip_24_all_param_2e-06_256/version_0/checkpoints/epoch=4899-val_loss=0.094.ckpt) for following analysis.

### 2.3 Other training

Train a model without memory: [model script](RNN_train_test/nn_nomemory_seq_includewave_lightning.py), [training script](RNN_train_test/train_nn_nomemory_seq_includewave_lightning)

Train a model with only the random forcing dataset: use the [intermediate model](RNN_train_test/logs/rand_4_to_50_seqlen_192_skip_48_noisywave1.0_6to20_damp2_seqlen_192_skip_24_h0_1e-04_256/version_0/checkpoints/epoch=999-step=42000.ckpt) during normal training

Train a model with only the coupled-wave forcing dataset: [model script](RNN_train_test/ultimaternn_addp_includewave_lightning_wavefirst.py), [training script](RNN_train_test/train_ultimaternn_includewave_addp_lightning_wavefirst), [model](RNN_train_test/logs/noisywave1.0_6to20_damp2_seqlen_192_skip_24_rand_4_to_50_seqlen_192_skip_48_h0_1e-04_256/version_0/checkpoints/epoch=999-step=42000.ckpt)

## Step 3: Analysis

This part is included in several notebooks:

- offline tests: [RNN](RNN_train_test/offline_tests_rnn.ipynb), [model without memory](RNN_train_test/offline_tests_nn_nomemory_seq.ipynb)
- online tests: [RNN](RNN_train_test/coupled_wave_rnn.ipynb), [model without memory](RNN_train_test/coupled_wave_nn_nomemory_seq.ipynb), [fitting initial hidden state for RNN](RNN_train_test/coupled_wave_rnn_fit_h0.ipynb)
  

---

This page is last updated on May 28, 2025. For any questions regarding the code or data, please contact Qiyu Song (qsong@g.harvard.edu).
