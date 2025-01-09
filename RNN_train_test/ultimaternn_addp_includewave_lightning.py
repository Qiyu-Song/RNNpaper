import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import netCDF4 as nc
import os, sys

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import copy
from utils import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class UltimateRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, sample_size,
                 A=None, B=None, C=None, lr=1e-3,
                 val_h0_idx_rand=[], val_h0_idx_wave=[],
                 inputs_val_rand=None, targets_val_rand=None,
                 inputs_val_wave=None, targets_val_wave=None):
        super(UltimateRNN, self).__init__()
        if A is None:
            A = np.eye(hidden_size, hidden_size)
        if B is None:
            B = np.eye(hidden_size, input_size)
        if C is None:
            C = np.eye(output_size, hidden_size)
        self.sample_size = sample_size
        self.val_h0_idx_rand = val_h0_idx_rand
        self.val_h0_idx_wave = val_h0_idx_wave
        self.hidden_size = hidden_size

        self.relu = nn.ReLU()
        self.input_layer_linear = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.input_layer_1 = nn.Linear(input_size + hidden_size, 3 * (input_size + hidden_size))
        self.input_layer_2 = nn.Linear(3 * (input_size + hidden_size), 3 * (input_size + hidden_size))
        self.input_layer_3 = nn.Linear(3 * (input_size + hidden_size), hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size, bias=False)

        # Initialize weight matrices and bias vectors for the RNN layer
        self.input_layer_linear.weight.data = torch.from_numpy(
            np.concatenate((copy.deepcopy(A), copy.deepcopy(B)), axis=1))
        nn.init.normal_(self.input_layer_1.weight, mean=0, std=0.001 * torch.std(self.input_layer_linear.weight.data))
        nn.init.normal_(self.input_layer_2.weight, mean=0, std=0.001 * torch.std(self.input_layer_linear.weight.data))
        nn.init.normal_(self.input_layer_3.weight, mean=0, std=0.001 * torch.std(self.input_layer_linear.weight.data))
        nn.init.zeros_(self.input_layer_1.bias)
        nn.init.zeros_(self.input_layer_2.bias)
        nn.init.zeros_(self.input_layer_3.bias)

        # Initialize output layers
        self.output_layer.weight.data = torch.from_numpy(copy.deepcopy(C))

        # initialize hidden state
        self.h0 = nn.Parameter(torch.zeros(sample_size, hidden_size))

        # Initialize lists to store loss values
        self.train_losses = []
        self.valid_losses = []
        self.val_rand_preces = []
        self.val_rand_biases = []
        self.val_wave_preces = []
        self.val_wave_biases = []
        self.pred_rand = None
        self.pred_wave = None

        self.register_buffer("inputs_val_rand", inputs_val_rand)
        self.register_buffer("targets_val_rand", targets_val_rand)
        self.register_buffer("inputs_val_wave", inputs_val_wave)
        self.register_buffer("targets_val_wave", targets_val_wave)

        self.lr = lr

    def forward(self, x, index):
        h = [self.h0[index, :]]
        # print(h[-1].shape, x.shape, index)
        for istep in range(x.shape[0] - 1):
            rnn_input = torch.cat((h[-1], x[istep, :, :]), axis=1)
            h.append(
                self.input_layer_linear(rnn_input) + self.input_layer_3(
                    self.relu(self.input_layer_2(self.relu(self.input_layer_1(rnn_input)))))
            )
        rnnh = torch.stack(h, axis=0)
        out = self.output_layer(rnnh)
        return out

    def training_step(self, batch, batch_idx):
        x, y, idx, y_std = batch
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        y_hat = self(x, idx)
        loss = self.loss_function(y_hat / y_std, y)
        self.log('train_loss', np.sqrt(loss.item()), sync_dist=True)
        self.train_losses.append(np.sqrt(loss.item()))
        return loss

    def validation_step(self, val_batch, batch_idx):
        print("Calling validation_step()")
        val_rand_loss = None
        val_wave_loss = None
        if self.inputs_val_rand is not None:
            pred_rand = self(self.inputs_val_rand, val_h0_idx_rand)
            diff_rand = pred_rand - self.targets_val_rand
            bias_rand = torch.mean(diff_rand, axis=0) / torch.std(self.targets_val_rand, axis=0)
            prec_rand = torch.std(diff_rand, axis=0) / torch.std(self.targets_val_rand, axis=0)
            self.val_rand_preces.append(prec_rand)
            self.val_rand_biases.append(bias_rand)
            val_rand_loss = torch.mean(diff_rand ** 2, axis=0) / torch.var(self.targets_val_rand, axis=0)
        if self.inputs_val_wave is not None:
            pred_wave = self(self.inputs_val_wave, val_h0_idx_wave)
            diff_wave = pred_wave - self.targets_val_wave
            bias_wave = torch.mean(diff_wave, axis=0) / torch.std(self.targets_val_wave, axis=0)
            prec_wave = torch.std(diff_wave, axis=0) / torch.std(self.targets_val_wave, axis=0)
            self.val_wave_preces.append(prec_wave)
            self.val_wave_biases.append(bias_wave)
            val_wave_loss = torch.mean(diff_wave ** 2, axis=0) / torch.var(self.targets_val_wave, axis=0)
        if val_rand_loss is None:
            loss = torch.sqrt(torch.mean(val_wave_loss))
        elif val_wave_loss is None:
            loss = torch.sqrt(torch.mean(val_rand_loss))
        else:
            loss = torch.sqrt(torch.mean(torch.cat((val_rand_loss, val_wave_loss), dim=0)))
        self.log('val_loss', loss.item(), prog_bar=False, logger=False)
        self.valid_losses.append(loss.item())

    def test_step(self, val_batch, batch_idx):
        print("Calling test_step()")
        if self.inputs_val_rand is not None:
            pred_rand = self(self.inputs_val_rand, self.val_h0_idx_rand)
            self.pred_rand = pred_rand
        if self.inputs_val_wave is not None:
            pred_wave = self(self.inputs_val_wave, self.val_h0_idx_wave)
            self.pred_wave = pred_wave

    def loss_function(self, output, target):
        scaled_loss = torch.mean((output - target) ** 2)
        return scaled_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=20000, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step'
                'frequency': 1
            }
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--batchsize", help="batch size", type=int)
    parser.add_argument("--max_epoch", help="max epoch", type=int)
    input_args = parser.parse_args()
    hidden_size = 64

    max_epoch = input_args.max_epoch
    lr = float(input_args.lr)
    lr_str = f"{lr:.0e}"
    batch_size = input_args.batchsize
    print(f"lr={lr_str}, batch_size={batch_size}")
    seq_len = 96 * 2
    spinup = 0
    skip_rand = 48
    skip_wave = 24

    # Two-stage training
    # Stage 1: train with random forcing data
    # Stage 2: train with both random forcing data and coupled-wave forcing data
    # For each stage, first train for h0 only, then train for all parameters
    stage = 1
    # stage = 2
    h0_only = True
    # h0_only = False

    load_h0 = not h0_only
    do_test = (stage == 2) and load_h0

    # define model name
    modeltype = f"rand_4_to_50_seqlen_{seq_len}_skip_{skip_rand}"
    if stage == 2:
        modeltype = modeltype + f"_noisywave1.0_6to20_damp2_seqlen_{seq_len}_skip_{skip_wave}"
    if h0_only:
        modeltype = modeltype + "_h0"
    else:
        modeltype = modeltype + "_all_param"

    # Load starting point model. If none, start from the linear model.
    # The loadmodel variable is the path to a checkpoint file in a previous training step.
    # Current values only work for the authors. Change to correct files before running the code.
    loadmodel = None
    if stage == 1 and (not h0_only):
        loadmodel = './logs/rand_4_to_50_seqlen_192_skip_48_test_h0_1e-04_256/version_0/checkpoints/epoch=999-step=29000.ckpt'
    elif stage == 2:
        if h0_only:
            loadmodel = './logs/rand_4_to_50_seqlen_192_skip_48_all_param_5e-06_128/version_0/checkpoints/epoch=9519-val_loss=0.093.ckpt'
        else:
            loadmodel = './logs/rand_4_to_50_seqlen_192_skip_48_noisywave1.0_6to20_damp2_seqlen_192_skip_24_test_h0_1e-04_256/version_0/checkpoints/epoch=999-step=42000.ckpt'

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')

    # read data
    # These npz files can be generated using extract_data.ipynb from SAM output files
    print('Start reading data...')
    SAMdata = []
    n_rand_exp = 0
    n_wave_exp = 0
    for msine_amp in [4, 6, 8, 10, 15, 20, 30, 40, 50]:
        for iexp in [0, 1]:
            print(f'   reading msine_{msine_amp}_{iexp}')
            data_exp = np.load(f'../data/msinefx{msine_amp}_{iexp}_256_addp.npz')
            SAMdata.append(
                data_struct(data_exp['input_weighted'], data_exp['target_weighted'], str(data_exp['experiment']))
            )
            n_rand_exp += 1

    if stage == 2:
        for wn in range(6, 21):
            for damp in [2]:
                print(f'   reading noisywave_1.0 wavenumber_{wn}_damp_{damp}_1')
                data_exp = np.load(
                    f'../data/wn{wn}_damp{damp}day_noadvectbg_noiselevel_1.0_1_addp.npz')
                SAMdata.append(
                    data_struct(data_exp['input_weighted'], data_exp['target_weighted'], str(data_exp['experiment']))
                )
                n_wave_exp += 1
                print(f'   reading noisywave_1.0 wavenumber_{wn}_damp_{damp}_2')
                data_exp = np.load(
                    f'../data/wn{wn}_damp{damp}day_noadvectbg_noiselevel_1.0_2_addp.npz')
                SAMdata.append(
                    data_struct(data_exp['input_weighted'], data_exp['target_weighted'], str(data_exp['experiment']))
                )
                n_wave_exp += 1

    print("Data loaded.")

    # extract data slices as samples
    inputs_slice = []
    targets_slice = []
    targets_std_slice = []
    val_h0_idx_rand = []
    val_h0_idx_wave = []
    total_samples = 0
    for idx, data in enumerate(SAMdata):
        if idx < n_rand_exp:
            val_h0_idx_rand.append(total_samples)
            skip = skip_rand
        else:
            val_h0_idx_wave.append(total_samples)
            skip = skip_wave
        data.normalize(data.target_std)
        inputs_slice_exp, targets_slice_exp = data.slice_data(seq_len, spinup, skip)
        num_sample_exp = inputs_slice_exp.shape[1]
        inputs_slice_exp = inputs_slice_exp[:, :int(num_sample_exp * 0.8), :]
        targets_slice_exp = targets_slice_exp[:, :int(num_sample_exp * 0.8), :]
        inputs_slice.append(inputs_slice_exp)
        targets_slice.append(targets_slice_exp)
        targets_std_slice_exp = data.target_std.repeat(int(num_sample_exp * 0.8), 1)
        targets_std_slice.append(targets_std_slice_exp)
        data.denormalize(data.target_std)
        total_samples += inputs_slice_exp.shape[1]
    inputs_slice = torch.cat(inputs_slice, axis=1)
    targets_slice = torch.cat(targets_slice, axis=1)
    targets_std_slice = torch.cat(targets_std_slice, axis=0)
    sample_size = targets_slice.shape[1]
    assert sample_size == total_samples
    print(f"inputs_slice.shape = {inputs_slice.shape}")
    print(f"val_h0_idx_rand = {val_h0_idx_rand}")
    print(f"val_h0_idx_wave = {val_h0_idx_wave}")
    train_dataset = TensorDataset(inputs_slice.permute(1, 0, 2),
                                  targets_slice.permute(1, 0, 2),
                                  torch.tensor(np.arange(sample_size)),
                                  targets_std_slice)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    # prepare validation data
    inputs_valid_rand = []
    targets_valid_rand = []
    inputs_valid_wave = []
    targets_valid_wave = []
    if n_rand_exp > 0:
        for idx in range(len(val_h0_idx_rand)):
            data = SAMdata[idx]
            inputs_valid_exp = data.input.unsqueeze(1)
            targets_valid_exp = data.target.unsqueeze(1)
            inputs_valid_rand.append(inputs_valid_exp)
            targets_valid_rand.append(targets_valid_exp)
        inputs_valid_rand = torch.cat(inputs_valid_rand, axis=1)
        targets_valid_rand = torch.cat(targets_valid_rand, axis=1)
        print(f"inputs_valid_rand.shape = {inputs_valid_rand.shape}")
    else:
        inputs_valid_rand = None
        targets_valid_rand = None
    if n_wave_exp > 0:
        for idx in range(len(val_h0_idx_wave)):
            data = SAMdata[idx + len(val_h0_idx_rand)]
            inputs_valid_exp = data.input.unsqueeze(1)
            targets_valid_exp = data.target.unsqueeze(1)
            inputs_valid_wave.append(inputs_valid_exp)
            targets_valid_wave.append(targets_valid_exp)
        inputs_valid_wave = torch.cat(inputs_valid_wave, axis=1)
        targets_valid_wave = torch.cat(targets_valid_wave, axis=1)
        print(f"inputs_valid_wave.shape = {inputs_valid_wave.shape}")
    else:
        inputs_valid_wave = None
        targets_valid_wave = None

    # fake validation dataset. needed because pl code requirement
    valid_dataset = TensorDataset(torch.from_numpy(np.array(val_h0_idx_rand + val_h0_idx_wave)),
                                  torch.from_numpy(np.array(val_h0_idx_rand + val_h0_idx_wave)))
    valid_loader = DataLoader(valid_dataset, batch_size=len(val_h0_idx_rand) + len(val_h0_idx_wave), shuffle=False,
                              num_workers=1, pin_memory=False)

    # train
    # load linear model
    A, B, C = read_params_from_ssm(hidden_size, input_size=40, output_size=41, name_fix="addp_4x")
    # initialize RNN model
    model = UltimateRNN(40, hidden_size, 41, sample_size, A, B, C, lr=lr,
                        val_h0_idx_rand=val_h0_idx_rand, val_h0_idx_wave=val_h0_idx_wave,
                        inputs_val_rand=inputs_valid_rand, targets_val_rand=targets_valid_rand,
                        inputs_val_wave=inputs_valid_wave, targets_val_wave=targets_valid_wave)
    # load previous checkpoint if exists
    if loadmodel is not None:
        checkpoint = torch.load(loadmodel)
        print(checkpoint["state_dict"].keys())
        print(model.state_dict().keys())

        model_state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if name == 'h0' and not load_h0:
                continue
            if name in checkpoint["state_dict"].keys():
                print(name)
                model_state_dict[name] = checkpoint["state_dict"][name]
        model.load_state_dict(model_state_dict)
    if h0_only:
        for param in model.parameters():
            param.requires_grad = False
        model.h0.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    model.input_layer_linear.weight.requires_grad = False
    model.output_layer.weight.requires_grad = False

    # configure log files, validation frequency, and checkpoint callbacks
    logger = CSVLogger(save_dir='./logs/', name=f'{modeltype}_{lr_str}_{batch_size}')
    check_val_every_n_epoch = 20
    if do_test and (not h0_only):
        # setup callbacks to save checkpoints
        best_checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # The metric to monitor
            save_top_k=3,  # Save the top 3 checkpoints
            mode='min',  # 'min' for minimizing (e.g., loss), 'max' for maximizing (e.g., accuracy)
            filename='{epoch}-{val_loss:.3f}'  # Checkpoint filename template
        )
        last_checkpoint_callback = ModelCheckpoint(
            save_last=True,
            filename='last_{epoch}-{val_loss:.3f}'  # Checkpoint filename for the last model
        )
        trainer = pl.Trainer(strategy="ddp", accelerator="gpu", num_nodes=1, max_epochs=max_epoch,
                             log_every_n_steps=1, check_val_every_n_epoch=check_val_every_n_epoch,
                             enable_progress_bar=False, logger=logger, profiler="simple",
                             callbacks=[best_checkpoint_callback, last_checkpoint_callback])
    else:
        # use default: only save last step
        trainer = pl.Trainer(strategy="ddp", accelerator="gpu", num_nodes=1, max_epochs=max_epoch,
                             log_every_n_steps=1, check_val_every_n_epoch=check_val_every_n_epoch,
                             enable_progress_bar=False, logger=logger, profiler="simple")

    # do the training
    trainer.fit(model, train_loader, valid_loader)

    # plot the results
    if not os.path.exists(f'./figures/{modeltype}/'):
        os.mkdir(f'./figures/{modeltype}/')
    if max_epoch > 0:
        train_losses = np.array(model.train_losses)
        train_losses = train_losses.reshape(max_epoch, train_losses.size // max_epoch).mean(axis=1)
        valid_losses = np.array(model.valid_losses)
        if n_rand_exp > 0:
            val_rand_preces = np.array([item.cpu().numpy() for item in model.val_rand_preces])
            val_rand_biases = np.array([item.cpu().numpy() for item in model.val_rand_biases])

            for i in range(len(val_h0_idx_rand)):
                print("*****************************************")
                print(f"Final validation for rand {i}")
                print(f"Precision: {val_rand_preces[-1, i, :]}")
                print(f"Bias: {val_rand_biases[-1, i, :]}")

            plt.figure()
            plt.plot(np.arange(max_epoch), train_losses, label='Training Loss')
            for i in range(len(val_h0_idx_rand) // 2):
                plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch),
                         val_rand_preces[:, i * 2:i * 2 + 2, :].mean(axis=(1, 2)),
                         color='k', alpha=0.1 + 0.8 * i / (len(val_h0_idx_rand) // 2),
                         label=f'Validation Precision Rand Group {i}')
                plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch),
                         abs(val_rand_biases[:, i * 2:i * 2 + 2, :]).mean(axis=(1, 2)),
                         color='r', alpha=0.1 + 0.8 * i / (len(val_h0_idx_rand) // 2),
                         label=f'Validation Bias Rand Group {i}')
            plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch), valid_losses,
                     color='g', label=f'Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curves lr={lr_str} batch_size={batch_size}')
            plt.legend()
            plt.savefig(f'./figures/{modeltype}/loss_ts_rand_{lr_str}_{batch_size}_epoch_{max_epoch}.png')

        if n_wave_exp > 0:
            val_wave_preces = np.array([item.cpu().numpy() for item in model.val_wave_preces])
            val_wave_biases = np.array([item.cpu().numpy() for item in model.val_wave_biases])

            for i in range(len(val_h0_idx_wave)):
                print("*****************************************")
                print(f"Final validation for wave {i}")
                print(f"Precision: {val_wave_preces[-1, i, :]}")
                print(f"Bias: {val_wave_biases[-1, i, :]}")

            plt.figure()
            plt.plot(np.arange(max_epoch), train_losses, label='Training Loss')
            for i in range(len(val_h0_idx_wave)):
                plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch),
                         val_wave_preces[:, i:i + 1, :].mean(axis=(1, 2)),
                         color='k', alpha=0.1 + 0.8 * i / (len(val_h0_idx_wave)),
                         label=f'Validation Precision Wave Exp {i}')
                plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch),
                         abs(val_wave_biases[:, i:i + 1, :]).mean(axis=(1, 2)),
                         color='r', alpha=0.1 + 0.8 * i / (len(val_h0_idx_wave)), label=f'Validation Bias Wave Exp {i}')
            plt.plot(np.arange(0, max_epoch + 1, check_val_every_n_epoch), valid_losses,
                     color='g', label=f'Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curves lr={lr_str} batch_size={batch_size}')
            plt.legend()
            plt.savefig(f'./figures/{modeltype}/loss_ts_wave_{lr_str}_{batch_size}_epoch_{max_epoch}.png')

    if do_test:
        trainer.test(model=model, dataloaders=valid_loader)
        # rand
        if n_rand_exp > 0:
            pred_rand = model.pred_rand.cpu()
            diff_rand = pred_rand - targets_valid_rand
            bias_rand = torch.mean(diff_rand[:, :, :], axis=0) / torch.std(targets_valid_rand, axis=0)
            prec_rand = torch.std(diff_rand[:, :, :], axis=0) / torch.std(targets_valid_rand, axis=0)
            length = inputs_valid_rand.shape[0]
            for icase in range(inputs_valid_rand.shape[1]):
                fig, ax = plt.subplots(21, 2, figsize=(24, 50))
                for i in range(41):
                    ax[i // 2, i % 2].plot(np.arange(length) / 96, targets_valid_rand[:, icase, i], alpha=0.6)
                    ax[i // 2, i % 2].plot(np.arange(length) / 96, pred_rand[:, icase, i].numpy(), alpha=0.6)
                    ax[i // 2, i % 2].grid()
                fig.tight_layout()
                plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_exp_rand_{icase}_pred_target_ts.png')
                plt.close(fig)

                fig, ax = plt.subplots(6, 8, figsize=(18, 10))
                for i in range(41):
                    ax[i // 8, i % 8].scatter(targets_valid_rand[192:, icase, i].numpy(),
                                              diff_rand[192:, icase, i].numpy(), s=0.25, alpha=0.5)
                    ax[i // 8, i % 8].axhline(y=0, color='k', linewidth=2)
                    ax[i // 8, i % 8].axvline(x=0, color='k', linewidth=2)
                    ax[i // 8, i % 8].grid()
                fig.tight_layout()
                plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_exp_rand_{icase}_diff_vs_target.png')
                plt.close(fig)

            fig, ax = plt.subplots(1, inputs_valid_rand.shape[1], figsize=(6 * inputs_valid_rand.shape[1], 5))
            accuracy_p = copy.deepcopy(bias_rand.numpy())
            accuracy_p[accuracy_p < 0] = np.nan
            accuracy_n = copy.deepcopy(bias_rand.numpy())
            accuracy_n[accuracy_n > 0] = np.nan
            accuracy_n *= -1.
            precision = copy.deepcopy(prec_rand.numpy())
            total = torch.mean(diff_rand[192:, :, :] ** 2, axis=(0, 2)) / torch.var(targets_valid_rand, axis=(0, 2))
            total = np.sqrt(total.numpy())
            for icase in range(inputs_valid_rand.shape[1]):
                ax[icase].scatter(np.arange(1, 42), accuracy_p[icase, :] * 100, c='r', label='accuracy +')
                ax[icase].scatter(np.arange(1, 42), accuracy_n[icase, :] * 100, c='b', label='accuracy -')
                ax[icase].scatter(np.arange(1, 42), precision[icase, :] * 100, c='k', label='precision')
                ax[icase].legend()
                ax[icase].grid()
                ax[icase].set_xlabel('Output')
                ax[icase].set_ylabel('Percentation Error (%)')
                ax[icase].set_title(f'Case rand_{icase} | Epoch: {max_epoch} | {round(total[icase] * 100, 2)}%')

            plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_rand_accuracy_and_precision.png')
            plt.close(fig)

        if n_wave_exp > 0:
            # wave
            pred_wave = model.pred_wave.cpu()
            diff_wave = pred_wave - targets_valid_wave
            bias_wave = torch.mean(diff_wave[:, :, :], axis=0) / torch.std(targets_valid_wave, axis=0)
            prec_wave = torch.std(diff_wave[:, :, :], axis=0) / torch.std(targets_valid_wave, axis=0)
            length = inputs_valid_wave.shape[0]
            for icase in range(inputs_valid_wave.shape[1]):
                fig, ax = plt.subplots(21, 2, figsize=(24, 50))
                for i in range(41):
                    ax[i // 2, i % 2].plot(np.arange(length) / 96, targets_valid_wave[:, icase, i], alpha=0.6)
                    ax[i // 2, i % 2].plot(np.arange(length) / 96, pred_wave[:, icase, i].numpy(), alpha=0.6)
                    ax[i // 2, i % 2].grid()
                fig.tight_layout()
                plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_exp_wave_{icase}_pred_target_ts.png')
                plt.close(fig)

                fig, ax = plt.subplots(6, 8, figsize=(18, 10))
                for i in range(41):
                    ax[i // 8, i % 8].scatter(targets_valid_wave[192:, icase, i].numpy(),
                                              diff_wave[192:, icase, i].numpy(), s=0.25, alpha=0.5)
                    ax[i // 8, i % 8].axhline(y=0, color='k', linewidth=2)
                    ax[i // 8, i % 8].axvline(x=0, color='k', linewidth=2)
                    ax[i // 8, i % 8].grid()
                fig.tight_layout()
                plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_exp_wave_{icase}_diff_vs_target.png')
                plt.close(fig)

            fig, ax = plt.subplots(1, inputs_valid_wave.shape[1], figsize=(6 * inputs_valid_wave.shape[1], 5))
            accuracy_p = copy.deepcopy(bias_wave.numpy())
            accuracy_p[accuracy_p < 0] = np.nan
            accuracy_n = copy.deepcopy(bias_wave.numpy())
            accuracy_n[accuracy_n > 0] = np.nan
            accuracy_n *= -1.
            precision = copy.deepcopy(prec_wave.numpy())
            total = torch.mean(diff_wave[:, :, :] ** 2, axis=(0, 2)) / torch.var(targets_valid_wave, axis=(0, 2))
            total = np.sqrt(total.numpy())
            for icase in range(inputs_valid_wave.shape[1]):
                ax[icase].scatter(np.arange(1, 42), accuracy_p[icase, :] * 100, c='r', label='accuracy +')
                ax[icase].scatter(np.arange(1, 42), accuracy_n[icase, :] * 100, c='b', label='accuracy -')
                ax[icase].scatter(np.arange(1, 42), precision[icase, :] * 100, c='k', label='precision')
                ax[icase].legend()
                ax[icase].grid()
                ax[icase].set_xlabel('Output')
                ax[icase].set_ylabel('Percentation Error (%)')
                ax[icase].set_title(f'Case wave_{icase} | Epoch: {max_epoch} | {round(total[icase] * 100, 2)}%')

            plt.savefig(f'./figures/{modeltype}/epoch_{max_epoch}_wave_accuracy_and_precision.png')
            plt.close(fig)
