import numpy as np
from numpy.fft import rfft, irfft

from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')
import matplotlib.pyplot as plt

import torch
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from typing import Iterator, Sized, List
from torch.utils.data import DataLoader
import torch
from cvnn_models import ComplexValuedNN, RealValuedNN
from cvnn_data import get_data, GlitchDataset

import matplotlib.pyplot as plt
import os
import sys
import random
sys.path.append('/Users/xangm/OneDrive/repos/antiglitch')
from antiglitch import SnippetNormed

from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard

# Torch setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

torch.backends.cuda.matmul.allow_tf32 = True

torch.cuda.memory._record_memory_history()


def memsum():
    """Prints a summary of the GPU memory usage."""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


# switches to control training
train_complex, test_complex = True, True
train_real, test_real = True, True
train_real_td, test_real_td = False, False

amp = False

# define dataset
NTRAIN = 70000
NTEST = 30000
NBATCH = 4096

rootdir = '/Users/xangm/OneDrive/repos/antiglitch/'
datadir = rootdir + 'data/'

# set seed for data generation - noise, augmentation, etc.
np.random.seed(0)

# get data from datadir
distributions, glitches, ifos, ml_models = get_data(datadir)
print(f"Loaded {len(distributions)} distributions, {len(glitches)} glitches, {ifos} ifos, and {ml_models} ml_models.")
# create training and test datasets
train_data = GlitchDataset(datadir, ifos, ml_models, glitches, distributions,
                           NTRAIN, NTEST, device, True, True, True, 'train', 'complex', None)
test_data = GlitchDataset(datadir, ifos, ml_models, glitches, distributions,
                          NTRAIN, NTEST, device, True, True, True, 'test', 'complex', train_data)

# # print max and mins of training and test data
print(
    f"Train data real + imag maxs: {torch.max(train_data.x_arr.real)}, {torch.max(train_data.x_arr.imag)}")
print(
    f"Test data real + imag maxs: {torch.max(test_data.x_arr.real)}, {torch.max(test_data.x_arr.imag)}")
print(
    f"Train data real + imag mins: {torch.min(train_data.x_arr.real)}, {torch.min(train_data.x_arr.imag)}")
print(
    f"Test data real + imag mins: {torch.min(test_data.x_arr.real)}, {torch.min(test_data.x_arr.imag)}")


# Plot some data

trainitem = train_data.__getitem__(0)
testitem = test_data.__getitem__(0)
trainitem = trainitem[0].cpu().numpy()
testitem = testitem[0].cpu().numpy()

# Get a real snippet
ifo = np.random.choice(ifos)
ml_model = np.random.choice(ml_models)
glitch_num = random.choice(glitches[ifo][ml_model])
glitch_num = glitch_num['num']
# create a SnippetNormed object
snip = SnippetNormed(ifo, ml_model, glitch_num, datadir)
inf = {}
inf['freqs'] = np.linspace(0, 4096, 513)
snip.set_infer(inf)
# get the glitch data in the frequency domain
actual_item = rfft(snip.whts)
# scale it by the training scalings
actual_item_real = (actual_item.real -
                    train_data.tr_x_mean_real) / train_data.tr_x_std_real
actual_item_imag = (actual_item.imag -
                    train_data.tr_x_mean_imag) / train_data.tr_x_std_imag
actual_item = actual_item_real + 1j * actual_item_imag
actual_item = np.abs(actual_item)
# plot them all side by side
plt.figure()
plt.subplot(1, 3, 1)
plt.loglog(np.abs(trainitem))
plt.title('Train')
plt.subplot(1, 3, 2)
plt.loglog(np.abs(testitem))
plt.title('Test')
plt.subplot(1, 3, 3)
plt.loglog(actual_item)
plt.title('Actual')
plt.savefig('train_test_actual.png')


class FastRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, batch_size: int, generator=None) -> None:
        self.data_source = data_source
        self.data_len = len(self.data_source)
        self.batch_size = batch_size
        self.generator = generator or torch.Generator()
        self.epoch_indices = []
        self.set_epoch(0)  # Shuffle at initialization

    def _shuffle_indices(self):
        self.epoch_indices = torch.randperm(
            self.data_len, generator=self.generator).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        # Yield batches of indices using slicing
        return (self.epoch_indices[i:i + self.batch_size] for i in range(0, self.data_len, self.batch_size))

    def __len__(self) -> int:
        # Number of batches in an epoch
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._shuffle_indices()  # Shuffle indices for the new epoch


# Load into DataLoader
tr_sampler = FastRandomSampler(train_data, NBATCH)
te_sampler = FastRandomSampler(test_data, NBATCH)
def collate_fn(x): return tuple(x)


train_loader = DataLoader(
    dataset=train_data, batch_size=NBATCH,   drop_last=False, shuffle=True)
test_loader = DataLoader(
    dataset=test_data, batch_size=NBATCH,   drop_last=False, shuffle=True)
list(train_loader.__iter__())[0][0].shape

# Training loop


def train(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs, model_path, writer, scaler=None):
    epoch_count = 0
    # Load model if it exists
    try:
        # find model with highest epoch number
        modelfn = model_path.split('/')[-1]
        modeldir = '/'.join(model_path.split('/')[:-1]) + '/'
        model_files = os.listdir(modeldir)
        model_files = [f for f in model_files if f.endswith(
            '.pt') and f.startswith(modelfn.split('.')[0])]
        if len(model_files) > 0:
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = modeldir + model_files[-1]
            print(f'Loading model {model_path}')
            model.load_state_dict(torch.load(model_path))
            print('Model loaded')
            epoch_count = int(model_path.split('_')[-1].split('.')[0])
    except:
        print('Model not loaded')
    torch.autograd.set_detect_anomaly(True)
    loss_arr = []
    val_loss_arr = []
    while epoch_count < num_epochs:
        model.train()
        total_training_loss = 0
        for inputs, targets in train_loader:
            # Forward pass
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_training_loss += loss.item()
            # if epoch_count >= 10:
            #     torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        loss_arr.append(total_training_loss/len(train_loader))

        # Test the model
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in test_loader:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                total_loss += val_loss.item()
            val_loss_arr.append(total_loss/len(test_loader))
        print(f"Epoch [{epoch_count+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {total_loss/len(test_loader):.4f}")

        # update the learning rate
        scheduler.step(total_loss/len(test_loader))

        # train_loader.sampler.set_epoch(epoch_count)
        epoch_count += 1

        writer.add_scalars("Loss", {"Train": total_training_loss/len(
            train_loader), "Validation": total_loss/len(test_loader)}, epoch_count)
        writer.add_scalar(
            "Learning rate", optimizer.param_groups[0]['lr'], epoch_count)

        # save the model every 10 epochs
        if epoch_count % 10 == 0 and epoch_count != 0:
            torch.save(model.state_dict(), model_path.split(
                '.')[0] + f'_{epoch_count}.pt')

        writer.flush()
    return loss_arr, val_loss_arr


if train_complex:
    if any([x in locals() for x in ['model', 'criterion', 'optimizer', 'scheduler', 'writer']]):
        del model, criterion, optimizer, scheduler, writer
        torch.cuda.empty_cache()
    # Create an instance of the network

    def get_complex_model():
        return ComplexValuedNN(n_conv_layers=2,
                               conv_filters=[4, 8],
                               conv_kernel_size=[1, 3],
                               n_fc_layers=2,
                               n_fc_units=128,
                               dropout=0.15,
                               pool_size=1)
    model = get_complex_model()
    # put model on device
    with torch.no_grad():
        dummy = torch.zeros(2, 513, dtype=torch.cfloat)
        print("shape before fc:", model.forward(dummy).shape)

    model.to(device)

    # model = torch.compile(model)
    # Print the model architecture
    print(model)
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=8, min_lr=3e-6)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    writer = SummaryWriter("Complex_valued")
    model_path = rootdir + 'notebooks/complex_nn_model.pt'
    scaler = torch.amp.GradScaler(enabled=amp)


torch.cuda.empty_cache()
if train_complex:
    complex_train_loss_arr,  complex_val_loss_arr = train(
        model, criterion, optimizer, scheduler, train_loader, test_loader, 10000, model_path, writer, scaler)
    # save the loss arrays into a single file
    np.savez(rootdir + 'notebooks/complex_loss_arrays.npz',
             train=complex_train_loss_arr, val=complex_val_loss_arr)
    del model, criterion, optimizer, scheduler, writer
    torch.cuda.empty_cache()
    memsum()

if test_complex:
    # Predictions from Complex model
    # Load model
    model = get_complex_model()
    model = torch.compile(model)

    modelfn = model_path.split('/')[-1]
    modeldir = '/'.join(model_path.split('/')[:-1])
    model_files = os.listdir(modeldir)
    model_files = [f for f in model_files if f.endswith(
        '.pt') and f.startswith(modelfn.split('.')[0])]
    if len(model_files) > 0:
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = modeldir + '/' + model_files[-1]

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

if test_complex:
    # Get 10 inputs
    inputs, targets = next(iter(test_loader))
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        outputs = model(inputs[0])

    if type(outputs) == torch.Tensor:
        outputs = outputs.cpu().detach().numpy()
    if type(targets) == torch.Tensor:
        targets = targets[0].cpu().detach().numpy()
    for i in range(0, 10):
        print(f"Prediction: {outputs[i]}")
        print(f"Target: {targets[i]}")
        print("\n")
    del model, inputs, targets, outputs
    torch.cuda.empty_cache()
