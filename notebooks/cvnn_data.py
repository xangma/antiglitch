import os
import re
import json
import numpy as np
from collections import defaultdict

from numpy.fft import rfft, irfft
from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')

import torch

import antiglitch
from antiglitch.utils import to_fd

rng = np.random.default_rng(seed=0)

def get_data(datadir):
	# Load data
	# files are named as follows: ifo-key-num.npz,
	# e.g. L1-lowblip-0355.npz
	# Find all ifos, keys, and numbers in the data directory
	distributions, glitches = None, None
	ifos = []
	ml_models = []
	numbers = []
	# filename pattern: ifo-key-num.npz
	filename_pattern = re.compile(r'([A-Z0-9]+)-([a-z]+)-(\d+).npz')
	for file in os.listdir(datadir):
		# Get unique ifos:
		try:
			# check filename format
			if filename_pattern.match(file):
				ifo = file.split('-')[0]
				if ifo not in ifos:
					ifos.append(ifo)
				# Get unique keys:
				ml_model = file.split('-')[1]
				if ml_model not in ml_models:
					ml_models.append(ml_model)
		except:
			pass
	datadict={}
	for ifo in ifos:
		datadict[ifo] = {}
		for ml_model in ml_models:
			datadict[ifo][ml_model] = []
			for file in os.listdir(datadir):
				try:
					ifo_file = file.split('-')[0]
					key_file = file.split('-')[1]
					num = file.split('-')[2].split('.')[0]
					if ifo_file == ifo and key_file == ml_model:
						datadict[ifo][ml_model].append(int(num))
				except:
					pass
	# load results
	resultsjson=datadir + 'all_PE_v3.json'
	with open(resultsjson, 'r') as f:
		results = json.load(f)
	# load all glitches into an array using the datadict and Snippet class
	glitches = {}
	for ifo in ifos:
		if ifo not in glitches:
			glitches[ifo] = {}
		for ml_model in ml_models:
			if ml_model not in glitches[ifo]:
				glitches[ifo][ml_model] = {}
			for num in datadict[ifo][ml_model]:
				# try:
					if num not in glitches[ifo][ml_model]:
						glitches[ifo][ml_model][num] = {}
					snip = antiglitch.SnippetNormed(ifo, ml_model, num, datadir)
					glitches[ifo][ml_model][num]['snip'] = snip
					glitches[ifo][ml_model][num]['data'] = to_fd(snip.whts)
					glitches[ifo][ml_model][num]['invasd'] = snip.invasd
					glitches[ifo][ml_model][num]['psd'] = np.load(datadir + ifo + '-'+ ml_model + '-' + f'{num:04d}' + '.npz')['psd']
				# except:
				#     pass
	ml_label_map = {'Blip_Low_Frequency':'lowblip', 'Blip':'blip', 'Koi_Fish':'koi', 'Tomte':'tomte'}
	results_keys = ['f0', 'f0_sd', 'gbw', 'gbw_sd', 'amp_r', 'amp_r_sd', 'amp_i', 'amp_i_sd', 'time', 'time_sd', 'num']
	nums = list(results['num'].values())
	for i in range(len(nums)):
		try:
			stri = str(i)
			ifo = results['ifo'][stri]
			ml_label = ml_label_map[results['ml_label'][stri]]
			glitches[ifo][ml_label][nums[i]]['snip'].inf = {}
			for key in results_keys:
				glitches[ifo][ml_label][nums[i]][key] = results[key][stri]
				glitches[ifo][ml_label][nums[i]]['snip'].inf[key] = results[key][stri]
		except:
			pass
	# Step 1: Restructure the dataset (Yes I know it was originally in this format, you leave me alone)
	distributions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	for ifo in glitches:
		for ml_model in glitches[ifo]:
			for glitch_num in glitches[ifo][ml_model]:
				for key in results_keys:
					distributions[ifo][ml_model][key].append(glitches[ifo][ml_model][glitch_num][key])
	return distributions, glitches, ifos, ml_models

def noise_function(x):
	noise_generator_function = lambda x: np.random.normal(0, 1, len(x)) + 1j * np.random.normal(0, 1, len(x))
	noise = np.array([noise_generator_function(x) for x in x])
	x += noise
	return x

def phase_shift(x, shift=np.pi):
	"""Multiply by e^i*shift"""
	phase_shifts = []
	for i in range(len(x)):
		phase_shift = rng.choice([-np.pi, 0, np.pi],)
		if phase_shift != 0:
			x[i] = x[i] * np.exp(1j * phase_shift)
		phase_shifts.append(phase_shift)
	return x, phase_shifts

def time_shift(x, shift=0):
	"""Multiply by e^-j*2*np.pi*f*t"""
	time_shifts = []
	farray = np.linspace(0, 4096, 513) * 4096
	for i in range(len(x)):
		time_shift = rng.uniform(-0.1, 0.1)
		x[i] = x[i] * np.exp(-1j * farray * time_shift)
		time_shifts.append(time_shift)
	return x, time_shifts

def normalize(x, dryrun=False):
	# check if complex or real
	if np.iscomplexobj(x):
		real_parts = np.real(x)
		imag_parts = np.imag(x)
		x_mean_real = np.mean(real_parts)
		x_std_real = np.std(real_parts)
		x_mean_imag = np.mean(imag_parts)
		x_std_imag = np.std(imag_parts)
		normalized_real = (real_parts - x_mean_real) / x_std_real
		normalized_imag = (imag_parts - x_mean_imag) / x_std_imag
		print(f"x_mean_real: {x_mean_real}, x_std_real: {x_std_real}, x_mean_imag: {x_mean_imag}, x_std_imag: {x_std_imag}")
		if not dryrun:
			x = normalized_real + 1j * normalized_imag
		return x, x_mean_real, x_std_real, x_mean_imag, x_std_imag
	else:
		x_mean = x.mean()
		x_std = x.std()
		if not dryrun:
			x = (x - x_mean) / x_std
		print(f"x_mean: {x_mean}, x_std: {x_std}")
		return x, x_mean, x_std

class GlitchDataset(torch.utils.data.TensorDataset):
	def __init__(self, datadir, ifos, ml_models, glitches, distributions, tr_size, te_size, device, noise=True, aug_phase=True, aug_time=True, split='train', outtype='complex', train_data=None, cachedataset_prefix="antiglitch_cvnn_dataset_clean"):
		self.datadir = datadir
		self.ifos = ifos
		self.ml_models = ml_models
		self.glitches = glitches
		self.distributions = distributions
		self.tr_size = tr_size
		self.te_size = te_size
		self.noise = noise
		self.aug_phase = aug_phase
		self.aug_time = aug_time
		self.device = device
		self.cachedataset_prefix = cachedataset_prefix
		self.size = tr_size if split == 'train' else te_size
		self.x_arr, self.y_arr = None, None
		self.tr_x_mean_real, self.tr_x_std_real, self.tr_x_mean_imag, self.tr_x_std_imag = None, None, None, None
		self.tr_x_mean, self.tr_x_std = None, None
		self.tr_y1_mean, self.tr_y1_std = None, None
		self.tr_y2_mean, self.tr_y2_std = None, None
		self.tr_phase_shifts = None
		self.te_phase_shifts = None
		tr_x_arr_irfft = None
		# create datasets if they don't exist
		for size in [self.tr_size, self.te_size]:
			x_arr, y_arr = [], []
			if cachedataset_prefix + str(size) + '.npz' not in os.listdir(datadir):
				for i in range(size):
					x, y = get_snip(ifos, ml_models)
					x_arr.append(x)
					y_arr.append(y)
				x_arr = np.array(x_arr)
				y_arr = np.array(y_arr)
				# normalize x_arr
				np.savez(datadir + cachedataset_prefix + str(size) + '.npz', x_arr=x_arr, y_arr=y_arr)

		if split == 'train':
			tr_data = np.load(datadir + cachedataset_prefix + str(self.tr_size) + '.npz')
			tr_x_arr = tr_data['x_arr']
			if noise:
				# create white noise in the frequency domain
				tr_x_arr = noise_function(tr_x_arr)
			if aug_phase:
				# Apply the phase shifts to the frequency domain signals
				tr_x_arr, self.tr_phase_shifts = phase_shift(tr_x_arr)
			if aug_time:
				# Apply the time shifts to the frequency domain signals
				tr_x_arr, self.tr_time_shifts = time_shift(tr_x_arr)
			if outtype == 'complex':
				print("Normalizing x train data (complex)")
				self.x_arr, self.tr_x_mean_real, self.tr_x_std_real, self.tr_x_mean_imag, self.tr_x_std_imag = normalize(tr_x_arr)
				self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.complex64)
			elif outtype == 'real':
				# irfft to get to time domain
				len_tr_x_arr = len(tr_x_arr)
				len_tr_x_samp = len(tr_x_arr[0])
				tr_x_arr_irfft = np.array([np.roll(irfft(x), len_tr_x_samp//2) for x in tr_x_arr])
				print("Normalizing x train data (real)")
				self.x_arr, self.tr_x_mean, self.tr_x_std = normalize(tr_x_arr_irfft)
				self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.float32)
    
			# normalize training y_arr
			tr_y_arr = tr_data['y_arr']
			print("Normalizing y train data (y1)")
			tr_y_arr[:,0], self.tr_y1_mean, self.tr_y1_std = normalize(tr_y_arr[:,0])
			print("Normalizing y train data (y2)")
			tr_y_arr[:,1], self.tr_y2_mean, self.tr_y2_std = normalize(tr_y_arr[:,1])
			self.y_arr = tr_y_arr
			self.y_arr = torch.tensor(self.y_arr, device=device, dtype=torch.float32)
   
		if split == 'test':
			# check train_data has been passed
			if train_data is None:
				raise ValueError("train_data must be passed if split is 'test'")
			te_data = np.load(datadir + cachedataset_prefix + str(self.te_size) + '.npz')
			te_x_arr = te_data['x_arr']
			if noise:
				# create white noise in the frequency domain
				te_x_arr = noise_function(te_x_arr)
			if aug_phase:
				# Apply the phase shifts to the frequency domain signals
				te_x_arr, self.te_phase_shifts = phase_shift(te_x_arr)
			if aug_time:
				# Apply the time shifts to the frequency domain signals
				te_x_arr, self.te_time_shifts = time_shift(te_x_arr)
			if outtype == 'complex':
				# normalize test x_arr with training scalings
				print("Normalizing x test data (complex)")
				normalize(te_x_arr, dryrun=True)
				norm_real = (te_x_arr.real - train_data.tr_x_mean_real) / train_data.tr_x_std_real
				norm_imag = (te_x_arr.imag - train_data.tr_x_mean_imag) / train_data.tr_x_std_imag
				self.x_arr = norm_real + 1j * norm_imag
				self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.complex64)
			elif outtype == 'real':
				# irfft to get to time domain
				len_te_x_arr = len(te_x_arr)
				len_te_x_samp = len(te_x_arr[0])
				te_x_arr_irfft = np.array([np.roll(irfft(x), len_te_x_samp//2) for x in te_x_arr])
				# normalize test x_arr with training scalings
				print("Normalizing x test data (real)")
				normalize(te_x_arr_irfft, dryrun=True)
				self.x_arr = (te_x_arr_irfft - train_data.tr_x_mean) / train_data.tr_x_std
				self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.float32)
    
			# normalize test y_arr with training scalings
			te_y_arr = te_data['y_arr']
			print("Normalizing y test data (y1)")
			normalize(te_y_arr[:,0], dryrun=True)
			te_y_arr[:,0] = (te_y_arr[:,0] - train_data.tr_y1_mean) / train_data.tr_y1_std
			print("Normalizing y test data (y2)")
			normalize(te_y_arr[:,1], dryrun=True)
			te_y_arr[:,1] = (te_y_arr[:,1] - train_data.tr_y2_mean) / train_data.tr_y2_std
			self.y_arr = te_y_arr
			self.y_arr = torch.tensor(self.y_arr, device=device, dtype=torch.float32)
   
	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		return self.x_arr[idx], self.y_arr[idx]

def gen_sample(distributions,ifo, ml_model, tosample=['f0', 'gbw', 'amp_r', 'amp_i', 'time']):
    res = {key:None for key in tosample}
    for key in tosample:
        draw = np.random.choice(distributions[ifo][ml_model][key])
        draw_sd = distributions[ifo][ml_model][key + '_sd'][distributions[ifo][ml_model][key].index(draw)]
        draw_final = np.random.normal(draw, draw_sd)
        res[key] = draw_final
    return res

def new_init_Snippet(self, invasd):
        self.invasd = invasd

SnippetNormed = type('SnippetNormed', (antiglitch.SnippetNormed,), {'__init__': new_init_Snippet})

def get_snip(distributions, glitches, ifos, ml_models, tosample=['f0', 'gbw', 'amp_r', 'amp_i', 'time']):
		# randomly select a glitch
		ifo = np.random.choice(ifos)
		ml_model = np.random.choice(ml_models)
		glitch_num = np.random.choice(glitches[ifo][ml_model])
		glitch_invasd = glitches[ifo][ml_model][glitch_num]['invasd']
		# create a SnippetNormed object
		snip = SnippetNormed(glitch_invasd)
		# generate a sample
		snip.ifo = ifo
		snip.key = ml_model
		snip.num = glitch_num
		inf = gen_sample(distributions,ifo, ml_model)
		inf['freqs'] = np.linspace(0, 4096, 513)
		snip.set_infer(inf)
		# get the glitch data in the frequency domain
		x = snip.fglitch
		# check for nans
		if np.isnan(x).any():
			return get_snip(ifos, ml_models, tosample)
		y = (snip.inf['f0'], snip.inf['gbw'])
		return x, y
