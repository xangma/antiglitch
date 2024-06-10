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

class GlitchDataset(torch.utils.data.TensorDataset):
	def __init__(self, datadir, ifos, ml_models, glitches, distributions, tr_size, te_size, device, noise=True, aug=True, split='train', outtype='complex', cachedataset_prefix="antiglitch_cvnn_dataset_clean"):
		self.ifos = ifos
		self.ml_models = ml_models
		self.glitches = glitches
		self.distributions = distributions
		self.tr_size = tr_size
		self.te_size = te_size
		self.size = tr_size if split == 'train' else te_size
		self.x_arr = []
		self.y_arr = []
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
		tr_data = np.load(datadir + cachedataset_prefix + str(self.tr_size) + '.npz')
		if split == 'test':
			te_data = np.load(datadir + cachedataset_prefix + str(self.te_size) + '.npz')
   
   
		tr_x_arr = tr_data['x_arr']
		if split == 'test':
			te_x_arr = te_data['x_arr']
  
		# add noise to x_arr
		if noise:
			# create white noise in the frequency domain
			noise_generator_function = lambda x: np.random.normal(0, 1, len(x)) + 1j * np.random.normal(0, 1, len(x))
			noise = np.array([noise_generator_function(x) for x in tr_x_arr])
			tr_x_arr += noise
			del noise
			if split == 'test':
				noise = np.array([noise_generator_function(x) for x in te_x_arr])
				te_x_arr += noise
				del noise
		
		if aug:
			# apply random phase shifts to x_arr
			self.tr_phase_shifts = np.random.uniform(-np.pi/4, np.pi/4, len(tr_x_arr))

			# Apply the phase shifts to the frequency domain signals
			for i in range(len(tr_x_arr)):
				tr_x_arr[i] = tr_x_arr[i] * np.exp(1j * self.tr_phase_shifts[i])
			if split == 'test':
				self.te_phase_shifts = np.random.uniform(-np.pi/4, np.pi/4, len(te_x_arr))
				for i in range(len(te_x_arr)):
					te_x_arr[i] = te_x_arr[i] * np.exp(1j * self.te_phase_shifts[i])

		# normalize training x_arr
  
		if outtype == 'complex':
			# normalize training x_arr
			tr_real_parts = np.real(tr_x_arr)
			tr_imag_parts = np.imag(tr_x_arr)
			
			tr_x_mean_real = np.mean(tr_x_arr.real)
			tr_x_std_real = np.std(tr_x_arr.real)
			tr_x_mean_imag = np.mean(tr_x_arr.imag)
			tr_x_std_imag = np.std(tr_x_arr.imag)
			self.x_arr_mean_real = tr_x_mean_real
			self.x_arr_std_real = tr_x_std_real
			self.x_arr_mean_imag = tr_x_mean_imag
			self.x_arr_std_imag = tr_x_std_imag
			tr_normalized_real = (tr_real_parts - tr_x_mean_real) / tr_x_std_real
			tr_normalized_imag = (tr_imag_parts - tr_x_mean_imag) / tr_x_std_imag
			
			self.x_arr = tr_normalized_real + 1j * tr_normalized_imag

			print(f"tr_x_mean_real: {tr_x_mean_real}, tr_x_std_real: {tr_x_std_real}, tr_x_mean_imag: {tr_x_mean_imag}, tr_x_std_imag: {tr_x_std_imag}")
			if split == 'test':
				# normalize test x_arr with training scalings
				te_real_parts = np.real(te_x_arr)
				te_imag_parts = np.imag(te_x_arr)
				te_x_mean_real = np.mean(te_x_arr.real)
				te_x_std_real = np.std(te_x_arr.real)	
				te_x_mean_imag = np.mean(te_x_arr.imag)
				te_x_std_imag = np.std(te_x_arr.imag)
				print(f"te_x_mean_real: {te_x_mean_real}, te_x_std_real: {te_x_std_real}, te_x_mean_imag: {te_x_mean_imag}, te_x_std_imag: {te_x_std_imag}")
				te_normalized_real = (te_real_parts - tr_x_mean_real) / tr_x_std_real
				te_normalized_imag = (te_imag_parts - tr_x_mean_imag) / tr_x_std_imag
				self.x_arr = te_normalized_real + 1j * te_normalized_imag
				del te_real_parts, te_imag_parts, te_normalized_real, te_normalized_imag
			del tr_real_parts, tr_imag_parts, tr_normalized_real, tr_normalized_imag, tr_x_mean_real, tr_x_std_real, tr_x_mean_imag, tr_x_std_imag
			self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.complex64)

		elif outtype == 'real':
			# irfft to get to time domain
			len_tr_x_arr = len(tr_x_arr)
			len_tr_x_samp = len(tr_x_arr[0])
			if 'tr_x_arr_irfft' not in locals():
				tr_x_arr_irfft = np.array([np.roll(irfft(x), len_tr_x_samp//2) for x in tr_x_arr])
			# normalize training x_arr
			tr_x_mean_irfft = tr_x_arr_irfft.mean()
			self.x_arr_mean = tr_x_mean_irfft
			tr_x_std_irfft = tr_x_arr_irfft.std()
			self.x_arr_std = tr_x_std_irfft
			self.x_arr = (tr_x_arr_irfft - tr_x_mean_irfft) / tr_x_std_irfft
			self.x_arr = torch.tensor(self.x_arr, device=device, dtype=torch.float32)

			if split == 'test':
				# normalize test x_arr with training scalings
				len_te_x_arr = len(te_x_arr)
				len_te_x_samp = len(te_x_arr[0])
				if 'te_x_arr_irfft' not in locals():
					te_x_arr_irfft = np.array([np.roll(irfft(x), len_te_x_samp//2) for x in te_x_arr])
				te_x_arr = (te_x_arr_irfft - tr_x_mean_irfft) / tr_x_std_irfft
				self.x_arr = torch.tensor(te_x_arr, device=device, dtype=torch.float32)
				del te_x_arr

		# normalize training y_arr
		tr_y_arr = tr_data['y_arr']
		tr_y1_mean = tr_y_arr[:,0].mean()
		tr_y1_std = tr_y_arr[:,0].std()
		tr_y2_mean = tr_y_arr[:,1].mean()
		tr_y2_std = tr_y_arr[:,1].std()

		tr_y_arr[:,0] = (tr_y_arr[:,0] - tr_y1_mean)/tr_y1_std
		tr_y_arr[:,1] = (tr_y_arr[:,1] - tr_y2_mean)/tr_y2_std
		self.y_arr = tr_y_arr
		if split == 'test':
			# normalize test y_arr with training scalings
			te_y_arr = te_data['y_arr']
			te_y_arr[:,0] = (te_y_arr[:,0] - tr_y1_mean)/tr_y1_std
			te_y_arr[:,1] = (te_y_arr[:,1] - tr_y2_mean)/tr_y2_std
			self.y_arr = te_y_arr
			del te_data, te_y_arr
		# convert to tensors
		del tr_data, tr_y_arr
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
