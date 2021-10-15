

import os
from PIL import Image
import torch.utils.data as data

from utils.transforms import Transforms

# PatchRoot = '/disk1/bingo_data/digestpath_data/size_1600'#
# PatchRoot = '/home/bingo/digestpath_data/size_1600'
# PatchRoot = '/home/bingo/paip2019_data/size_1600_s_800_cropped'




def Fun_GetFilelist(PatchRoot):
	PatchListPath = os.path.join(PatchRoot, 'patch_list.txt')
	PatchImgDir = os.path.join(PatchRoot, 'training_patch')
	PatchMaskDir = os.path.join(PatchRoot, 'training_mask')
	namelist = [line.strip().split('\t')[1] for line in open(PatchListPath,'r')]
	filelist = []
	for name in namelist:
		ImgPath = os.path.join(PatchImgDir,name)
		MaskPath = os.path.join(PatchMaskDir, 'Mask_' + name[:-4] + '.png')
		filelist.append((ImgPath,MaskPath))
	return filelist[:]

class DatasetForTraining(data.Dataset):

	def __init__(self, crop_size, patch_size, mean_rgb, std_rgb, PatchRoot):
		super(DatasetForTraining, self).__init__()
		self.filelist = Fun_GetFilelist(PatchRoot)
		self.transforms = Transforms('TRAIN', crop_size, patch_size, mean_rgb, std_rgb)
		# self.crop_size = crop_size
		# self.patch_size = patch_size
		# self.mean_rgb = mean_rgb
		# self.std_rgb = std_rgb

	def __getitem__(self, index):
		input_data = Image.open(self.filelist[index][0]).convert('RGB') # PIL img
		label_data = Image.open(self.filelist[index][1]) # PIL mask
        #print(label_data)
		sample = {'image': input_data, 'label': label_data}
		sample = self.transforms( sample )
		return sample
	def __len__(self):
		return len(self.filelist)
