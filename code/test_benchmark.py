import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_path', default='../saved-models/netG_epoch_4x_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--dataset_name', default='urban100', type=str, help='the name of dataset to test benchmark')
parser.add_argument('--lr_path', default='../data/classical_SR_datasets/urban100/4x_downsampled', type=str, help='LR image path')
parser.add_argument('--hr_path', default='../data/classical_SR_datasets/urban100/Original', type=str, help='HR image path')
parser.add_argument('--num_works', default=8, type=int, help='number of thread to work')
parser.add_argument('--out_path', default='../benchmark_results/%s_%sx', type=str, help='path to save benchmark results')
parser.add_argument('--statistics_path', default='../statistics', type=str, help='path to save statistics')
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

UPSCALE_FACTOR = opt.upscale_factor
MODEL_PATH = opt.model_path
LR_PATH = opt.lr_path
HR_PATH = opt.hr_path
NUM_WORK = opt.num_works
STATISTICS_PATH = opt.statistics_path
DATASET_NAME = opt.dataset_name
OUT_PATH = opt.out_path % (DATASET_NAME, UPSCALE_FACTOR)

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

results = {DATASET_NAME: {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

test_set = TestDatasetFromFolder(LR_PATH, HR_PATH, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=NUM_WORK, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark dataset -- %s]' % DATASET_NAME)

with torch.no_grad():
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        lr_image = Variable(lr_image).to(device)
        hr_image = Variable(hr_image).to(device)

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item() # .data[0]

        test_images = torch.stack([
            display_transform()(hr_restore_img.squeeze(0)),
            display_transform()(hr_image.data.cpu().squeeze(0)),
            display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, OUT_PATH + '/' + DATASET_NAME + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results[DATASET_NAME]['psnr'].append(psnr)
        results[DATASET_NAME]['ssim'].append(ssim)

saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(STATISTICS_PATH + '/' + DATASET_NAME + '_' + str(UPSCALE_FACTOR) + 'x_test_results.csv', index_label='DataSet')
