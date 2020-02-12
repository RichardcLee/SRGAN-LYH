import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from os import listdir

from model import Generator

parser = argparse.ArgumentParser(description='Test generate Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_path', default='../data/Test/4x_downsampled', type=str, help='test low resolution image path')
parser.add_argument('--model_path', default='../saved-models/netG_epoch_4x_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--save_path', default='../data/Test/genHR', type=str, help='save the HR generated by netG')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_PATH = opt.image_path
MODEL_PATH = opt.model_path
SAVE_PATH = opt.save_path

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))

with torch.no_grad():
    for img_name in listdir(IMAGE_PATH):
        image = Image.open(IMAGE_PATH + '/' + img_name)
        image = Variable(ToTensor()(image)).unsqueeze(0)

        if TEST_MODE:
            image = image.cuda()

        start = time.clock()
        out = model(image)
        elapsed = (time.clock() - start)
        print('[ %s cost %ss ]' % (img_name, str(elapsed)))
        out_img = ToPILImage()(out[0].data.cpu())
        out_img.save(SAVE_PATH + '/out_srf_' + str(UPSCALE_FACTOR) + '_' + img_name)