import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.cuda import empty_cache
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch to start training')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_works', default=8, type=int, help='number of thread use to load data')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or cpu')
parser.add_argument('--lr', type=float, default=1e-3, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--eps', type=float, default=1e-8, help='adam: ϵ')
parser.add_argument('--weight_decay', type=float, default=0, help='adam: weight decay')
parser.add_argument('--train_set_path', default='../data/Train/Crop_Original', type=str, help='the path of train set')
parser.add_argument('--validation_set_path', default='../data/Validation/Original', type=str, help='the path of validation set')
parser.add_argument('--model_save_path', default='../saved-models', type=str, help='the path of model to save')
parser.add_argument('--model_load_path', default='../saved-models', type=str, help='the path of model to load')
parser.add_argument('--statistics_save_path', default='../statistics', type=str, help='the path of statistics to save')
parser.add_argument('--training_results_save_path', default='../training_results', type=str, help='the path of training_results to save')
parser.add_argument('--checkpoint_interval', default=20, type=int, help='interval between model checkpoints')
parser.add_argument('--statistics_interval', default=20, type=int, help='interval between statistics saving')


opt = parser.parse_args()
print(opt)

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
START_EPOCH = opt.start_epoch
NUM_EPOCHS = opt.num_epochs
NUM_WORKS = opt.num_works
USE_GPU = opt.use_gpu
BATCH_SIZE = opt.batch_size
LEARNING_RATE = opt.lr
BETAS = (opt.b1, opt.b2)
EPS = opt.eps
WEIGHT_DECAY = opt.weight_decay
TRAIN_SET_PATH = opt.train_set_path
VALIDATION_SET_PATH = opt.validation_set_path
MODEL_SAVE_PATH = opt.model_save_path
MODEL_LOAD_PATH = opt.model_load_path
STATISTICS_SAVE_PATH = opt.statistics_save_path
TRAINING_RESULTS_SAVE_PATH = opt.training_results_save_path
CHECK_POINT_INTERVAL = opt.checkpoint_interval
STATISTICS_INTERVAL = opt.statistics_interval
OUT_PATH = TRAINING_RESULTS_SAVE_PATH + '/SRF_' + str(UPSCALE_FACTOR)

device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if not os.path.exists(STATISTICS_SAVE_PATH):
    os.makedirs(STATISTICS_SAVE_PATH)

start_time = time()

# load data set
train_set = TrainDatasetFromFolder(TRAIN_SET_PATH, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder(VALIDATION_SET_PATH, upscale_factor=UPSCALE_FACTOR)
# big dataset set pin_memory=True 这里哟
train_loader = DataLoader(dataset=train_set, num_workers=NUM_WORKS, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_set, num_workers=NUM_WORKS, batch_size=1, shuffle=False, pin_memory=False)

netG = Generator(UPSCALE_FACTOR).to(device)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator().to(device)
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss().to(device)

if START_EPOCH != 0:
    # Load trained models
    netG.load_state_dict(torch.load(MODEL_LOAD_PATH + '/netG_epoch_%dx_%d.pth' % (UPSCALE_FACTOR, START_EPOCH+1)))
    netD.load_state_dict(torch.load(MODEL_LOAD_PATH + '/netD_epoch_%dx_%d.pth' % (UPSCALE_FACTOR, START_EPOCH+1)))

# Adam, weight_dacy=0.0001
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(START_EPOCH + 1, NUM_EPOCHS + 1):
    epoch_start_time = time()

    train_bar = tqdm(train_loader)  # 循环进度条
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:  # 每次一个batch_size数量的图片
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target).to(device, non_blocking=True)  # 异步加载此数据到gpu或cpu
        z = Variable(data).to(device)
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.item() * batch_size # data[0]
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    # 验证
    netG.eval()
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    with torch.no_grad():
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = Variable(val_lr).to(device)
            hr = Variable(val_hr).to(device, non_blocking=True)  # 异步加载数据到GPU或CPU
            sr = netG(lr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0))])
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, OUT_PATH + '/epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

    # save model parameters
    if epoch % CHECK_POINT_INTERVAL == 0:
        torch.save(netG.state_dict(), MODEL_SAVE_PATH + '/netG_epoch_%dx_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), MODEL_SAVE_PATH + '/netD_epoch_%dx_%d.pth' % (UPSCALE_FACTOR, epoch))

        empty_cache()   # 顺便清空一次，删除无用变量

    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    # save statistics
    if epoch % STATISTICS_INTERVAL == 0:
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(STATISTICS_SAVE_PATH + '/srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

    epoch_end_time = time()
    print('[ epoch %d cost %fs ]' % (epoch, epoch_end_time - epoch_start_time))

end_time = time()
print('total cost:', end_time - start_time, 's')