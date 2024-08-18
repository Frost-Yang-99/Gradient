import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.backends.cudnn as cudnn
import argparse
from torchvision import models
from utils import *
from model import *
from MPRNet import MPRNet
from torchvision.transforms import Compose, ToTensor, Normalize
from dataset_utils import *
from PIL import Image, ImageFile
import math
import torch.nn as nn
import sys
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=8, help='number of epochs to update learning rate')
    parser.add_argument('--model_name', type=str, default='MPRNet_Gradient')
    parser.add_argument('--patch_size', type=int, default=128, help='size of training patch size')
    parser.add_argument('--denoise_dir', type=str, default='../noise_dataset/', help='path of noise dataset')
    parser.add_argument('--dehaze_dir', type=str, default='../hazy_dataset/', help='path of hazy dataset')
    parser.add_argument('--derain_dir', type=str, default='../rainy_dataset/', help='size of rain patch size')
    parser.add_argument('--de_type', type=list, default=['dehaze',], help='which type of degradations is training and testing for.')
    return parser.parse_args()

def train(train_loader, cfg):

    restorer = MPRNet()

    restorer = restorer.cuda()
    transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    criterion_sem = kl_loss(0.0, 0.0, 0.0, 1.0, 0.0).cuda()
    criterion_rec = nn.L1Loss()

    optimizer = torch.optim.Adam([paras for paras in restorer.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180,], gamma=cfg.gamma)

    gradient_sem = {}
    gradient_rec = {}

    loss_semantic_epoch = []
    loss_reconstruction_epoch = []
    loss_epoch = []


    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        for idx_iter, (info, degraded_img, gt_img) in enumerate(train_loader):
            degraded_img, gt_img = degraded_img.to(cfg.device), gt_img.to(cfg.device)


            restored_img1 = restorer(degraded_img)
            loss_sem = criterion_sem(restored_img1[0], gt_img)
            optimizer.zero_grad()
            loss_sem.backward()
            for name, params in restorer.named_parameters():
                if params.requires_grad:
                    gradient_sem[name] = params.grad.clone()



            restored_img2 = restorer(degraded_img)
            loss_rec = criterion_rec(restored_img2[0], gt_img)
            optimizer.zero_grad()
            loss_rec.backward()
            for name, params in restorer.named_parameters():
                if params.requires_grad:
                    gradient_rec[name] = params.grad.clone()


            optimizer.zero_grad()
            for name, params in restorer.named_parameters():
                if params.requires_grad:
                    inner_production = torch.sum(gradient_sem[name] * gradient_rec[name])
                    if inner_production >= 0:
                        params.grad = gradient_sem[name] + gradient_rec[name]
                    else:
                        rec_rec = torch.sum(gradient_rec[name] * gradient_rec[name])
                        projection = inner_production / rec_rec
                        temper = projection * gradient_rec[name]
                        params.grad = gradient_sem[name] - temper + gradient_rec[name]

            optimizer.step()

            loss_semantic_epoch.append(loss_sem.data.cpu())
            loss_reconstruction_epoch.append(loss_rec.data.cpu())
            loss_epoch.append(loss_sem.data.cpu() + loss_rec.data.cpu())


        scheduler.step()

        if (idx_epoch + 1) % 2 == 0:
            torch.save({'epoch': idx_epoch + 1, 'state_dict': restorer.state_dict()},
                   '../log/' + cfg.model_name + '_'+ str(idx_epoch + 1) + '.pth.tar')


def main(cfg):
    train_set = TrainDataset(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

