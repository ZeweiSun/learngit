
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import numpy as np
from models import *
import torch
import torch.optim
import kornia
from skimage.measure import compare_psnr
from utils.denoising_utils import *
from utils.CVLossFunctions import *


dtype = torch.cuda.FloatTensor
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'
reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01
OPTIMIZER='adam' # 'LBFGS'
imsize =-1
sigma = 25
sigma_ = sigma/255.
num_iter = 5000
input_depth = 32 
figsize = 4
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
PLOT = True
#losses
mse = torch.nn.MSELoss().type(dtype)
ssim = kornia.losses.SSIM(1,reduction='mean')
l1_loss = torch.nn.L1Loss(reduce=True, reduction='mean')
tv = kornia.losses.TotalVariation()
block = [0,1,2,3,4]
weight = [0.2,0.2,0.2,0.2,0.2]
pl = PerceptualLoss(block,weight,'cuda')
    
    
net = get_net(input_depth, 'skip', pad,
					  skip_n33d=128, 
					  skip_n33u=128, 
					  skip_n11=4, 
					  num_scales=5,
					  upsample_mode='bilinear').type(dtype)	
out_avg = None
last_net = None
psrn_noisy_last = 0
i=0

def main():
    global net_input, net_input_saved, noise, img_noisy_torch, img_noisy_np, img_np, NumPix
    parser = argparse.ArgumentParser(description='DIP')
    parser.add_argument('--fname', type=str, default='F16_GT.png', help='fname')
    parser.add_argument('--show_every', type=int, default=100, help='show_every')
    parser.add_argument('--exp_weight', type=float, default=0.99, help='exp_weight')
    parser.add_argument('--LOSS', type=str, default='mse+tv', help='LOSS')
    parser.add_argument('--savedir', type=str, default='./log', help='savedir')
    parser.add_argument('--cudanum', type=str, default='3', help='which GPU run')
    parser.add_argument('--tv_weight', type=float, default=1e-1, help='tv_loss weight')
    parser.add_argument('--pl_weight', type=float, default=1e0, help='pl_loss weight')
    # parser.add_argument('--devices', type=str, default='3', help='CUDA used')
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    
    
    img_pil = crop_image(get_image('data/denoising/'+args['fname'], imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    SIZE = img_np.shape
    NumPix = SIZE[0]*SIZE[1]*SIZE[2] 
    
    DIP(args).inference()


class DIP:
    def __init__(self, args):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.savedir = args['savedir']
        self.writer = SummaryWriter(self.savedir)
        self.show_every = args['show_every']
        self.exp_weight = args['exp_weight']
        self.LOSS = args['LOSS']
        self.logdir = args['fname']
        self.tv_weight = args['tv_weight']
        self.pl_weight = args['pl_weight']
        
                
    def inference(self):
        # Compute number of parameters
        p = get_params(OPT_OVER, net, net_input)
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
        print ('Number of params: %d' % s)
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(p, lr=LR)
        for j in range(num_iter):
            optimizer.zero_grad()
            self.closure()
            optimizer.step()
            
            
    def out_loss(self,LOSS, out, img_noisy_torch):  
        tv_weight = self.tv_weight
        pl_weight = self.pl_weight
        if LOSS == 'mse':  
            LOSS_write = 'mse+'
            total_loss = mse(out, img_noisy_torch)
        elif LOSS == 'ssim':
            LOSS_write = 'ssim'
            total_loss = ssim(out, img_noisy_torch)
        elif LOSS == 'ssim+l1':
            LOSS_write = 'ssim+'+'l1'
            total_loss = ssim(out, img_noisy_torch) + l1_loss(out, img_noisy_torch)
        elif LOSS == 'mse+tv':
            LOSS_write = 'mse+'+str(tv_weight)+'tv'
            total_loss = mse(out, img_noisy_torch) + tv_weight*tv(out)/NumPix
        elif LOSS == 'ssim+tv':
            LOSS_write = 'ssim+'+str(tv_weight)+'tv'
            total_loss = ssim(out, img_noisy_torch) + tv_weight*tv(out)/NumPix
        elif LOSS == 'mse+ssim+tv':
            LOSS_write = 'mse+'+'ssim'+str(tv_weight)+'tv'
            total_loss = mse(out, img_noisy_torch) + ssim(out, img_noisy_torch) + tv_weight*tv(out)/NumPix
        elif LOSS == 'mse+pl':
            LOSS_write = 'mse+'+str(pl_weight)+'pl'
            total_loss = mse(out, img_noisy_torch) + pl_weight*pl(out, img_noisy_torch)
        elif LOSS == 'mse+pl+tv':
            LOSS_write = 'mse+'+str(pl_weight)+'pl'+str(tv_weight)+'tv'
            total_loss = mse(out, img_noisy_torch) + pl_weight*pl(out, img_noisy_torch) + tv_weight*tv(out)/NumPix
        else:
            assert False 
        return total_loss, LOSS_write
        
    def closure(self):

        global i, out_avg, psrn_noisy_last, last_net
        LOSS = self.LOSS
        exp_weight = self.exp_weight
        logdir = self.logdir
        show_every = self.show_every
        writer = self.writer
        

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        
        total_loss, LOSS_write = self.out_loss(LOSS, out, img_noisy_torch)
        total_loss.backward()
        

        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        

        writer.add_scalars(logdir,{LOSS_write+'total_loss'+current_time:total_loss.item(),LOSS_write+'psrn_noisy'+current_time:psrn_noisy,LOSS_write+'psrn_gt'+current_time:psrn_gt,LOSS_write+'psrn_gt_sm'+current_time:psrn_gt_sm}, i)    
        if PLOT and i % show_every == 0:
            writer.add_images(logdir+LOSS_write+current_time,torch.cat((out,out_avg),0),i)    

        # Backtracking
        # if i % show_every:
            # if psrn_noisy - psrn_noisy_last < -5: 
                # print('Falling back to previous checkpoint.')

                # for new_param, net_param in zip(last_net, net.parameters()):
                        # net_param.data.copy_(new_param.cuda())

                # return total_loss*0
            # else:
                # last_net = [x.detach().cpu() for x in net.parameters()]
                # psrn_noisy_last = psrn_noisy
            
        i += 1
        writer.close()

        return total_loss

if __name__ == '__main__':
    main()

# out_np = torch_to_np(net(net_input))
# q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);

