# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
import cv2
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/home/janischl/HRNet/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
                        required=False,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='/home/janischl/HRNet/output/imagenet/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100/model_best.pth.tar')
    
    args = parser.parse_args()
    
    update_config(config, args)

    return args

def main(img):
    args = parse_args()


    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    #logger.info(pprint.pformat(args))
    #logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    #logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        #logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))#['state_dict']
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'model_best.pth.tar')
        #logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Data loading code
    #valdir = os.path.join(config.DATASET.ROOT,
    #                      config.DATASET.TEST_SET)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #valid_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        #transforms.Resize((244,244)),
    #        transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
    #        transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
    #        transforms.ToTensor(),
    #       normalize,
    #    ])),
    #    batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #    shuffle=False,
    #    num_workers=config.WORKERS,
    #    pin_memory=True
    #)
    test_transforms = transforms.Compose([
            #transforms.Resize((244,244)),
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            #normalize,
            ])

    #image = cv2.imread(img_path)
    #to_pil = transforms.ToPILImage()
    #images, labels = get_random_images(5)
    #fig=plt.figure(figsize=(10,10))
    #image = to_pil(image)
    model.eval()
    image_tensor = test_transforms(img)#.float()
    image_tensor = image_tensor.unsqueeze(0)
    #print(image_tensor)
    output = model(image_tensor)
    #print(output)
    #print(torch.max(output, dim=1))
    #print(output)
    output = torch.argmax(output, dim=1)
    #print(output)
    output = output.item()
    #print(output)
    #print (index)
    # evaluate on validation set
    #validate(config, valid_loader, model, criterion, final_output_dir,
             #tb_log_dir, None)
    return output


if __name__ == '__main__':
    #image = Image.open("/home/janischl/HRNet/imagenet/images/valid/Tool/1_1_1_140.png")
    #image = Image.open("/home/janischl/HRNet/imagenet/images/train/Background/1_1_12.png")
    #image = Image.open("/home/janischl/HRNet/imagenet/images/train/Flankwear/1_1_1131.png")
    image = Image.open("/home/janischl/HRNet/imagenet/images/train/Tool/1_1_1136.png")
    #to_pil = transforms.ToPILImage()
    #imagee = to_pil(image)
    
    main(image)