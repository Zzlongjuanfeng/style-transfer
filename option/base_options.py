import argparse
import os
from util import util
import torch
import time


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--dataroot', type=str, default='/media/Data/dataset_xian/F16/trainA', help='path to images')
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256,help='scale images to this size')
        self.parser.add_argument('--style_image', type=str, default='./style_images/plane.jpg',help='style image')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--exp_name', type=str, default='default_dir', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--seed', type=int, default=1080, help='seed for random!')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        self.parser.add_argument('--display_freq', type=int, default=10,help='frequency of showing training results on screen')
        self.parser.add_argument('--epochs', type=int, default=30, help='training epochs')
        self.parser.add_argument('--content_layer', type=int, default=1, help='content layer for content loss')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--content_weight', type=float, default=1,help='weight for content')
        self.parser.add_argument('--style_weight', type=float, default=1e5,help='weight for style')
        self.parser.add_argument('--regularization_weight', type=float, default=1e-6,help='weight for style')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # exp_name
        date = time.strftime('%Y%m%d', time.localtime(time.time()))
        if (self.opt.exp_name == 'default_dir'):
            self.opt.exp_name = '{}_{}_{}'.format(date, self.opt.style_weight, self.opt.regularization_weight)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt
