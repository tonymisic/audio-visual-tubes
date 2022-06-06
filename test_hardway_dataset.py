import os
from PIL import Image
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import json
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset
import cv2
from sklearn.metrics import auc
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy import signal

def gkern(kernlen=21, std=None):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def cosine(kernsize=14, std=None):
    kern1d = signal.cosine(kernsize).reshape(kernsize, 1)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d
def random(kernsize=14, std=None):
    return torch.randn(kernsize, kernsize)
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset',default='flickr',type=str,help='testset,(flickr or vggss)')
    parser.add_argument('--data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--og_data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--gt_path',default='',type=str)
    parser.add_argument('--og_gt_path',default='',type=str)
    parser.add_argument('--summaries_dir',default='',type=str,help='Model path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--epsilon', default=0.65, type=float, help='pos')
    parser.add_argument('--epsilon2', default=0.4, type=float, help='neg')
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg',action='store_true')
    parser.set_defaults(Neg=True)

    return parser.parse_args()

def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # load model
    model= AVENet(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load('pretrained/lvs_soundnet.pth.tar')
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    print('load pretrained model.')

    # dataloader
    testdataset = GetAudioVideoDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print("Loaded dataloader.")

    # gt for vggss
    if args.testset == 'vggss':
        args.gt_all = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    model.eval()
    print('SOTA cIoU 0.7349397590361446')
    print('SOTA auc 0.5778112449799198')
    iou = []
    # center based gaussian
    for std in range(10):
        gaussian = gkern(14, std=(std+1))
        for step, (image, spec, audio, name, im) in enumerate(testdataloader):
            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            for i in range(spec.shape[0]):
                gaussian_now = cv2.resize(gaussian, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                gaussian_now = normalize_img(-gaussian_now)
                gt_map = testset_gt(args, name[i])
                pred = 1 - gaussian_now
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                pred[pred>threshold] = 1
                pred[pred<1] = 0
                evaluator = Evaluator()
                ciou,_,_ = evaluator.cal_CIOU(pred,gt_map,0.5)
                iou.append(ciou)
        results = []
        for i in range(21):
            result = np.sum(np.array(iou) >= 0.05 * i)
            result = result / len(iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = auc(x, results)
        print('std', std+1)
        print('cIoU' , np.sum(np.array(iou) >= 0.5)/len(iou))
        print('auc',auc_)
if __name__ == "__main__":
    main()