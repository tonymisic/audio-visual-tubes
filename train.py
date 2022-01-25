import os
import torch
from torch.optim import *
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import json
import argparse, torch.optim as optim 
from model import FullModel
from datasets import WholeVideoDataset
import cv2
from sklearn.metrics import auc
from losses import HardWayLoss
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset',default='flickr',type=str,help='testset,(flickr or vggss)') 
    parser.add_argument('--data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--gt_path',default='',type=str)
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
    os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

    # load model
    model = FullModel(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).cuda()
    model.to(device)
    criterion = HardWayLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # from paper

    # dataloader
    trainset = WholeVideoDataset(args, mode='train')
    traindataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers = 1)
    print("Loaded dataloader.")

    # gt for vggss
    if args.testset == 'vggss':
        args.gt_all = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    iou = []
    running_loss = 0.0
    for step, (frames, spec, audio, samplerate, name) in enumerate(traindataloader):
        print('%d / %d' % (step, len(traindataloader) - 1))
        spec = Variable(spec).cuda()
        attention_map = model(spec.float(), frames)
        for sample in range(attention_map.size(0)):
            for frame in range(attention_map.size(1)):
                heatmap_now = cv2.resize(attention_map[sample,frame].cpu().detach().numpy(), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                #heatmap_now = normalize_img(-heatmap_now)
                #image_now = normalize_img(frames[sample, frame])
                #im = Image.fromarray(image_now[0][0].cpu().numpy() * 255).convert('RGB')
                colored_map = cv2.applyColorMap(np.uint8(heatmap_now * 255), cv2.COLORMAP_JET)
                im2 = Image.fromarray(colored_map).convert('RGB')
                #im.save("tmp/original.jpg")
                im2.save("tmp/heatmap.jpg")
        break
if __name__ == "__main__":
    main()