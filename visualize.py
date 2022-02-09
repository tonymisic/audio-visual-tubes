import os
import torch
from torch.optim import *
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import argparse, torch.optim as optim 
from model import AVENet
from datasets import SubSampledFlickr, GetAudioVideoDataset, PerFrameLabels
import cv2
from sklearn.metrics import auc
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
import warnings
warnings.filterwarnings('ignore')
def get_arguments():
    parser = argparse.ArgumentParser()
    # from testing code
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
    # from training code
    parser.add_argument('--learning_rate',default=1e-4,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads',default=4,type=int,help='Number of threads for multi-thread loading')
    parser.add_argument('--epochs',default=200,type=int,help='Number of total epochs to run')
    # novel arguments
    parser.add_argument('--sampling_rate', default=20, type=int,help='Sampling rate for frame selection')
    return parser.parse_args() 

def save_image(image, index, pred=None, gt_map=None):
    image = normalize_img(image)
    temp = cv2.applyColorMap(np.uint8(gt_map * 128), cv2.COLORMAP_JET)
    temp2 = cv2.applyColorMap(np.uint8(pred * 255), cv2.COLORMAP_JET)
    Image.fromarray(np.uint8(np.add((image[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.4, np.add(temp * 0.5, temp2 * 0.5) * 0.6))).convert('RGB').save("tmp/pred_heatmap" + str(index) + ".jpg")

def pause():
    input("Enter to continue:")

def main():
    # get all arguments
    args = get_arguments()
    #  gpu and model init
    device = torch.device("cuda")
    model = AVENet(args)
    model = model.cuda() 
    model = nn.DataParallel(model)
    model.to(device)
    # load pretrained model if it exists
    if os.path.exists(args.summaries_dir) and False:
        print('load pretrained')
        checkpoint = torch.load(args.summaries_dir)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # init datasets
    #dataset = SubSampledFlickr(args,  mode='train')
    testdataset = PerFrameLabels(args, mode='test')
    #original_testset = GetAudioVideoDataset(args, mode='test')
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    #originaldataloader = DataLoader(original_testset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    # loss
    criterion = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")
    # optimiser
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Optimizer loaded.")
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[50,100,150,180], gamma=0.1)

    # epoch
    for epoch in range(args.epochs):
        for step, (frames, spec, _, _, name) in enumerate(testdataloader):
            print("Training Step: " + str(step) + "/" + str(len(testdataloader)))
            model.train()
            for count, i in enumerate(range(args.sampling_rate * 5, frames.size(2), args.sampling_rate)):
                spec = Variable(spec).cuda()
                heatmap, out, _, _ = model(frames[:,:,i,:,:].float(), spec.float())
                target = torch.zeros(out.shape[0]).cuda().long() 
                loss = criterion(out, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                heatmap_arr =  heatmap.data.cpu().numpy()
                heatmap_now = cv2.resize(heatmap_arr[0, 0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(-heatmap_now)
                pred = 1 - heatmap_now
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                pred[pred>threshold] = 1
                pred[pred<1] = 0
                gt_map = testset_gt_frame(args, name[0], i)
                evaluator = Evaluator() 
                ciou,_,_ = evaluator.cal_CIOU(pred, gt_map, 0.5)
                print("Video: " + name[0] + " Frame: " + str(i) + " cIoU: " + str(ciou) + " loss: " + str(float(loss)))
                save_image(frames[:,:,i,:,:].float(), epoch, pred, gt_map)
                #pause()
                break
            break
        scheduler.step()
if __name__ == "__main__":
    main()