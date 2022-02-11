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
import wandb
train, test, val, record, save = False, True, True, False, False 

if record:
    wandb.init(entity="tonymisic", project="Audio-Visual Tubes",
        config={
            "Model": "Hard Way",
            "dataset": "flickr10k",
            "testset": 9,
            "epochs": 200,
            "batch_size": 128
        }
    )

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
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
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
    parser.add_argument('--epochs',default=1,type=int,help='Number of total epochs to run')
    # novel arguments
    parser.add_argument('--sampling_rate', default=20, type=int,help='Sampling rate for frame selection')
    return parser.parse_args() 

def main():
    # get all arguments
    args = get_arguments()
    #  gpu and model init
    device = torch.device("cuda")
    model = AVENet(args)
    model = model.cuda() 
    model = nn.DataParallel(model)
    model.to(device)
    # load pretrained model if it exists, off for now
    if os.path.exists(args.summaries_dir):
        print('load pretrained')
        checkpoint = torch.load(args.summaries_dir)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # init datasets
    dataset = SubSampledFlickr(args,  mode='train')
    testdataset = PerFrameLabels(args, mode='test')
    original_testset = GetAudioVideoDataset(args, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    originaldataloader = DataLoader(original_testset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    # loss
    criterion = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")
    # optimiser
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Optimizer loaded.")
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[50,100,150,180], gamma=0.1)
    if record:
        wandb.watch(model, optim, log="all", log_freq=1000)
    for epoch in range(args.epochs):
        # Train
        if train:
            running_loss = 0.0
            for step, (frames, spec, audio, samplerate, name) in enumerate(dataloader):
                #print("Training Step: " + str(step) + "/" + str(len(dataloader)))
                model.train()
                sample_loss = 0.0
                for count, i in enumerate(range(frames.size(2))):
                    spec = Variable(spec).cuda()
                    heatmap, out, _, _ = model(frames[:,:,i,:,:].float(), spec.float())
                    target = torch.zeros(out.shape[0]).cuda().long()     
                    loss = criterion(out, target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    sample_loss += float(loss)
                running_loss += sample_loss / (count + 1)
                if record:
                    wandb.log({"step": step})
            final_loss = running_loss / float(step + 1)
            if record:
                wandb.log({"loss": final_loss})
            print("Epoch " + str(epoch) + " training done.")
            scheduler.step()
        
        if test:
            with torch.no_grad():
                model.eval()
                # My Testset
                ious,aucs = [], []
                for step, (frames, spec, audio, samplerate, name) in enumerate(testdataloader):
                    #print("Testing Step: " + str(step) + "/" + str(len(testdataloader)))
                    iou = []
                    for i in range(args.sampling_rate, frames.size(2), args.sampling_rate):
                        spec = Variable(spec).cuda()
                        heatmap, out, _, _ = model(frames[:,:,i,:,:].float(), spec.float())
                        target = torch.zeros(out.shape[0]).cuda().long() 
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
                        iou.append(ciou)
                    results = []
                    for i in range(21):
                        result = np.sum(np.array(iou) >= 0.05 * i)
                        result = result / len(iou)
                        results.append(result)
                    x = [0.05 * i for i in range(21)]
                    auc_ = auc(x, results)
                    ious.append(np.sum(np.array(iou) >= 0.5) / len(iou))
                    aucs.append(auc_)
                print("Whole Video cIoU ", np.sum(ious) / len(ious))
                print("Whole Video auc ", np.sum(aucs) / len(aucs))
                if record:
                    wandb.log({ "Whole Video cIoU": np.sum(ious) / len(ious),
                                "Whole Video AUC": np.sum(aucs) / len(aucs)})
        if val:
            with torch.no_grad():
                model.eval()
                iou = []
                for step, (image, spec, audio, name, im) in enumerate(originaldataloader):
                    #print('%d / %d' % (step,len(originaldataloader) - 1))
                    spec = Variable(spec).cuda()
                    image = Variable(image).cuda()
                    heatmap,_,Pos,Neg = model(image.float(),spec.float())
                    heatmap_arr =  heatmap.data.cpu().numpy()
                    for i in range(spec.shape[0]):
                        heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                        heatmap_now = normalize_img(-heatmap_now)
                        gt_map = testset_gt(args, name[i])
                        pred = 1 - heatmap_now
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
                print("Hardway Test cIoU ", np.sum(np.array(iou) >= 0.5)/len(iou))
                print("Hardway Test auc ", auc_)
                if record:
                    wandb.log({ "Hardway Test cIoU": np.sum(np.array(iou) >= 0.5)/len(iou),
                                "Hardway Test AUC": auc_})
        if save:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, args.summaries_dir + 'model_ep%s.pth.tar' % (str(epoch)) 
            )
        
if __name__ == "__main__":
    main()