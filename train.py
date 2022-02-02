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
import convert_jpg_to_mp4, subprocess
os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

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

    # load model
    model = FullModel(args)
    #model.vidnet.load_state_dict(torch.load('pretrained/r3d18_KM_200ep.pth')['state_dict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).cuda()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # from paper

    # dataloader
    trainset = WholeVideoDataset(args, mode='test')
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
    ious, aucs = [], []
    running_loss = 0.0
    qualitative, quantitative = True, False
    for step, (frames, spec, audio, samplerate, name) in enumerate(traindataloader):
        subprocess.call("mkdir imgs/" + name[0].strip('.mp4'), cwd=os.getcwd(), shell=True)
        print('%d / %d' % (step, len(traindataloader) - 1))
        spec = Variable(spec).cuda()
        attention_map = model(spec.float(), frames).unsqueeze(1)
        upsample = torch.nn.Upsample(size=(frames.size(2), 7, 7), mode='trilinear')
        attention_map = upsample(attention_map).squeeze(1)
        for sample in range(attention_map.size(0)):
            for frame in range(attention_map.size(1)):
                heatmap_now = cv2.resize(attention_map[sample,frame].cpu().detach().numpy(), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(heatmap_now)
                image_now = normalize_img(frames[sample, :, frame])
                pred = 1 - heatmap_now
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                pred[pred>threshold]  = 1
                pred[pred<1] = 0
                if qualitative:
                    colored_map = cv2.applyColorMap(np.uint8(pred * 255), cv2.COLORMAP_JET)
                    im2 = Image.fromarray(np.uint8(np.add((image_now.cpu().numpy() * 255).transpose((1,2,0)) * 0.5, colored_map * 0.5))).convert('RGB')
                    im2.save("imgs/" + name[0].strip('.mp4') + "/pred_heatmap" + str(frame) + ".jpg")
                if quantitative:
                    if (frame % 20 == 0 and frame != 0): # labeled frames to calculate cIoU
                        gt_map = testset_gt_frame(args, name[0], frame)
                        evaluator = Evaluator()
                        ciou,_,_ = evaluator.cal_CIOU(pred,gt_map,0.5)
                        iou.append(ciou)
        convert_jpg_to_mp4.main()
        subprocess.call(str("rm -rf imgs/" + name[0].strip('.mp4') + "/*"), cwd=os.getcwd(), shell=True)
        break
        results = []
        for i in range(21):
            result = np.sum(np.array(iou) >= 0.05 * i)
            result = result / len(iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = auc(x, results)
        print('cIoU' , np.sum(np.array(iou) >= 0.5) / len(iou))
        print('auc',auc_)
        ious.append(np.sum(np.array(iou) >= 0.5) / len(iou))
        aucs.append(auc_)
    print("Average cIoU ", np.sum(ious) / len(ious))
    print("Average auc ", np.sum(aucs) / len(aucs))
if __name__ == "__main__":
    main()