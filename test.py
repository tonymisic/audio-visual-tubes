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

# get layer activations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

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

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # load model
    model= AVENet(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.cuda()
    model.module.imgnet.layer4.register_forward_hook(get_activation('layer4'))
    checkpoint = torch.load(args.summaries_dir)
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
    iou = []
    for step, (image, spec, audio,name,im) in enumerate(testdataloader):
        print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
        heatmap_arr =  heatmap.data.cpu().numpy()
        gaussian = gkern(14, std=5)
        all_ones = np.ones([14,14])
        write_heatmaps, write_preds = False, False
        for i in range(spec.shape[0]):
            heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now = normalize_img(-heatmap_now)
            image_now = normalize_img(image)
            all_ones_now = cv2.resize(all_ones, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            all_ones_now = normalize_img(all_ones_now)
            # Activation mapping layer4 resnet.
            current_activation = torch.mean(activation['layer4'], 1).cpu().numpy()
            activation_now = cv2.resize(current_activation[i], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            activation_now = normalize_img(activation_now)
            gaussian_now = cv2.resize(gaussian, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            gaussian_now = normalize_img(-gaussian_now)
            if write_heatmaps:
                colored_act_map = cv2.applyColorMap(np.uint8(activation_now * 255), cv2.COLORMAP_JET)
                visualization = Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, colored_act_map * 0.5))).convert('RGB')
                visualization.save('tmp/activation.jpg')
                gaussian_now_map = cv2.applyColorMap(np.uint8(gaussian_now * 255), cv2.COLORMAP_JET)
                visualization = Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, gaussian_now_map * 0.5))).convert('RGB')
                visualization.save('tmp/gaussian.jpg')
                # original heatmap activations
                im = Image.fromarray(image_now[0][0].cpu().numpy() * 255).convert('RGB')
                colored_map = cv2.applyColorMap(np.uint8(heatmap_now * 255), cv2.COLORMAP_JET)
                im2 = Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, colored_map * 0.5))).convert('RGB')
                im.save("tmp/original.jpg")
                im2.save("tmp/heatmap.jpg")
            # end of visualization
            gt_map = testset_gt(args, name[i])
            #gt_map = cv2.resize(gt_map[int(gt_map.shape[0] / 4):int(gt_map.shape[0] / 4) + 150, 0:150], 
            #    dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            pred = 1 - activation_now # CHANGE WHEN COMPARING QUANTITATIVE
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            pred[pred>threshold] = 1
            pred[pred<1] = 0
            evaluator = Evaluator()
            ciou,_,_ = evaluator.cal_CIOU(pred,gt_map,0.5)
            pred_same = activation_now # CHANGE WHEN COMPARING QUANTITATIVE
            threshold = np.sort(pred_same.flatten())[int(pred_same.shape[0] * pred_same.shape[1] / 2)]
            pred_same[pred_same>threshold] = 1
            pred_same[pred_same<1] = 0
            evaluator = Evaluator()
            ciou_same,_,_ = evaluator.cal_CIOU(pred_same,gt_map,0.5)
            if ciou_same > ciou:
                iou.append(ciou_same)
            else:
                iou.append(ciou)
            pred2 = 1 - activation_now
            threshold = np.sort(pred2.flatten())[int(pred2.shape[0] * pred2.shape[1] / 2)]
            pred2[pred2>threshold] = 1
            pred2[pred2<1] = 0
            ciou2,_,_ = evaluator.cal_CIOU(pred2,gt_map,0.5)
            pred3 = 1 - gaussian_now
            threshold = np.sort(pred3.flatten())[int(pred3.shape[0] * pred3.shape[1] / 2)]
            pred3[pred3>threshold] = 1
            pred3[pred3<1] = 0
            ciou3,_,_ = evaluator.cal_CIOU(pred3,gt_map,0.5)
            if write_preds:
                print("Heatmap cIoU: " + str(ciou))
                print("Activation cIoU: " + str(ciou2))
                print("Gaussian cIoU: " + str(ciou3))
                temp = cv2.applyColorMap(np.uint8(gt_map * 255), cv2.COLORMAP_JET)
                Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, temp * 0.5))).convert('RGB').save("tmp/gt.jpg")
                temp2 = cv2.applyColorMap(np.uint8(pred * 255), cv2.COLORMAP_JET)
                Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, temp2 * 0.5))).convert('RGB').save("tmp/pred_heatmap.jpg")
                temp3 = cv2.applyColorMap(np.uint8(pred2 * 255), cv2.COLORMAP_JET)
                Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, temp3 * 0.5))).convert('RGB').save("tmp/pred_activation.jpg")
                temp4 = cv2.applyColorMap(np.uint8(pred3 * 255), cv2.COLORMAP_JET)
                Image.fromarray(np.uint8(np.add((image_now[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, temp4 * 0.5))).convert('RGB').save("tmp/pred_gaussian.jpg")
            print("Done batch!")
    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)
    print('cIoU' , np.sum(np.array(iou) >= 0.5)/len(iou))
    print('auc',auc_)

if __name__ == "__main__":
    main()