import os, torch, wandb, torch.nn as nn, numpy as np, argparse, cv2, einops, warnings
from torch.optim import *
from torch.nn.functional import grid_sample
from torchvision.transforms import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
from model import AVENet
from datasets import SubSampledFlickr, GetAudioVideoDataset, PerFrameLabels
from sklearn.metrics import auc
from PIL import Image
from flownet2.models import FlowNet2
from losses import FlowLoss
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3,5,6"

train = True
test = False
test_hardway = False
record = False
record_qualitative = False
save = False
selected_whole_qualitative = ['2432219254.mp4', '3484198977.mp4', '3727937033.mp4', '6458319057.mp4', '10409146004.mp4']
if record:
    wandb.init(entity="tonymisic", project="Audio-Visual Tubes",
        config={
            "Model": "Hard Way",
            "dataset": "flickr10k",
            "testset": 69,
            "frames": 16,
            "lr": 1e-6,
            "epochs": 200,
            "batch_size": 20
        }
    )
    wandb.run.name = "16-10k, HardWay, FlowLoss"
    wandb.run.save()

def get_arguments():
    parser = argparse.ArgumentParser()
    # from testing code
    parser.add_argument('--testset',default='flickr',type=str,help='testset,(flickr or vggss)') 
    parser.add_argument('--data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--og_data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--image_size',default=256,type=int,help='Height and width of inputs')
    parser.add_argument('--gt_path',default='',type=str)
    parser.add_argument('--og_gt_path',default='',type=str)
    parser.add_argument('--summaries_dir',default='',type=str,help='Model path')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch Size')
    parser.add_argument('--epsilon', default=0.65, type=float, help='pos')
    parser.add_argument('--epsilon2', default=0.4, type=float, help='neg')
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg',action='store_true')
    parser.set_defaults(Neg=True)
    # from training code
    parser.add_argument('--learning_rate',default=1e-6,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads',default=4,type=int,help='Number of threads for multi-thread loading')
    parser.add_argument('--epochs',default=200,type=int,help='Number of total epochs to run')
    parser.add_argument('--frame_density',default=16,type=int,help='Training frame sampling density')
    # new arguments
    parser.add_argument('--sampling_rate', default=16, type=int,help='Sampling rate for frame selection')
    # flownet args
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    return parser.parse_args() 

def save_image(image, recording_name, pred=None, gt_map=None):
    image = normalize_img(image)
    temp = cv2.applyColorMap(np.uint8(gt_map * 128), cv2.COLORMAP_JET)
    temp2 = cv2.applyColorMap(np.uint8(pred * 255), cv2.COLORMAP_JET)
    wandb.log({recording_name:
        wandb.Image(
            Image.fromarray(np.uint8(np.add((image[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.4, np.add(temp * 0.5, temp2 * 0.5) * 0.6))).convert('RGB')
        )
    })
def save_labels(image, recording_name, gt_map=None):
    image = normalize_img(image)
    temp = cv2.applyColorMap(np.uint8(gt_map * 255), cv2.COLORMAP_JET)
    final = Image.fromarray(np.uint8(np.add((image[0].cpu().numpy() * 255).transpose((1,2,0)) * 0.5, temp * 0.5))).convert('RGB')
    final.save("tmp/" + recording_name + ".jpg")

def main():
    args = get_arguments()

    device = torch.device("cuda") 
    model = AVENet(args)
    model = model.cuda() 
    model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load('pretrained/lvs_soundnet.pth.tar')
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    flow_model = FlowNet2(args)
    flow_model = flow_model.cuda() 
    flow_model = nn.DataParallel(flow_model)
    flow_model.to(device)
    print('load pretrained')
    checkpoint = torch.load('flownet2/FlowNet2_checkpoint.pth.tar')
    flow_model.module.load_state_dict(checkpoint['state_dict'])
    flow_model.eval()
    # init datasets
    dataset = SubSampledFlickr(args,  mode='train', subset=10)
    testdataset = PerFrameLabels(args, mode='test')
    original_testset = GetAudioVideoDataset(args, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    originaldataloader = DataLoader(original_testset, batch_size=1, shuffle=False, num_workers=args.n_threads)
    # loss
    criterion = nn.CrossEntropyLoss()
    criterion2 = FlowLoss(14 * 14)
    print("Loaded dataloader and loss function.")
    # Optimizers
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Optimizer loaded.")
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[60,100,150,180], gamma=0.1)
    if record:
        wandb.watch(model, optim, log="all", log_freq=1000)
    
    for epoch in range(args.epochs):
        if train:
            model.train()
            running_loss = 0.0
            for step, (frames, spec, _, _, name) in enumerate(dataloader):
                print("Training Step: " + str(step) + "/" + str(len(dataloader)))
                # flow prediction from FlowNet2
                flow_frames = torch.cat((einops.rearrange(frames[:, :, :-1, :, :], 'b c t h w -> (b t) c h w').unsqueeze(2),
                                        einops.rearrange(frames[:, :, 1:, :, :], 'b c t h w -> (b t) c h w').unsqueeze(2)), dim=2)
                flows = flow_model(flow_frames).permute(0,2,3,1)
                # Vanilla Hard Way Loss
                spec = Variable(spec).cuda()
                spec = einops.rearrange(spec.unsqueeze(2).repeat(1, 1, 16, 1, 1), 'b c t h w -> (b t) c h w')
                frames = einops.rearrange(frames, 'b c t h w -> (b t) c h w')
                heatmap, out, _, _ = model(frames.float(), spec.float())
                target = torch.zeros(out.shape[0]).cuda().long()
                hardway_loss = criterion(out, target)
                # Flow loss from: "Unsupervised Temporal Consistency Metric for Video Segmentation in Highly-Automated Driving"
                heatmap = heatmap.reshape(args.batch_size,args.frame_density, 16, 16)
                heatmap = torchvision.transforms.Resize(size=args.image_size)(heatmap)
                heatmap = normalize_img(-heatmap)
                pred = 1 - heatmap
                cpu_pred = pred.detach().cpu()
                threshold = np.sort(cpu_pred.flatten())[int(cpu_pred.nelement() / 2)]
                pred[pred>threshold] = 1
                pred[pred<1] = 0
                skewed_pred = grid_sample(einops.rearrange(pred[:,:-1,:], 'b t h w -> (b t) h w').unsqueeze(1), flows, mode='bilinear')
                original_pred = einops.rearrange(pred[:,1:,:], 'b t h w -> (b t) h w').unsqueeze(1)
                skewed_pred[0:15]
                original_pred[0:15]
                frames[0:16]
                flow2img(flows[0])
                optim.zero_grad()
                hardway_loss.backward()
                optim.step()
                running_loss += float(hardway_loss)

            final_loss = running_loss / float(step + 1)
            print("Epoch " + str(epoch) + " training done.")
            scheduler.step()
            if record:
                wandb.log({ "loss": final_loss
                })
        
        if test:
            with torch.no_grad():
                model.eval()
                ious,aucs, mTCs = [], [], []
                for step, (frames, spec, _, _, name) in enumerate(testdataloader):
                    print("Testing Step: " + str(step) + "/" + str(len(testdataloader)))
                    iou, preds, gt_maps = [], [], []
                    for i in range(args.sampling_rate, frames.size(2) - 1, args.sampling_rate):
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
                        preds.append(pred)
                        gt_maps.append(gt_map)
                        iou.append(ciou)
                        if name[0] in selected_whole_qualitative and record_qualitative:
                            save_labels(frames[:,:,i,:,:].float(), name[0] + "_pred" + str(i), pred)
                    mTCs.append(float(mTC(preds, gt_maps)))
                    results = []
                    for i in range(21):
                        result = np.sum(np.array(iou) >= 0.05 * i)
                        result = result / len(iou)
                        results.append(result)
                    x = [0.05 * i for i in range(21)]
                    auc_ = auc(x, results)
                    ious.append(np.sum(np.array(iou) >= 0.5) / len(iou))
                    aucs.append(auc_)
                print("Testing cIoU ", np.sum(ious) / len(ious))
                print("Testing auc ", np.sum(aucs) / len(aucs))
                print("Testing mTC ", np.sum(mTCs) / len(mTCs))
                if record:
                    wandb.log({ "Testing cIoU": np.sum(ious) / len(ious), "Testing AUC": np.sum(aucs) / len(aucs), "Testing mTC": np.sum(mTCs) / len(mTCs)})
        
        if test_hardway:
            with torch.no_grad():
                model.eval()
                iou = []
                for step, (image, spec, _, name, _) in enumerate(originaldataloader):
                    print('%d / %d' % (step,len(originaldataloader) - 1))
                    spec = Variable(spec).cuda()
                    image = Variable(image).cuda()
                    heatmap,_,_,_ = model(image.float(),spec.float())
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
                    wandb.log({ "Hardway Test cIoU": np.sum(np.array(iou) >= 0.5)/len(iou), "Hardway Test AUC": auc_})

        
        if save:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, args.summaries_dir + 'model_16frm_10k_ep%s.pth.tar' % (str(epoch)) 
            )
if __name__ == "__main__":
    main()