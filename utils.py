import csv
import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
from scipy import stats
from sklearn import metrics
import sklearn
import numpy as np, math
import xml.etree.ElementTree as ET

TAG_CHAR = np.array([202021.25], np.float32)
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel
def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img
def flow2img(flow_data):
	"""
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
	# print(flow_data.shape)
	# print(type(flow_data))
	u = flow_data[:, :, 0]
	v = flow_data[:, :, 1]

	UNKNOW_FLOW_THRESHOLD = 1e7
	pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
	pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
	idx_unknown = (pr1 | pr2)
	u[idx_unknown] = v[idx_unknown] = 0

	# get max value in each direction
	maxu = -999.
	maxv = -999.
	minu = 999.
	minv = 999.
	maxu = max(maxu, np.max(u))
	maxv = max(maxv, np.max(v))
	minu = min(minu, np.min(u))
	minv = min(minv, np.min(v))

	rad = np.sqrt(u ** 2 + v ** 2)
	maxrad = max(-1, np.max(rad))
	u = u / maxrad + np.finfo(float).eps
	v = v / maxrad + np.finfo(float).eps

	img = compute_color(u, v)

	idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
	img[idx] = 0

	return np.uint8(img)

class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer>=thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value

def testset_gt(args,name):

    if args.testset == 'flickr':
        gt = ET.parse(args.og_gt_path + '%s.xml' % name[:-4]).getroot()
        gt_map = np.zeros([224,224])
        bboxs = []
        for child in gt: 
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index,ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map>1] = 1

    elif args.testset == 'vggss':
        gt = args.gt_all[name[:-4]]
        gt_map = np.zeros([224,224])
        for item_ in gt:
            item_ =  list(map(lambda x: int(224* max(x,0)), item_) )
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[ymin:ymax,xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map>0] = 1
    return gt_map

def testset_gt_frame(args,name,frame):

    if args.testset == 'flickr':
        gt = ET.parse(args.og_gt_path + '%s_%s.xml' % (name[:-4], str(frame))).getroot()
        gt_map = np.zeros([224,224])
        bboxs = []
        for child in gt: 
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index,ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
            gt_map += temp
        #gt_map /= 2
        #gt_map[gt_map>1] = 1
        
    elif args.testset == 'vggss':
        gt = args.gt_all[name[:-4]]
        gt_map = np.zeros([224,224])
        for item_ in gt:
            item_ =  list(map(lambda x: int(224* max(x,0)), item_) )
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[ymin:ymax,xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map>0] = 1
    return gt_map

def mTC(predictions, gt_maps):
    assert len(predictions) == len(gt_maps)
    cious = torch.zeros(len(predictions) - 1)
    for i in range(len(predictions) - 1):
        evaluator = Evaluator()
        ciou,_,_ = evaluator.cal_CIOU(predictions[i], predictions[i + 1], 0.5)
        cious[i] = ciou
    return torch.div(torch.sum(cious, dim=0), len(predictions) - 1)
    
    # gives ~96% acc without training
    # assert len(predictions) == len(gt_maps)
    # cious = torch.zeros(len(predictions))
    # for i in range(len(predictions)):
    #     evaluator = Evaluator()
    #     ciou,_,_ = evaluator.cal_CIOU(predictions[i], gt_maps[i], 0.5)
    #     cious[i] = ciou
    # return 1 - torch.div(torch.sum(torch.abs(torch.diff(cious, dim=0))), len(predictions) - 1)