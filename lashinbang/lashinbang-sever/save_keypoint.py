import pickle
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import math
import cv2
import copy
from datetime import datetime
from models.matching import Matching
from models.superpoint import SuperPoint
from models.utils import (convert_img_to_tensor, make_draw_matches)
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
RANSAC_THRESH = 10
SIZE_0= 720
SIZE_1= 400

def resize_rect(src, img_size=480):
    MAX_SIZE = (img_size, img_size)

    xscale = MAX_SIZE[0] / src.shape[0]
    yscale = MAX_SIZE[1] / src.shape[1]
    scale = min(xscale, yscale)
    if scale > 1:
        return src
    dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
    return dst

def resize_prop_rect(src, img_size=SIZE_0):
        MAX_SIZE = (img_size, img_size)

        xscale = MAX_SIZE[0] / src.shape[0]
        yscale = MAX_SIZE[1] / src.shape[1]
        scale = min(xscale, yscale)
        if scale > 1:
            return src
        dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
        return dst

if __name__ == '__main__':
    name_method="rmac"
    config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)


    with open(f'/media/anlab/data/kbooks_bl/file_pickle/{name_method}/dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)[:]
    # kps=[]
    for i,path in enumerate(tqdm(dataset)):
        image1 = cv2.imread('/media/anlab/data/kbooks_bl/'+path)
        try:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        except:
            print(f"err {path}")
        gray_second = resize_prop_rect(gray1, img_size=SIZE_1)
        inp1 = convert_img_to_tensor(gray_second, device)
        with torch.no_grad():
            pred1 = superpoint({'image': inp1})
        # pred1=pred1
        # pred1['keypoints']=pred1['keypoints'][0].cpu().detach().numpy()
        # pred1['scores']=pred1['scores'][0].cpu().detach().numpy()
        # pred1['descriptors']=pred1['descriptors'][0].cpu().detach().numpy()
        # print(pred1.keys())
        # print(len(pred1))
        # kps.append(pred1)
       
        with open(f"lashinbang-server/save_kp_data/kps{i}.pickle","wb") as f:
            pickle.dump(pred1,f)


