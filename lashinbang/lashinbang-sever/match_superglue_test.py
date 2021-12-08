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
from models.matching import Matching
from models.utils import (convert_img_to_tensor, make_draw_matches)
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from datetime import datetime
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
RANSAC_THRESH = 10
SIZE_0= 720
SIZE_1= 400
class SuperglueMatching:
    def __init__(self, first_img, second_img = None, threshold_area=0.19,index_element=None):
        self.gray_first = self.resize_prop_rect(first_img, img_size=SIZE_0)
        print("gray_first",self.gray_first.shape)
        self.inp0 = convert_img_to_tensor(self.gray_first, device)
        self.threshold_area = threshold_area
        self.index_element=index_element
        self.corner_points_img = np.array(
                    [[(0, 0), (first_img.shape[1], 0), (first_img.shape[1], first_img.shape[0]), (0, first_img.shape[0])]],
                    np.float32)
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
        print('Running inference on device \"{}\"'.format(device))
        if second_img is not None:
            self.gray_second = self.resize_prop_rect(second_img,  img_size=SIZE_1)
            self.inp1 = convert_img_to_tensor(self.gray_second, device)
        else:
            self.gray_second = None
            self.inp1 = None
        self.matching = Matching(config,index_element).eval().to(device)
        # self.matching.index_element= index_element
        self.vis = None

    #find match 
    def find_matches_superglue(self, second_img, debug=True):
        if second_img is not None:
            self.gray_second = self.resize_prop_rect(second_img,  img_size=SIZE_1)
            print("gray2", self.gray_second.shape)
            self.inp1 = convert_img_to_tensor(self.gray_second, device)
        ret_matches = False
        
        if self.inp1 is None:
            return ret_matches, None, None, None, None
        # print("souce shape" , self.gray_first.shape,self.gray_second .shape )
        pred =self.matching({'image0': self.inp0, 'image1': self.inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        self.kpts0, self.kpts1 = pred['keypoints0'], pred['keypoints1']
        self.matches, self.conf = pred['matches0'], pred['matching_scores0']
        # Keep the matching keypoints.
        valid = self.matches > -1
        mkpts0 = self.kpts0[valid]
        
        ret_matches, score_matches = self.is_relevant_superglue()
        if debug:
            self.vis = self.draw_matches()
            cv2.imwrite("output.jpg" , self.vis )
        
        return ret_matches, score_matches

    @staticmethod
    def resize_prop_rect(src, img_size=SIZE_0):
        MAX_SIZE = (img_size, img_size)

        xscale = MAX_SIZE[0] / src.shape[0]
        yscale = MAX_SIZE[1] / src.shape[1]
        scale = min(xscale, yscale)
        if scale > 1:
            return src
        dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
        return dst

    @staticmethod
    def intersect(i_a, i_b, i_c, i_d):
        def ccw(c_a, c_b, c_c):
            return (c_c[1] - c_a[1]) * (c_b[0] - c_a[0]) > (c_b[1] - c_a[1]) * (c_c[0] - c_a[0])

        return ccw(i_a, i_c, i_d) != ccw(i_b, i_c, i_d) and ccw(i_a, i_b, i_c) != ccw(i_a, i_b, i_d)

    def is_convex(self):
        points = self.transformed_corner_points[0]
        # print("points",points)
        # print(" a",points[-4])
        # print(" b",points[-4+2])
        # print(" c",points[-4+1])
        # print(" d",points[-4+3])
        for i in range(-4, 0):
            if not self.intersect(points[i], points[i+2], points[i+1], points[i+3]):
                return False
        return True

    @staticmethod
    def angle_of_3_points(a, b, c):
        """
        Calculate angle abc of 3 points
        :param a:
        :param b:
        :param c:
        :return:
        """
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def angle_conditions(self):
        points = self.transformed_corner_points[0]
        for i in range(0, 4):
            a = points[i % 4]
            b = points[(i+1) % 4]
            c = points[(i+2) % 4]
            angle = self.angle_of_3_points(a, b, c)
            print("Angle: ", angle)
            if angle > 160 or angle < 20:
                return False
        return True
    
    @staticmethod
    def order_points(pts):
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        D = distance.cdist(tl[np.newaxis], right_most, "euclidean")[0]
        (br, tr) = right_most[np.argsort(D)[::-1], :]
        return np.array([tl, tr, br, bl], dtype="float32")

    def is_relevant_superglue(self):
        score_matches = 0
        valid = self.matches > -1
        mkpts0 = self.kpts0[valid]
        print("mkpts0",mkpts0.shape)
        mkpts1 = self.kpts1[self.matches[valid]]
        print("mkpts1",mkpts1.shape)

        if len(mkpts0) < 8:
            # print("matches size " , len(mkpts0))
            return False, score_matches
        src_pts = []
        dst_pts = []
        h, w  = self.gray_first.shape[:2]

        #######add code here################
        print("self.gray_second",self.gray_second.shape)
        h2,w2= self.gray_second.shape
        ####################################

        

        # mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
        box_max = [w, h , 0 , 0]
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # pt_1 = (int(x0), int(y0))
            # pt_2 = (int(x1) , int(y1))
            pt_1 = (x0, y0)
            pt_2 = (x1 , y1)
            src_pts.append(pt_1)
            dst_pts.append(pt_2)
            box_max[0] = min(box_max[0] , x0)
            box_max[1] = min(box_max[1] , y0)
            box_max[2] = max(box_max[2] , x0)
            box_max[3] = max(box_max[3] , y0)
        score_matches = (box_max[2] - box_max[0])*(box_max[3] - box_max[1])/(w*h)

        ###########add code here##########################
        src_pts2 = []
        dst_pts2 = []
        box_max2 = [w2, h2 , 0 , 0]

        for (x0, y0), (x1, y1) in zip(mkpts1, mkpts0):
            # pt_1 = (int(x0), int(y0))
            # pt_2 = (int(x1) , int(y1))
            pt_1 = (x0, y0)
            pt_2 = (x1 , y1)
            src_pts2.append(pt_1)
            dst_pts2.append(pt_2)
            box_max2[0] = min(box_max2[0] , x0)
            box_max2[1] = min(box_max2[1] , y0)
            box_max2[2] = max(box_max2[2] , x0)
            box_max2[3] = max(box_max2[3] , y0)
        score_matches2 = (box_max2[2] - box_max2[0])*(box_max2[3] - box_max2[1])/(w2*h2)


        if(score_matches2 > score_matches):
            score_matches= score_matches2
            src_pts=src_pts2
            dst_pts=dst_pts2
            box_max=box_max2
            h=h2
            w=w2
        ##################################################
        # print("score_matches",score_matches)
        if score_matches < self.threshold_area:
            print("checkkkkkkkkkkkk")
            # print("area_matches" , score_matches)    
            return False, score_matches
        h_matrix, status = cv2.findHomography( np.float32(src_pts), np.float32(dst_pts), cv2.RANSAC, RANSAC_THRESH)
        self.corner_points_img = np.array(
                    [[(box_max[0], box_max[1]), (box_max[2], box_max[1]), (box_max[2], box_max[3]), (box_max[0], box_max[3])]],
                    np.float32)
        self.transformed_corner_points = cv2.perspectiveTransform(self.corner_points_img, h_matrix)

        rect = cv2.minAreaRect(self.transformed_corner_points)
        rotated_box = self.order_points(cv2.boxPoints(rect))

        side_length = [euclidean(rotated_box[0], rotated_box[1]), euclidean(rotated_box[0], rotated_box[-1])]
        side_length_ratio = max(side_length) / min(side_length)
        # print("side_length_ratio" ,side_length_ratio)
        points_4 = self.transformed_corner_points[0]
        drawing = cv2.cvtColor(self.gray_second, cv2.COLOR_GRAY2RGB)
        for i in range(-1, 3):
            cv2.line(drawing, (points_4[i][0], points_4[i][1]), (points_4[i + 1][0], points_4[i + 1][1]),
                     (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite("out_boxes.jpg" ,drawing )
        if  not self.is_convex():
            print("is_ordered or is_convex")
            return False, score_matches

        if not self.angle_conditions():
            print("angle")
            return False, score_matches
        
        return True	, score_matches

    def draw_matches(self):
        image0 = cv2.cvtColor(self.gray_first, cv2.COLOR_GRAY2RGB)
        image1 = cv2.cvtColor(self.gray_second, cv2.COLOR_GRAY2RGB)
        rot0, rot1 = 0, 0
        # Visualize the matches.
        valid = self.matches > -1
        
        mkpts0 = self.kpts0[valid]
        mkpts1 = self.kpts1[self.matches[valid]]
        mconf = self.conf[valid]
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(self.kpts0), len(self.kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format("image1", 'image2'),
        ]
        img_match = make_draw_matches(image0, image1, self.kpts0, self.kpts1, mkpts0, mkpts1, color,
            text, small_text=small_text)

        return img_match

def resize_rect(src, img_size=480):
    MAX_SIZE = (img_size, img_size)

    xscale = MAX_SIZE[0] / src.shape[0]
    yscale = MAX_SIZE[1] / src.shape[1]
    scale = min(xscale, yscale)
    if scale > 1:
        return src
    dst = cv2.resize(src, None, None, scale, scale, cv2.INTER_LINEAR)
    return dst
if __name__ == '__main__':
    import pickle
    name_method="rmac"
    # with open(f'/media/anlab/data/lashinbang/lashinbang-server/query_new.pickle', 'rb') as f:
    #     query = pickle.load(f)
    with open(f'/media/anlab/data/kbooks_bl/file_pickle/{name_method}/dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)[:]
    with open(f'/media/anlab/data/kbooks_bl/file_pickle/{name_method}/query.pickle', 'rb') as f:
        query = pickle.load(f)
    
        # print("query",len(query))
        # print("query",query[69])
    with open(f'/media/anlab/data/lashinbang/lashinbang-server/bbs_new.pickle', 'rb') as f:
        bbs = pickle.load(f)
        # print("bbs",len(bbs))
        # print("bbs",bbs[69])
    with open(f'/media/anlab/data/lashinbang/lashinbang-server/image_correspond.pickle', 'rb') as f:
        image_correspond = pickle.load(f)
    list_bad= os.listdir("/media/anlab/data/lashinbang/lashinbang-server/matching_NG")
    # print("list_bad",len(list_bad))
    # print("list_bad",list_bad[0])
    list_bad_new=[]
    bbxs=[]
    for i,value in enumerate(query):
        if(value.split("/")[-1] in list_bad):
            list_bad_new.append(value)
            bbxs.append(bbs[i])
    # print("len", len(list_bad_new))
    # bbxs=bbs
    # list_bad_new=query
    for i,value in enumerate(list_bad_new[11:12]):
        now = datetime.now().time()
        i=11
        # print("value",value)
        index= query.index(value)
        # print("index",index)
        image0_ = cv2.imread('/media/anlab/data/kbooks_bl/'+query[index])
        image0=resize_rect(image0_,img_size=SIZE_0)
        # print("image0", image0.shape)


        # if(image0.shape[0]<480):
        #     image0=cv2.copyMakeBorder(image0, 0, 480-image0.shape[0], 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
        
        # width=image0.shape[1]
        # height=image0.shape[0]
        # x= bbxs[i][0]
        # # print("x",x)
        # w=bbxs[i][2]- bbxs[i][0]
        # # print("w",w)
        # y= bbxs[i][1]
        # h= bbxs[i][3]-bbxs[i][1]
        # image0_crop = image0_[y:y+h, x:x+w]
        # # print("image0_crop",image0_crop.shape)
        # image0_crop = resize_rect(image0_crop)
        # # print("check image0_crop",image0_crop.shape)



        # # width_=max(width,image0_crop.shape[1])
        # width_=image0_crop.shape[1]
        # image0= cv2.copyMakeBorder(image0, 0, 0, 0, width_, cv2.BORDER_CONSTANT, None, value = 0)
        # # print("shape img0_0",image0.shape)

        # # print("width",width)
        # # print("image0_crop.shape[0]",image0_crop.shape[0])
        # # print("image0_crop.shape[1]",image0_crop.shape[1])
        # # print("image0.shape[1]",image0.shape[1])
        # image0[0:0+image0_crop.shape[0], width:image0_crop.shape[1]+image0.shape[1]]=image0_crop
        # # print("shape img0_1",image0.shape)
        # max_size0=max(image0.shape[0],image0.shape[1]) 

        scores=[]
        result_new=[]
        Ms=[]
        ret_matchess=[]
        name= query[index].split("/")[-1].split(".")[0]

        if not os.path.exists(f"/media/anlab/data/lashinbang/lashinbang-server/out_put_test/{i}_{name}"):
            os.mkdir(f"/media/anlab/data/lashinbang/lashinbang-server/out_put_test/{i}_{name}")
        # print("name",name)

        # exit()
        for j,element in enumerate(image_correspond[index][:1]):
            print("checkkkkk",dataset.index(element))
            index_element= dataset.index(element)
            # print("/media/anlab/data/kbooks_bl/"+element)
            image1 = cv2.imread('/media/anlab/data/kbooks_bl/'+element)
            # image1=cv2.resize(image1,(max_size0,max_size0))

            gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            
            M = SuperglueMatching(gray0,index_element=index_element)
            # M.index_element = index_element
            Ms.append(M)
            ret_matches, score = M.find_matches_superglue( gray1, debug=True)
            print("ret_matches", ret_matches, score )
            now1 = datetime.now().time()
            print("now",now)
            print("now1",now1)
            scores.append(score)
            ret_matchess.append(ret_matches)
            
          
        scores=np.array(scores)
        idxs= np.argsort(-scores)
        print("idxs",idxs)
        check=[1,0,0,2,0,0,3]
        # for k,elem in enumerate(idxs):
        #     # print("elem",elem)
        #     cv2.imwrite(f"/media/anlab/data/lashinbang/lashinbang-server/out_put_test/{i}_{name}/{elem}_{ret_matchess[elem]}_{scores[elem]}.jpg" , Ms[elem].vis)
        print(scores)



