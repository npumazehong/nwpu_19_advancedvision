# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 
import time
from mrcnn.config import Config
from datetime import datetime
from skimage.measure import find_contours
from line_detect import line_detect
# Root directory of the project
#ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

def takeSecond(elem):
    return elem[1]
def max_confidence(corners):
    corners.sort(key=takeSecond)
    return corners[-1]
def mask2contour(mask):
    contour=[]
    print("######################################################")
    #print(mask.shape)
    #print(mask)
    for i in range(0,640,2):
        for j in range(0,480,2):
            #print(mask[j,i])
            if mask[j,i]==1:
                contour.append([j,i])
    return contour
    """


    #print(mask)
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    #print(padded_mask)
    contours = find_contours(padded_mask, 0.5)
    #print("next is contours")
    #print(contours)
    return contours[0]
"""
"""
def masks2contours_smooth(masks,ids,image):
    contours=[]
    for i in range(len(ids)):
        contours.append(mask2contour(masks[:,:,i]))
    #print(type(contours))
    #print(contours)
    smooth_contours=[]
    for each in contours:
        smooth_contours.append(cv2.GaussianBlur(each,(3,3),0))
"""   


def accurate_corner(contours,flag):
    if flag==1:
        #line1:  y=-x
        candidate=[]
        for i in range(len(contours)):
            current_distance=abs(contours[i][0]+0.5*contours[i][1])/math.sqrt(1.25)
            
            #此处为五个点中进一步选取，可用作后期修正
            if len(candidate)<=5:
                candidate.append([i,current_distance])
                candidate.sort(key=takeSecond)
            elif current_distance<candidate[-1][1]:
                candidate[-1]=[i,current_distance]
                candidate.sort(key=takeSecond)
        index=0
        for j in range(len(candidate)):
            if j==0:
                y_min=contours[candidate[j][0]][0]
            elif contours[candidate[j][0]][0]<y_min:
                y_min=contours[candidate[j][0]][0]
                index=j
            else:
                continue
        candidate=candidate[index]
    elif flag==2:
        #line1:  y=x+480
        candidate=[]
        for i in range(len(contours)):
            current_distance=abs(contours[i][0]-contours[i][1]-480)/math.sqrt(2)
            #此处为五个点中进一步选取，可用作后期修正
            if len(candidate)<=5:
                candidate.append([i,current_distance])
                candidate.sort(key=takeSecond)
            elif current_distance<candidate[-1][1]:
                candidate[-1]=[i,current_distance]
                candidate.sort(key=takeSecond)
        index=0
        for j in range(len(candidate)):
            if j==0:
                x_min=contours[candidate[j][0]][1]
            elif contours[candidate[j][0]][1]<x_min:
                x_min=contours[candidate[j][0]][1]
                index=j
            else:
                continue
        candidate=candidate[index]
            
        """
            #直接取最短距离
            if i==0:
                min_distance1=current_distance
                candidate=[0,min_distance1]
            elif current_distance<min_distance1:
                min_distance1=current_distance
                candidate=[i,min_distance1]
            else:
                continue
        """
    elif flag==3:
        #line1:  y=-x+1120
        candidate=[]
        for i in range(len(contours)):
            current_distance=abs(contours[i][0]+contours[i][1]-1120)/math.sqrt(2)
            #直接取最短距离
            if i==0:
                min_distance1=current_distance
                candidate=[0,min_distance1]
            elif current_distance<min_distance1:
                min_distance1=current_distance
                candidate=[i,min_distance1]
            else:
                continue
    elif flag==4:
        #line4:  y=x-640
        candidate=[]
        for i in range(len(contours)):
            current_distance=abs(contours[i][0]-contours[i][1]+640)/math.sqrt(2)
            
            #此处为五个点中进一步选取，可用作后期修正
            if len(candidate)<=5:
                candidate.append([i,current_distance])
                candidate.sort(key=takeSecond)
            elif current_distance<candidate[-1][1]:
                candidate[-1]=[i,current_distance]
                candidate.sort(key=takeSecond)
        index=0
        for j in range(len(candidate)):
            if j==0:
                y_min=contours[candidate[j][0]][0]
            elif contours[candidate[j][0]][0]<y_min:
                y_min=contours[candidate[j][0]][0]
                index=j
            else:
                continue
        candidate=candidate[index]
    return int(contours[candidate[0]][1]), int(contours[candidate[0]][0])
def corner_point_in_houghlines(corner,lines):
    threshold=1000
    x_final=0
    y_final=0
    #print(lines)
    if lines is None:
        print('no lines detected')
    else:
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:

                if y2<y1:
                    a=x1
                    x1=x2
                    x2=a
                    b=y1
                    y1=y2
                    y2=b
                if ((x2-corner[0])*(x2-corner[0])+(y2-corner[1])*(y2-corner[1]))<threshold and abs((y2-y1)/(x2-x1))>0.2:
                    x_final=x1
                    y_final=y1
                    threshold=(x2-corner[0])*(x2-corner[0])+(y2-corner[1])*(y2-corner[1])
                    #print([x2,y2])
                    #print('----->')
                    #print([x1,y1])
    if x_final==0 and y_final==0:
        return False,0,0
    else:
        return True,x_final,y_final


    


def corner_points(image,model_corner):
    #config_corner = InferenceConfig()
    # Create model object in inference mode.
    #model_corner= modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_corner)
    # Load weights trained on MS-COCO
    #model_corner.load_weights('./logs/corner_model/mask_rcnn_shapes_0006.h5', by_name=True)
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    corner_class_names = ['BG', 'corner1','corner2','corner3','corner4']
    # Load a random image from the images folder
    #file_names = next(os.walk(IMAGE_DIR))[2]
    # Run detection
    corner_results = model_corner.detect([image], verbose=1)

    r_corner = corner_results[0]


    print(r_corner['class_ids'])

    corners1=[]
    corners2=[]
    corners3=[]
    corners4=[]
    four_corners=[]
    for i in range(len(r_corner['class_ids'])):
            if r_corner['class_ids'][i]==1:
                corners1.append([r_corner['class_ids'][i],r_corner['scores'][i],r_corner['rois'][i],r_corner['masks'][:,:,i]])
            elif r_corner['class_ids'][i]==2:
                corners2.append([r_corner['class_ids'][i],r_corner['scores'][i],r_corner['rois'][i],r_corner['masks'][:,:,i]])
            elif r_corner['class_ids'][i]==3:
                corners3.append([r_corner['class_ids'][i],r_corner['scores'][i],r_corner['rois'][i],r_corner['masks'][:,:,i]])
            elif r_corner['class_ids'][i]==4:
                corners4.append([r_corner['class_ids'][i],r_corner['scores'][i],r_corner['rois'][i],r_corner['masks'][:,:,i]])
    four_corners.append(corners1)
    four_corners.append(corners2)
    four_corners.append(corners3)
    four_corners.append(corners4)
    ##此处先自己估计直线来进行距离比较，之后可以采用与对角线垂直的直线
    point_set={}
    left_corner=[]
    right_corner=[]
    for i in range(len(four_corners)):
        if four_corners[i]!=[]:
            corner1=max_confidence(four_corners[i])
            x=int(corner1[2][1]+corner1[2][3])/2
            y=int(corner1[2][0]+corner1[2][2])/2
            
            print('result')
            print((x,y))
            point_set['corner%d'%(i+1)]=[x,y]
            if i==1:
                left_corner.append([x,y])
            if i==2:
                right_corner.append([x,y])
                
            #cv2.circle(image,(x,y),2,(0,255,0),2)
        else:
            print("masks detect no corner%d"%(i+1))
    houghlines=line_detect(image)
    if 'corner1' not in point_set :
        if left_corner!=[]:
            print('left_corner is')
            print(left_corner)
            flag,x,y=corner_point_in_houghlines(left_corner[0],houghlines)
            if flag==True:
                point_set["corner1"]=[x,y]
            else:
                print("failed to find corner1")
    if 'corner4' not in  point_set:
        if right_corner!=[]:
            flag,x,y=corner_point_in_houghlines(right_corner[0],houghlines)
            if flag==True:
                point_set["corner4"]=[x,y]
            else:
                print("failed to find corner4")
            

        else:
            print("no corne2,so can't find corner1")
                
    """
    for i in range(len(four_corners)):
        if four_corners[i]!=[]:
            corner1=max_confidence(four_corners[i])
            contour1=mask2contour(corner1[3])
            #print(contour1)
            x,y=accurate_corner(contour1,i+1)
            print('result')
            print((x,y))
            point_set.append([x,y])
            #cv2.circle(image,(x,y),2,(0,255,0),2)
        else:
            print("no corner%d"%(i+1))
    """
    print(point_set)
    return point_set