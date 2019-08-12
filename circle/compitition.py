# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import cv2 
import time
import pyrealsense2 as rs
from mrcnn.config import Config
from datetime import datetime
from skimage.measure import find_contours
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from disturbance_delete import find_plane 
from disturbance_delete import is_disturbance
from mrcnn import visualize2
from find_ellipse import find_ellipse
from corner_detect import mask2contour
from datetime import datetime
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 29
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =150
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200
 
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config_detect = InferenceConfig()
# Create model object in inference mode.
model_detect= modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_detect)
# Load weights trained on MS-COCO
model_detect.load_weights('./logs/detect_model/mask_rcnn_shapes_0030.h5', by_name=True)
def coordinate_transform(x,y,id,M):
    a=np.array([[[x,y]]],dtype='float32')
    #a=np.array([a])
    true_coordinate=cv2.perspectiveTransform(a,M)
    
    radius={1:0,
            2:4.3,
            3:0,
            4:2.0,
            5:0,
            6:3.3,
            7:3.2,
            8:3.3,
            9:0,
            10:2.6,
            11:0,
            12:3.4,
            13:0,
            14:3.3,
            15:2.6,
            16:3.1,
            17:3.3,
            18:3.3,
            19:3.5,
            20:4.6,
            21:2.7,
            22:3.5,
            23:3,
            24:3,
            25:2.9,
            26:11.5,
            27:3,
            28:2,
            29:2}
    return int(true_coordinate[0][0][0]),int(true_coordinate[0][0][1]-radius[id])

def detectandmeasure(model_detect,pipeline,pipe_profile):
    #开始检测并开始计时
    start=datetime.now()
    detect_class_names = ['BG', 'toilet_soap', 'liquid_soap','toothpaste', 'toilet_water','duck', 'porridge','water','old_godmother','tang','gum','soda','copico','melon_seeds','red_bull','AD_milk','juice',
                'Wanglaoji','Jiaduobao', 'green_tea', 'snow_pear', 'coconut', 'black_tea','coca_cola', 'sprite', 'fenta', 'cookie', 'noodles','tea_pi','fries']
    detect_class_names1=['BG','ZA001','ZA002','ZA003','ZA004','ZA005','ZB001','ZC014','ZB002','ZB004','ZB005','ZB007','ZB008','ZB010','ZC004','ZC005','ZC006',
                        'ZC007','ZC008','ZC010','ZC011','ZC013','ZC009','ZC002','ZC001','ZC003','ZB003','ZB006','ZC012','ZB009']
    # 对齐变量初始化
    align_to_color=rs.align(rs.stream.color)
    # 提前读几帧，避免开始电压不稳，图片不稳定
    #for i in range(25):
    #    frames = pipeline.wait_for_frames()
    #    color_frame = frames.get_color_frame()

    # 等待读入流
    frames = pipeline.wait_for_frames()
    # 深度向彩色对齐
    frames = align_to_color.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Intrinsics & Extrinsics 摄像头内参和外参
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    # 深度传感器初始化
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    # 深度传感器参数调整（目前只会motion_range）
    depth_sensor.set_option(rs.option.motion_range,180)
    # 深度比例
    depth_scale = depth_sensor.get_depth_scale()
    #设置count，用来判断成功了几次完整的检测
    count=0
    while True:
        frame,point_set=find_ellipse(pipeline)
        if frame is None:
            continue
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #if point_set is not None:
        #    for point in point_set:
        #        cv2.circle(np.array(frame),(int(point[0]),int(point[1])),2,(0,255,0),2)   
        #cv2.imshow('corner_point',color_image)
        #cv2.waitKey(10)
        ##当四个点都找到时进行透视变换##################
        if len(point_set)==4:
            dst=np.array([[0, 0], [599, 0],[599, 599],[0, 599]], dtype=np.float32)
            src = np.array([point_set[0],point_set[1],point_set[2],point_set[3]], dtype=np.float32)
            m = cv2.getPerspectiveTransform(src, dst)
            #  使用m矩阵变换，结果为图像大小，使用白色填充
            res = cv2.warpPerspective(
            frame,
            m,
            (549, 549),
            borderValue=(255, 255, 255, 255)
            )
            cv2.imshow("perspective_transform",res)
            cv2.waitKey(10)
            #透视变换结束后开始进行物体检测
            detect_results = model_detect.detect([image], verbose=1)
            r_detect = detect_results[0]
            coordinate=[]
            #寻找桌面所在平面的方程，为后面排除干扰项做准备
            #params=find_plane(point_set,depth_frame,r_detect['rois'],depth_intrin,depth_to_color_extrin)
            for i in range(len(r_detect['class_ids'])):
                x=int((r_detect['rois'][i][1]+r_detect['rois'][i][3])/2)
                contour=mask2contour(r_detect['masks'][:,:,i])
                y_max=0
                x_max=0
                angle=0
                #如果是干扰项，进行下一项判断，不是则输出并添加到坐标中
                #if is_disturbance(params,contour,depth_frame,depth_intrin,depth_to_color_extrin):
                #    continue
                if detect_class_names[r_detect['class_ids'][i]] in ['toothpaste','toilet_soap','snow_pear','tea_pi','tang']:
                    #roi_mat=np.ones((480,640,3),dtype=np.uint8)*255
                    frame_copy=np.array(frame)
                    roi_mat=frame_copy[r_detect['rois'][i][0]:r_detect['rois'][i][2],r_detect['rois'][i][1]:r_detect['rois'][i][3],:].copy()
                    #print('1',frame_copy[r_detect['rois'][i][0]:r_detect['rois'][i][2],r_detect['rois'][i][1]:r_detect['rois'][i][3],:])
                    #print('2',roi_mat)
                    edges = cv2.Canny(cv2.cvtColor(roi_mat,cv2.COLOR_BGR2GRAY),100, 255, apertureSize=3)
                    LINES=cv2.HoughLinesP(edges, 1.0, np.pi / 180, 30,1, 5,15)
                    #print("线段数",len(LINES))
                    line_lenth=0
                    angle_x1=0
                    angle_x2=0
                    angle_y1=0
                    angle_y2=0
                    if LINES is None:
                        pass
                    else:
                        for line in LINES:
                            for x1,y1,x2,y2 in line:
                                if (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)>line_lenth:
                                    line_lenth=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
                                    angle_x1=x1
                                    angle_x2=x2
                                    angle_y1=y1
                                    angle_y2=y2
                            #cv2.line(roi_mat,(x1, y1), (x2, y2), (0, 255,0 ), 3)
                    #cv2.imshow('roi',roi_mat)
                    #cv2.waitKey(10)
                    mat_tmp1=np.array([[[angle_x1,angle_y1]]],dtype='float32')
    
                    true_coordinate1=cv2.perspectiveTransform(mat_tmp1,m)
                    angle_x1_plane=true_coordinate1[0][0][0]
                    angle_y1_plane=true_coordinate1[0][0][1]
                    mat_tmp1=np.array([[[angle_x2,angle_y2]]],dtype='float32')
                    true_coordinate1=cv2.perspectiveTransform(mat_tmp1,m)
                    angle_x2_plane=true_coordinate1[0][0][0]
                    angle_y2_plane=true_coordinate1[0][0][1]
                    angle=math.atan2(angle_y2_plane-angle_y1_plane,angle_x2_plane-angle_x1_plane)
                    angle=math.degrees(angle)-3+180
                    if angle>180:
                        angle=angle-180
                
                for each in contour:
                    if each[0]>y_max:
                        y_max=each[0]
                y=y_max
                x1,y1=coordinate_transform(x,y,r_detect['class_ids'][i],m)
                print(detect_class_names[r_detect['class_ids'][i]])
                #print((x1/10,y1/10), angle)
                radius=math.sqrt((x1-299)*(x1-299)+(y-299)*(y-299))/10
                print(radius)
                coordinate.append([detect_class_names1[r_detect['class_ids'][i]],r_detect['scores'][i],r_detect['rois'][i],[x1,y1,angle],radius])
            #cv2.imshow('result',color_image)  
            #cv2.waitKey(1000)  
                #展示最原始的检测信息（可注释）
                #visualize.display_instances(image, r_detect['rois'], r_detect['masks'], r_detect['class_ids'],detect_class_names, r_detect['scores'])
            count+=1
        if count==1:
            end=datetime.now()
            print('total time:',(end-start).seconds)
            image=visualize2.video_display1(frame,coordinate)
            return image
            break
        
        
        
    