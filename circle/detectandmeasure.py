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
import pyrealsense2 as rs
from mrcnn.config import Config
from datetime import datetime
from skimage.measure import find_contours
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from corner_detect import corner_points
from corner_detect import mask2contour
from corner_config import Inference1Config

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
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
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
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

if __name__=='__main__':
    start=datetime.now()
    config_detect = InferenceConfig()
    config_corner = Inference1Config()
    # Create model object in inference mode.
    model_detect= modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_detect)
    model_corner= modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_corner)
    # Load weights trained on MS-COCO
    model_detect.load_weights('./logs/detect_model/mask_rcnn_shapes_0030.h5', by_name=True)
    model_corner.load_weights('./logs/corner_model/mask_rcnn_shapes_0027.h5', by_name=True)
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    detect_class_names = ['BG', 'toilet_soap', 'liquid_soap','toothpaste', 'toilet_water','duck', 'porridge','water','old_godmother','tang','gum','soda','copico','melon_seeds','red_bull','AD_milk','juice',
                'Wanglaoji','Jiaduobao', 'green_tea', 'snow_pear', 'coconut', 'black_tea','coca_cola', 'sprite', 'fenta', 'cookie', 'noodles','tea_pi','fries']
    # Load a random image from the images folder
    


    """
    
    file_names = next(os.walk(IMAGE_DIR))[2]



    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipe_profile=pipeline.start(config)
    align_to_color=rs.align(rs.stream.color)

    for i in range(3):
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    #摄像头参数
    frames = pipeline.wait_for_frames()
    frames = align_to_color.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    #消除开始时绿色
    

    while True:
        
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        #cv2.imshow("BGR",frame)
        frame1=image.copy()
        # Run detection

        detect_results = model_detect.detect([image], verbose=1)####verbode用不用改

        r_detect = detect_results[0]

        #masks2contours_smooth(r_detect['masks'],r_detect['class_ids'],frame)


        point_set=corner_points(frame1,model_corner)

        #打印点的信息
        for key in point_set:
            print(key,rs.rs2_deproject_pixel_to_point(depth_intrin, [int(point_set[key][0]),int(point_set[key][1])], depth_frame.get_distance(int(point_set[key][0]),int(point_set[key][1]))))
        #cv2.imshow("frame1",frame1)
        #cv2.imshow("RGB",image)
        ##透视变换##################
        dst=np.array([[0, 0], [549, 0],[0, 549],[549, 549] ], dtype=np.float32)
        if len(point_set)==4:
        #if 1:
            src = np.array([point_set['corner1'],point_set['corner4'],point_set['corner2'],point_set['corner3']], dtype=np.float32)
            #src = np.array([[130,41],[473,38],[45,329],[592,323]], dtype=np.float32)
            
            m = cv2.getPerspectiveTransform(src, dst)


            #  使用m矩阵变换，结果为图像大小，使用白色填充
            res = cv2.warpPerspective(
            color_image,
            m,
            (549, 549),
            borderValue=(255, 255, 255, 255)
            )
            
            coordinate=[]
            for i in range(len(r_detect['class_ids'])):
                if is_disturbance
                x=int((r_detect['rois'][i][1]+r_detect['rois'][i][3])/2)
                contour=mask2contour(r_detect['masks'][:,:,i])
                y_max=0
                x_max=0
                #世界坐标
                y_sum=0
                x_sum=0
                #print(contour)
                
                for each in contour:
                    if each[0]>y_max:
                        y_max=each[0]
                #print(contour)
                #print("y is")
                #print(y_max)
                #y=int(r_detect['rois'][i][2])
                y=y_max
                x1,y1=coordinate_transform(x,y,r_detect['class_ids'][i],m)
                print(detect_class_names[r_detect['class_ids'][i]])
                print((x1/10,y1/10))
                
                coordinate.append([detect_class_names[r_detect['class_ids'][i]],r_detect['scores'][i],[x1/10,y1/10]])

            end=datetime.now()
            print((end-start).seconds)
            ##透视变换结束
            cv2.imshow("masks",res)
        if point_set:
            for key in point_set:
                cv2.circle(color_image,(int(point_set[key][0]),int(point_set[key][1])),2,(0,255,0),2)   
        cv2.imshow('frame',color_image)
        cv2.waitKey(10)  
        visualize.display_instances(image, r_detect['rois'], r_detect['masks'], r_detect['class_ids'],
                                    detect_class_names, r_detect['scores'])
    """
    file_names = next(os.walk(IMAGE_DIR))[2]

    image = skimage.io.imread("./test_set/288.jpg")#图片类型的区别##395可以补角,258找错底面的两个点，
    #cv2.imshow("BGR",frame)
    frame=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #cv2.imshow("BGR",frame)
    frame1=image.copy()
    # Run detection

    detect_results = model_detect.detect([image], verbose=1)####verbode用不用改

    r_detect = detect_results[0]

    #masks2contours_smooth(r_detect['masks'],r_detect['class_ids'],frame)


    point_set=corner_points(frame1,model_corner)
    #cv2.imshow("frame1",frame1)
    #cv2.imshow("RGB",image)
    ##透视变换##################

    dst = np.array([[0, 0], [549, 0],[0, 549],[549, 549] ], dtype=np.float32)
    if len(point_set)==4:
    #if 1:
        src = np.array([point_set["corner1"],point_set['corner4'],point_set['corner2'],point_set['corner3']], dtype=np.float32)
        #src = np.array([[130,41],[473,38],[45,329],[592,323]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)


        #  使用m矩阵变换，结果为图像大小，使用白色填充
        res = cv2.warpPerspective(
        frame,
        m,
        (549, 549),
        borderValue=(255, 255, 255, 255)
        )
        #print(type(m))
        #print(m)
        #x_true,y_true=coordinate_transform(352,192,m)#此处还需要修改，最好输入点集，直接返回点集
        #print((x_true,y_true))
        #cv2.circle(res,(x_true,y_true),2,(0,255,0),2)
        coordinate=[]
        for i in range(len(r_detect['class_ids'])):
                x=int((r_detect['rois'][i][1]+r_detect['rois'][i][3])/2)
                contour=mask2contour(r_detect['masks'][:,:,i])
                y_max=0
                x_max=0
                angle=0
                
                for each in contour:
                    if each[0]>y_max:
                        y_max=each[0]
                if detect_class_names[r_detect['class_ids'][i]] in ['toothpaste','toilet_soap','snow_pear','tea_pi','tang']:
                    #roi_mat=np.ones((480,640,3),dtype=np.uint8)*255
                    frame_copy=np.array(frame)
                    roi_mat=frame_copy[r_detect['rois'][i][0]:r_detect['rois'][i][2],r_detect['rois'][i][1]:r_detect['rois'][i][3],:].copy()
                    print('1',frame_copy[r_detect['rois'][i][0]:r_detect['rois'][i][2],r_detect['rois'][i][1]:r_detect['rois'][i][3],:])
                    print('2',roi_mat)
                    edges = cv2.Canny(cv2.cvtColor(roi_mat,cv2.COLOR_BGR2GRAY),100, 255, apertureSize=3)
                    LINES=cv2.HoughLinesP(edges, 1.0, np.pi / 180, 30,1, 5,15)
                    #print("线段数",len(LINES))
                    line_lenth=0
                    angle_x1=0
                    angle_x2=0
                    angle_y1=0
                    angle_y2=0
                    if LINES is not None:
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
                    angle=math.degrees(angle)
                y=y_max
                x1,y1=coordinate_transform(x,y,r_detect['class_ids'][i],m)
                print(detect_class_names[r_detect['class_ids'][i]])
                print((x1/10,y1/10),   angle)
                coordinate.append([detect_class_names[r_detect['class_ids'][i]],r_detect['scores'][i],[x1,y1]])

        end=datetime.now()
        print((end-start).seconds)
        ##透视变换结束
        cv2.imshow("result",res)
        
    visualize.display_instances(image, r_detect['rois'], r_detect['masks'], r_detect['class_ids'],
                                detect_class_names, r_detect['scores'])
    