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
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0013.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
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
    NUM_CLASSES = 1 + 27  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
a1=datetime.now() 
config = InferenceConfig()
b1=datetime.now()
c1=b1-a1
print("the time of inferenceConfig is %d"%(c1.seconds))
#model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
a2=datetime.now()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
b2=datetime.now()
c2=b2-a2
print("the time of MaskRCNN is %d"%(c2.seconds)) 
# Load weights trained on MS-COCO
a3=datetime.now()
model.load_weights(COCO_MODEL_PATH, by_name=True)
b3=datetime.now()
c3=b3-a3
print("the time of loading model is %d"%(c3.seconds)) 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'toilet_soap', 'liquid_soap','toothpaste', 'toilet_water','duck', 'porridge','water','old_godmother','tang','gum','soda','copico','melon_seeds','red_bull','AD_milk','juice',
                'Wanglaoji','Jiaduobao', 'green_tea', 'snow_pear', 'coconut', 'black_tea','coca_cola', 'sprite', 'fenta', 'cookie', 'noodles']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread("./242.jpg")
frame=cv2.imread("./242.jpg")
a=datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b=datetime.now()
# Visualize results
print("shijian",(b-a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
#visualize.video_display(frame, r['rois'],r['class_ids'],class_names, r['scores'])
