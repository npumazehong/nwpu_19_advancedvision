# -*- coding: utf-8 -*-
import os
import pyrealsense2 as rs
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
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import cv2 as cv
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize2
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"detect_model/mask_rcnn_shapes_0010.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")



class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        #self.cap = cv2.VideoCapture()  # 视频流
        #self.CAM_NUM = 1  # 为0时表示视频流来自笔记本内置摄像头
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 10)
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.TIME = 0

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.textEdit = QTextEdit()
        self.textEdit.setFixedSize(400, 800)

        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(1282, 962)  # 给显示视频的Label设置大小为641x481

        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        self.__layout_main.addWidget(self.textEdit)
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
    '''槽函数之一'''
    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            #self.cap.open(1)
            flag=self.pipeline.start(self.config)
            print(flag)
            #flag = self.cap.open(1)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(10)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('结束识别')
        else:
            self.timer_camera.stop()  # 关闭定时器
            #self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('开始识别')

    def show_camera(self):
        tick = time.time()

        #flag, self.image = self.cap.read()  # 从视频流中读取
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.image = np.asanyarray(color_frame.get_data())
        results = model.detect([self.image], verbose=1)
        r = results[0]
        self.image = visualize2.video_display1(self.image, r['rois'], r['class_ids'], class_names, r['scores'])
        print(self.image)
        show = cv2.resize(self.image, (1280, 960))  # 把读到的帧的大小重新设置为 1280x960
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        tock = time.time()
        self.TIME = self.TIME + (tick - tock)
        self.show_result()

    def show_result(self):
        fi = open('result.txt', 'r')
        txt = fi.readlines()
        txt = '\n'.join(txt)
        self.textEdit.setPlainText(txt)


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
    TRAIN_ROIS_PER_IMAGE =200
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200
 
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

##########################################################
#  此处开启界面
###########################################################

while 1:
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    '''
    ret, image = cap.read()
    a = datetime.now()
    # Run detection
    '''

    '''
    b=datetime.now()
    # Visualize results
    print("shijian",(b-a).seconds)
    '''
    ui = Ui_MainWindow()
    #visualize.video_display(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
