import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
pipe_profile=pipeline.start(config)
align_to_color=rs.align(rs.stream.color)

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
background_flag=0
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)) #此处调整核，从而控制边框与椭圆的距离
kernel = np.ones((5, 5), np.uint8)
background = None
while True:
    
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    #frames = align_to_color.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    frame_lwpCV = np.asanyarray(color_frame.get_data())

    gray_lwpCV = cv2.cvtColor(frame_lwpCV,cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (27, 27), 0)
    
    # 将第一帧设置为整个输入的背景
    if background_flag==0:
        background =gray_lwpCV
        background_flag=1
        continue
    # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    #diff = cv2.absdiff(background, gray_lwpCV)
    #cv2.imshow('diff_pre',diff)
    diff = cv2.threshold(gray_lwpCV, 160, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
    
    # 显示矩形框
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) < 10000: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
        cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('dis', diff)
    
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
    background_flag=0