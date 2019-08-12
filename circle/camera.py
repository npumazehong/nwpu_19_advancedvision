import pyrealsense2 as rs
import numpy as np
import cv2
import time
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
pipeline.start(config)
frames = pipeline.wait_for_frames()
time.sleep(1)
i=2
while True:
    path='./picture718/'
    filename='%d.jpg'%(i)
    key=cv2.waitKey(100)
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('frame',color_image)
    if key==ord('p'):
        i+=1
        cv2.imwrite(path+filename,color_image)
        cv2.imshow('photo',color_image)
        cv2.waitKey(500)
    elif key==ord('q'):

        break
    else:
        continue
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    
    #cv2.applyColorMap伪色彩，个人理解为加个滤镜
    # 其中cv2.convertScaleAbs可以将16位图转换为cv可以识别的8位图，并且可以利用alpha调整图片强度
    # cv2.COLORMAP_JET为指定的方法，值为 3 ，一般用来生成热力图
    # Stack both images horiz
    #水平拼接，并排显示两个图片
    # Show images

