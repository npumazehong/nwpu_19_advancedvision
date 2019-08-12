import cv2# as cv
import math
import numpy as np
#def checkEllipseShape(image,contour,ellipse,ratio):
#    center=ellipse[0]




"""


def find_MaxRect_and_image(image):
    frame=np.zeros((image.shape[0],image.shape[1],1),dtype=np.uint8)
    b,g,r=cv.split(image)
    for i in range(r.shape[0]): #height
        for j in range(r.shape[1]): #weight
            if 135<b[i,j]<180 and 125<g[i,j]<200 and 125<r[i,j]<200:
                frame[i,j]=255
    frame=cv.dilate(frame,np.ones((5,5),np.uint8), iterations=1)
    _,contours,hier=cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    Max_S=0
    book=True
    minEllipse=[]
    point_set_MaxRect=[]
    for i in range(len(contours)):
        s=cv.contourArea(contours[i])
        if len(contours[i])<10:
            continue
        if s<10:
            continue
        minEllipse=cv.fitEllipse(np.array(contours[i]))
        #if checkEllipseShape(image,contours[i],minEllipse):
        #    continue
        #cv.ellipse(image,minEllipse,(0,0,255),5)
        minRect=cv.minAreaRect(np.array(contours[i]))
        box=cv.boxPoints(minRect)
        
        s=minRect[1][0]*minRect[1][1]
        k1=math.fabs((box[0][1]-box[1][1])/(box[0][0]-box[1][0]))
        k2=math.fabs((box[2][1]-box[1][1])/(box[2][0]-box[1][0]))
        k_tmp=min(k1,k2)
        if s>Max_S:
            Max_S=s
            point_set_MaxRect=box
            book=False
            k=k_tmp
    if book:
        return [],[],0,90
    #for i in range(len(box)):
    #    cv.line(image,box[i],box[(i+1)%4]
    return point_set_MaxRect,image,Max_S,k
def find_ellipse(pipeline):
    Max_s=0
    final_image=[]
    final_box=[]
    for i in range(10):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            if i!=0:
                i-=1
            continue
        color_image = np.asanyarray(color_frame.get_data())
        box,image,s,k=find_MaxRect_and_image(color_image)
        if s>Max_s and k<0.3:
            Max_s=s
            final_image=image
            final_box=box
        
        if box!=[] and image!=[]:
            #cv.imshow('process',image)
            cv.waitKey(200)
        else:
            continue
    return final_image,final_box
        

"""

def find_ellipse(pipeline):   
    frame=None
    frame_tmp=None
    (x2,y2,w2,h2)=(0,0,0,0)
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(300):
        max_area_in_all_image=0
        # 每15帧取一次
        if i%15==0:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Convert images to numpy arrays
            frame_lwpCV = np.asanyarray(color_frame.get_data())

            gray_lwpCV = cv2.cvtColor(frame_lwpCV,cv2.COLOR_BGR2GRAY)
            gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (27, 27), 0)
            diff = cv2.threshold(gray_lwpCV, 160, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
            diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
            
            # 显示矩形框
            image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
            max_area_in_per_image=1000
            (x1,y1,w1,h1)=(0,0,0,0)
            for c in contours:
                if cv2.contourArea(c) <max_area_in_per_image: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                    continue
                else:
                    (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
                    if w>h:
                        frame_lwpCV_copy=frame_lwpCV.copy()
                        cv2.rectangle(frame_lwpCV_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)
                        max_area_in_per_image=cv2.contourArea(c) 
                        (x1,y1,w1,h1)=(x,y,w,h)
                        frame_tmp=frame_lwpCV
                    
                        # 显示某一帧中的最大矩形，只用来调试，别的时候要直接注释，不然原图会多框
                        cv2.imshow('contours', frame_lwpCV_copy)
                        cv2.imshow('dis', diff)
            if max_area_in_per_image>max_area_in_all_image:
                max_area_in_all_image=max_area_in_per_image
                frame=frame_tmp
                (x2,y2,w2,h2)=(x1,y1,w1+30,h1+15)
    return frame,[[x2,y2],[x2+w2,y2],[x2+w2,y2+h2],[x2,y2+h2]]
