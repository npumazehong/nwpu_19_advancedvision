import cv2 as cv 
import numpy as np
import math
 
def line_detect(img):
        img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
        #img=cv.GaussianBlur(img,(11,11), 0)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        #img1=cv.imread('206.jpg',0)
        #img=cv.GaussianBlur(img,(11,11), 0)
        #ret, binary = cv.threshold(img,yuzhi, 255, cv.THRESH_BINARY_INV )
        #cv.imshow('Result', binary)

        edges = cv.Canny(gray,100, 255, apertureSize=3)
        
        #cv.imshow('Result', edges)
        minLineLength = 70
        maxLineGap = 50
        lines=cv.HoughLinesP(edges, 1.0, np.pi / 180, 30,1, minLineLength,maxLineGap)
        #print(len(lines))
        #print(lines)
        """
        for line in lines:
                print(line)
                for another in lines:
                        if line.any!=another.any:
                                if distance(line[0][0],line[0][1],line[0][2],line[0][3],another[0][0],another[0][1])<=2:
                                        lines.
        """                     
        #print(len(lines))
        if lines is None:
                pass
        else:
                for line in lines: 
                        for x1,y1,x2,y2 in line:                       
                                cv.line(img, (x1, y1), (x2, y2), (0, 255,0 ), 3)
        #cv.imshow("lines", img)
        #cv.waitKey(0)

        return lines
        #cv.destroyAllWindows()
def nothing(x):
        pass   
        
def distance(x1,y1,x2,y2,x3,y3):
        a=math.sqrt((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2))
        b=math.sqrt((y3-y2)*(y3-y2)+(x3-x2)*(x3-x2))
        c=math.sqrt((y1-y3)*(y1-y3)+(x1-x3)*(x1-x3))
        p=float(a+b+c)/2
        S=math.sqrt(p*(p-a)*(p-b)*(p-c))
        distance=2*S/a
        return distance
    