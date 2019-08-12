import cv2
import math
import pyrealsense2 as rs
def point_in_mask(point,rois):
    for roi in rois:
        if roi[1]<point[0]<roi[3] and roi[0]<point[1]<roi[1]:
            return True
    return False
def point_transform(point,depth_frame,rois,depth_intrin,depth_to_color_extrin):
    left_point=[0,0]
    for i in range(5,100):
        left_point=[point[0]+i,point[1]-i]
        if not point_in_mask(left_point,rois) and depth_frame.get_distance(int(left_point[0]),int(left_point[1]))!=0:
            break
    if left_point!=[0,0]:
        depth_point1=rs.rs2_deproject_pixel_to_point(depth_intrin, left_point, depth_frame.get_distance(int(left_point[0]),int(left_point[1])))
        color_point1=rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point1) 
    else:
        color_point1=[0,0,0]
        print('point_transform fail')
    return color_point1
def find_plane(points,depth_frame,rois,depth_intrin,depth_to_color_extrin):
    left_point=[0,0]
    right_point=[0,0]
    third_point=[0,0]
    #left_point
    for i in range(5,100):
        left_point=[points['corner2'][0]+i,points['corner2'][1]-i]
        if not point_in_mask(left_point,rois) and depth_frame.get_distance(int(left_point[0]),int(left_point[1]))!=0:
            break
    if left_point!=[0,0]:
        depth_point1=rs.rs2_deproject_pixel_to_point(depth_intrin, left_point, depth_frame.get_distance(int(left_point[0]),int(left_point[1])))
        color_point1=rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point1) 
    else:
        color_point1=[0,0,0]
        print('color_point1_transform fail')
    #right_point
    for i in range(5,100):
        right_point=[points['corner3'][0]-i,points['corner3'][1]-i]
        if not point_in_mask(right_point,rois) and depth_frame.get_distance(int(right_point[0]),int(right_point[1]))!=0:
            break
    if right_point!=[0,0]:
        depth_point2=rs.rs2_deproject_pixel_to_point(depth_intrin, right_point, depth_frame.get_distance(int(right_point[0]),int(right_point[1])))
        color_point2=rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point2) 
    else:
        color_point2=[0,0,0]
        print('color_point2_transform fail')
    #third_point
    if left_point!=[0,0] and right_point!=[0,0]:
        for i in range(20,200):
            third_point=[int((right_point[0]+left_point[0])/2),int((right_point[1]+left_point[1])/2)-i]
            if not point_in_mask(third_point,rois) and depth_frame.get_distance(int(third_point[0]),int(third_point[1]))!=0:
                break 
        if third_point!=[0,0]:
            depth_point3=rs.rs2_deproject_pixel_to_point(depth_intrin, third_point, depth_frame.get_distance(int(third_point[0]),int(third_point[1])))
            color_point3=rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point3)
        else:
            color_point3=[0,0,0]
            print('color_point3_transform fail')
    else:
        print('in depth_image,we can not find left or right point')
    if left_point!=[0,0] and right_point!=[0,0] and third_point!=[0,0]:
        L1=[color_point2[0]-color_point1[0],color_point2[1]-color_point1[1],color_point2[2]-color_point1[2]]
        L2=[color_point3[0]-color_point1[0],color_point3[1]-color_point1[1],color_point3[2]-color_point1[2]]
        A=L1[1]*L2[2]-L1[2]*L2[1]
        B=L1[2]*L2[0]-L1[0]*L2[2]
        C=L1[0]*L2[1]-L1[1]*L2[0]
        D=(A*color_point1[0]+B*color_point1[1]+C*color_point1[2])*(-1)
        return [A,B,C,D]
    else:
        return [0,0,0,0]

def is_disturbance(plane,contour,depth_frame,depth_intrin,depth_to_color_extrin):
    distance_sum=0
    i=0
    for each in contour:
        if depth_frame.get_distance(int(each[1]),int(each[0]))!=0:
            depth_point=rs.rs2_deproject_pixel_to_point(depth_intrin, [each[1],each[0]], depth_frame.get_distance(int(each[1]),int(each[0])))
            color_point=rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
            distance=abs(color_point[0]*plane[0]+color_point[1]*plane[1]+color_point[2]*plane[2]+plane[3])/math.sqrt(plane[0]*plane[0]+plane[1]*plane[1]+plane[2]*plane[2])
            distance_sum+=distance
            i+=1
    if i!=0:
        average_distance=distance_sum/i
        print(average_distance)
    else:
        print("the points' distance to camera are all 0 ")
        average_distance=1
    if average_distance<0.004:
        return True
    else:
        return False
