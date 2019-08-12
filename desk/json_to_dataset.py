import os
path = 'D:\\QQPCmgr\\Desktop\\先进视觉\\物体检测数据集\\0718json'  # path为json文件存放的路径
json_file  = os.listdir(path)

for file in json_file:
    #print(file) 
    #print("python labelme_json_to_dataset %s"%(path + '/' + file))
    os.system("labelme_json_to_dataset %s"%(path + '/' + file))
