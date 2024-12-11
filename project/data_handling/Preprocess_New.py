# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:31:03 2020

@author: penda
"""

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import cv2
import numpy as np
import re
import os.path as osp
import os
import pandas as pd
import Config
from keras.utils import np_utils


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_image(depth_dir,H,W):

    """Load and preprocess image."""
    
    depth_image_temp = []
    depth_image_temp_index = []    
    depth_image_path = [f for f in files_in_subdirs(depth_dir, '.jpg')]
    print('Load Image Num Total:',len(depth_image_path))
    
    #x = image.load_img(path, target_size=(H, W))
    
    for i in range(len(depth_image_path)):
        path = depth_image_path[i]
        #x = image.load_img(path, target_size=(H, W))
        
        x = cv2.imread(path)
        x = cv2.resize(x,(H,W))
        x = image.img_to_array(x)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        index_id = depth_image_path[i].split('/')[-1].split('.jpg')[0]
        
        depth_image_temp_index.append(index_id)
        depth_image_temp.append(x)
        
    return depth_image_temp, depth_image_temp_index


                
def Process_Labels(depth_dir_train,label_dir_train):
    label_list_train = pd.read_csv(label_dir_train, sep=' ', header=None)
    label_list_train.columns = ['name','1','2','3','4','label']
    #label_list_train = pd.read_csv(label_dir_train,  header=None)
    #label_list_train.columns = ['name','label']    
    
    print(label_dir_train)
    print(label_list_train)
    label_ground_train = []
    depth_image_path = [f for f in files_in_subdirs(depth_dir_train, '.jpg')]
    print('Image Number:',len(depth_image_path))

    for i in range(len(label_list_train)):
        if i>=len(depth_image_path):
            break
        Index_Name = depth_image_path[i].split('/')[-1]
        Name = Index_Name.split('.')[0]
        #print(Name)
        label_temp = label_list_train[label_list_train['name'] == Name]['label'].values
        label_ground_train.append(label_temp[0])

    
    label_ground_train = np.asarray(label_ground_train)
    label_ground_train = np.reshape(label_ground_train,[len(depth_image_path),1])
    
    return label_ground_train 

##############################################################################    



All_Path = '../Database/' + Config.Folder_Name + '/'
depth_dir_train = All_Path + Config.Type + '/'    

   
Regress_dir_train = All_Path +  'All_data.csv'
Context_dir_train = All_Path +  'Context_data.csv'
State_dir_train = All_Path +  'State_data.csv'

'''
if Config.Context_Flag == True:
    label_ground_train = Process_Labels(depth_dir_train,Context_dir_train)
if Config.State_Flag == True:
    label_ground_train = Process_Labels(depth_dir_train,State_dir_train)
if Config.Regress_Flag == True:
    label_ground_train = Process_Labels(depth_dir_train,label_dir_train)
'''

State_ground_train = Process_Labels(depth_dir_train,State_dir_train)
Context_ground_train = Process_Labels(depth_dir_train,Context_dir_train)
Regress_ground_train = Process_Labels(depth_dir_train,Regress_dir_train)
print(min(Regress_ground_train),max(Regress_ground_train))

def Regular_Regress(Regress_ground_train):
    Y_train = Regress_ground_train
    
    y_train = Y_train.reshape(-1,1)
    
    if Config.Robust_Flag == True:
        rbs = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0), copy=True)
    else:
        rbs = MinMaxScaler()
    
    y_train_s = rbs.fit_transform(y_train)
    return y_train_s

Regress_ground_train = Regular_Regress(Regress_ground_train)


#NPZ_Name = Config.Folder_Name + '_' + Config.Type + '_RegressGood' + '_database.npz'

#y_test_s = rbs.transform(np.log(y_test))
    
##############################################################################        
   
# Process Images  
NPZ_Name = Config.Folder_Name + '_' + Config.Type +  '_' + str(Config.img_rows) +  '_overall_database.npz'

if __name__ == '__main__':

    if Config.Process_Image_Flag == True:
        depth_image_temp_train, depth_image_temp_index_train = load_image(depth_dir_train, Config.img_rows, Config.img_cols)
        print('Load Image Finish')    

        X_train = np.asarray(depth_image_temp_train)
        X_train = X_train.astype('float32')
        y_train = State_ground_train
        np.savez(NPZ_Name,X_train=X_train,Y_train_Context=Context_ground_train,Y_train_State=State_ground_train,Y_train_Regress=Regress_ground_train)




    else:
        Used_NPZ_Name = Config.Folder_Name + '_' + Config.Type +  '_' + str(Config.img_rows) + '_' + Config.Model_Type + '_database.npz'
        Database_Used = np.load(Used_NPZ_Name)
        X_train = Database_Used['X_train']
        print('Not process Image')    
        
    
    
    #np.savetxt('All_labels.csv',y_train)
