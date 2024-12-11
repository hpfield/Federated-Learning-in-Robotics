# -*- coding: utf-8 -*-

Folder_Name = 'Videos_Database_20_Robot'

Type = 'WebCam'
##############################################################################

Process_Image_Flag = True
Robust_Flag = True

img_rows, img_cols = 256,256
img_rows, img_cols = 50,50

img_channels = 3
batch_size = 20
nb_epoch = 100

Cup_Type = 'Big'
if Cup_Type == 'Medium':
    nb_classes = 9
if Cup_Type == 'Big':
    nb_classes = 10
if Cup_Type == 'Small':
    nb_classes = 7   
        



