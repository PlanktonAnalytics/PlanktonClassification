# -*- coding: utf-8 -*-
#"""
# Created on Mon Feb 07 2022
# VERSION 0.9
#
#@author: Phil Culverhouse, Plankton Analytics ltd.
# developed from code supplie by James Scott & Rob Blackwell at UEA & CEFAS
# Classifies plankton images from Pi-1
# using the Torch model developed by Rob Blackwell et al. at the Turing DSG Dec 2021
# assumes Plankton Imager 10-minute folder structure
# processes day folders or one 10-min. folder, saves CSV results in each 10-min folder

# ************* NOTE: the model location fixed in program SEE LINES 49-61 to set models ###############

# call as
# python3 DNN_preSort.py --fpath=PATH --sort

# [Fname, Class, p0, p1, p2, Area, Major, Minor]

# Option to moves images into sorted folders (done by PI_Label() in c++)
# call as
#
# python DNN_PreSort.py -fpath abd_path_name -sort
#
## Uses OpenVino https://docs.openvino.ai/latest/notebooks/401-object-detection-with-output.html
## adapted to just create a CSV file full of labels for a Day processed data set
## uses ONNX FP16 model created using TorchExport.py

import os
import sys
import time
import shutil
import getopt, sys # for cmd line parse

import csv
import numpy as np

from PIL import Image # use for TIF read, as opencv is not always built with TIF library
#from PIL.TiffTags import TAGS # for metadata in the future
import exifread ##pip install ExifRead

import cv2
import xml.etree.ElementTree as ET

#two alteratives for run time
from openvino.inference_engine import IECore


########## Manually set to location of folders for non-main() operation #####################
SEP='/'
DIR="/Users/culverhouse/Code/"
if sys.platform == 'win32':
  SEP='\\'
  DIR='C:'

INrootDIR=DIR + SEP + "openvino_optimised" + SEP + "data" + SEP + "rawdata" + SEP + ""

######################### MODEL DETAILS #####################
modelname="CopNonDetritus_42317_FP32" # DSG DEC 2021 model

#labelstr="CopNonDetritus_42317_Labels.xml"
labelstr="CopNonDetritus_42317_OpenCV_Labels.xml" ## PFC 2 ways of formatting XML, the OpenCV one is the easiest to port Python3/c++"

########## CHANGE TO UPDATE MODEL LOCATION #################
modelDIR= SEP + "openvino_optimised" + SEP + "models" + SEP + "Model-CEFAS-Pi-CopNonDetritus_42317" + SEP
############################################################


##### add another below this to set your chosen folder to process, as you can see I have two,
##### comment out if not required
#target_directory    = "2020_10_21-short"
#target_directory    = "2020_10_21" ### 6 minutes with ONNX FP16 model
#target_directory    = "CEnd-May-2021-Good_images"

#InputDIR= INrootDIR + target_directory + SEP ### now over written by main() below from arg list

# ################ Use FP16 model ####################
# note: could set this as cmd arg
converted_model_path = "" + DIR+ modelDIR + modelname + ".xml"
print(f'Model:{modelname}')
#################### load model ######################
# initialize inference engine
ie_core = IECore()
# read the network and corresponding weights from file
net = ie_core.read_network(model=converted_model_path)

# load the model on the CPU (you can choose manually CPU, GPU, MYRIAD etc.)
# or let the engine choose best available device (AUTO)
exec_net = ie_core.load_network(network=net, device_name="CPU")

# get input and output names of nodes
input_key = list(exec_net.input_info)[0]
output_key = list(exec_net.outputs.keys())[0]
#print (f'IN: {input_key}, OUT: {output_key}')

# get input size
height, width = exec_net.input_info[input_key].tensor_desc.dims[2:]
#print (f'network input DIM: {height},{width}')

################## CLASSES used to train DNN model ########################
## note keep the orer the same as for the training regime ##

#classes = [    "Copepoda", "Detritus", "NONCopepoda"]
## now read from the Labels.xml in the same location as the model
def ReadLabels(path):
    
# PFC USE EITHER XML FORMAT
#    path="/Users/culverhouse/Code/openvino_optimised/models/DSG_DNN_PreSort_Model/CopNonDetritus_42317_Labels.xml"
#    tree = ET.parse(path)
#    root = tree.getroot()
#    classes=[]
#    for child in root:
#        name = child.get('Class')
#        classes.append(name)
#    #print(f'ReadLabels:{Classes}')
#    return(classes)
    
# PFC OR USE OPENCV FILESTORAGE FORMAT
# use opencv filestorage class as its compatible with c++ & python
    #path="/Users/culverhouse/Code/openvino_optimised/models/Model-CEFAS-Pi-CopNonDetritus_42317/CopNonDetritus_42317_OpenCV_Labels.xml"
    if os.path.exists(path):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        fn = fs.getNode("Labels")
        Classes=[]
        for i in range(fn.size()):
            Classes.append(fn.at(i).string())
        #print(f'ReadLabels:{Classes}')
        return(Classes)
    else:
        print(f'Error: no Labels file:{path}')
        exit (-2)
            
################# Softmax() ########################
# PFC from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
# PFC need to figure how to do this insie the ONNX model
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
    
##################### CLASSIFY #####################
def classify(imageC,classes):
 
     # resize image and change dims to fit neural network input
     imageC=np.asarray(imageC) # make an array that resize can use
     imageResized = cv2.resize(imageC, dsize=(width, height), interpolation=cv2.INTER_AREA)
     input_img=imageResized/255 #convert to float

     
     # create batch of images (size = 1)
     input_img = np.expand_dims(input_img.transpose(2, 0, 1), 0)

     # get results noramlly in openvino
     results = exec_net.infer(inputs={input_key: input_img})[output_key]
     result_index = np.argmax(results)

     label = classes[np.argmax(results)]
     #print (f'label:{label}')
     results=results.tolist()
     #print(f'{results[0]} -> {classes[result_index]}')
    
     probs=softmax(results[0])
     return (label,probs)
     
############## measure image for area and fit ellipse ##################

def GetPropsFromImage(imageC, Fname):
    imgGray = imageC.convert('L')
    img_UINT8 = np.uint8(np.abs(imgGray))
    img_UINT8 = 255-img_UINT8
    (T,Ithr)=cv2.threshold(img_UINT8, 0, 255, cv2.THRESH_OTSU)
    
    
    # find largest contour
    cnts  = cv2.findContours(Ithr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # extract contours regardless of version!
    biggest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    
    if (area >30):
        # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
        ellipse = cv2.fitEllipse(biggest)
        (xc,yc),(d1,d2),angle = ellipse
        #print(xc,yc,d1,d1,angle)
    else: # if too small just zero the parameters, coul in future move the image to the _smallitems folder??
        area=0
        d1=0
        d2=0
        #cv2.imshow('THR', Ithr)
        #cv2.waitKey(0)

    # draw ellipse
    #drawing = np.array(imageC)
    #cv2.ellipse(drawing, ellipse, (0, 255, 0), 1)
    #cv2.imshow(Fname, drawing)
    #cv2.waitKey(0)
    
    return([area, d1, d2])

# ## my old code PyTorch code ##########################
     # # Convert Image to tensor and resize it
     #
     # t = functional.resize(t, (256, 256))
     # model expects a batch of images so lets convert this image tensor to batch of 1 image
     # t = t.unsqueeze(dim=0)
     # #print((type(t),t.shape))
     # t = t.to(device)
     
     # with torch.set_grad_enabled(False):
         # outputs = model(t)
         # # select top 1 from outputs
         # _, preds = torch.max(outputs, 1)
         
         # #PFC from https://discuss.pytorch.org/t/how-to-extract-probabilities/2720/14
         # sm = torch.nn.Softmax()
         # probabilities = sm(outputs[0])
         # #print(f'{filename} : {probabilities} : {labels[preds[0]]}') # get probabilities!!
     # return(labels[preds[0]],probabilities)

# save P-values in CSV file in the rawdata folder, so PI_ScaleBar() c++ can convert to

# have one CSV file for each 10-minute folder
def Create_CSV(folder):
    os.makedirs(folder, exist_ok=True)
    csv_fileSTR=os.path.join(folder, "DNN_Results.csv")
    #print(csv_fileSTR)
    CSVfile = open(csv_fileSTR, 'w')
    CSVwriter = csv.writer(CSVfile)
    CSVwriter.writerow([modelname]) # audit trail save the model
    
    headers=['Fname' , 'Class','Copepoda_(p)','Detritus_(p)','NONCopepoda_(p)', 'Area', 'Major', 'Minor', 'DateTime','GPSLatitudeRef','GPSLatitude','GPSLongitudeRef','GPSLongitude']
    CSVwriter.writerow(headers),
    return CSVfile, CSVwriter #so we can close it
    
#### check if folder exists, list it
def listdirs(folder):
    if os.path.exists(folder):
         return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    else:
         return []
         
def usage():
    print('-fp, --fpath xyz : set location of the sample, either ay or 10-min folder')
    print('-s, --sort : sort classified images into Class folders')
    print('Creates ./Desc/DNN_Results.csv in each ten-minute folder')
    print('-h, --help : this message')
    
###### MAIN runs through the folder hierarchy fining 10-minute folders with "/Images" subfolder
# assumes not much else in the folder structure
# excepting std Plankton Imager files (and ZIP files for easy teset image storing
# DNN_PreSort -fpath -sort
def main():

    sort=False # set true by cmd
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "fpath=", "sort"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    output = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-fp", "--fpath"):
            InputDIR = a
        elif o in ("-s", "--sort"):
            sort = True
        else:
            assert False, "unhandled option"
  ##      print (f'Args:: {InputDIR}, {sort}')

    
    # then classifying images
    # measure processing time
    start_time = time.time()

    classes=ReadLabels(path = DIR + modelDIR + labelstr)
    print(f'Classes: {classes}')

    Icount=0
    ten_min_folders=listdirs(InputDIR)
    #print(f'ten_min_folders:{ten_min_folders}')
    
    # set GPS vars
    DateTime=""
    GPSLatitudeRef=""
    GPSLatitude=""
    GPSLongitudeRef=""
    GPSLongitude=""
    
    for folder_item in ten_min_folders:
    
        if SEP + 'Desc' in folder_item:
            print(f'Desc-Images removed {folder_item}')
            continue ## PFC there may be an Images folder in the Desc folder, skip it
    
        images_folder=InputDIR + SEP  + folder_item + SEP + "Images" + SEP
        if not (os.path.exists(images_folder)):
            continue
            
        Desc_folder=InputDIR + SEP  + folder_item + SEP + "Desc" + SEP
        #print(f' csv results: {Desc_folder}')

        CSV_fd, CSVwriter = Create_CSV(Desc_folder) # create target CSV results file
        shutil.copy2(DIR + modelDIR+labelstr , Desc_folder + labelstr) ## copy Labels.xml to target
        
        files = os.listdir(images_folder)
        
        ####### create class folders in Images folder, so they can be scaled by a quick C++ program and
        ####### moved to ScaledImages folder.
        if (sort):
            # mkdir for each class
            for class_item in classes:
                class_folder =images_folder + class_item
                os.makedirs(class_folder, exist_ok=True)
                #print(f'created {class_folder}')
                
            os.makedirs(images_folder + "Unknown", exist_ok=True) ## always have an catchall bin
                    
        
        ### run through each 10-min folder, each image, classify and then sort (if required)
        ####for root, dirs, files in os.walk(images_folder):
        for file in files:

            if '.DS_Store' in file: ## for OSX !! PFC
                continue
            if 'DNN_results.csv' in file: ## skip for moment, reorganise format later
                continue
            if 'Background.tif' in file: ## skip for moment, reorganise format later
                continue
            if 'Images.zip' in file: ## skip for moment, reorganise format later
                continue
            if 'RawImages.zip' in file: ## skip for moment, reorganise format later
                continue
                
            Fpath = images_folder + file # full path name for an image

            #print(f'Images_folder:{images_folder}, FILE: {file}')
            # filter for images in directory, if not  #### IMPORTANT
            #if (SEP + 'Images') in Fpath:
            imageC =  Image.open(Fpath)
            #imageC = np.array(imageC)
## do nothing with the tagfs at present, just know how to read them.
            with open(Fpath, 'rb') as imgTAGS:
                tags = exifread.process_file(imgTAGS)
            #print(f'Tags: {tags}')
            for tag in tags.keys():
                if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                    #print ("Key: %s, value %s" % {tag, tags[tag])
                    #print(f'Key: {tag}, = {tags[tag]}')
                    if tag in 'Image DateTime':
                        DateTime=tags[tag]
                        #print(f'{DateTime}')
                        continue
                    if tag in 'GPS GPSLatitude':
                        GPSLatitude=tags[tag]
                        #print(f'{GPSLatitude}')
                        continue
                    if tag in 'GPS GPSLatitudeRef':
                        GPSLatitudeRef=tags[tag]
                        #print(f'{GPSLatitudeRef}')
                        continue
                    if tag in 'GPS GPSLongitude':
                        GPSLongitude=tags[tag]
                        #print(f'{GPSLongitude}')
                        continue
                    if tag in 'GPS GPSLongitudeRef':
                        GPSLongitudeRef=tags[tag]
                        #print(f'{GPSLongitudeRef}')
                        continue
            #print(f'{file}:{DateTime},{GPSLatitudeRef},{GPSLatitude},{GPSLongitudeRef},{GPSLongitude}')
            # cannot use opencv as the Intel opencv binary libraries do not have TIF lib!!
            #imageC =  cv2.imread(Fpath)
            #cv2.imshow("test",imageC8)
            #cv2.waitKey(0)
                                    

# TODO PFC metadata
            #meta_dict = {TAGS[key] : imageC.tag[key] for key in imageC.tag_v2}
                    # Print the tag/ value pairs
#                    # Print the tag/ value pairs
            #for tag in meta_dict:
                 #if tag not in ('GPSInfoIFD', 'DateTime', 'Document', 'ImageDescription', 'EXIF MakerNote'):
            #     if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            #        print ("Key: %s, value %s" % (tag, meta_dict[tag]))
                            
            props=GetPropsFromImage(imageC,file) # get maj/min axes etc
            label,probs = classify(imageC,classes) #CLASSIFY

            #deal with probabilities --- could move this to the Classify function
            CSV_row=file, label, probs[0].round(4), probs[1].round(4),probs[2].round(4), round(props[0],4), round(props[1],4),round(props[2],4),\
                    DateTime,GPSLatitudeRef,GPSLatitude,GPSLongitudeRef,GPSLongitude
            #print(f'CSV_ROW: {CSV_row}')
            # write a row to the csv file
            CSVwriter.writerow(CSV_row)
            
            if (sort): ##TODO PFC arrange to loop for length of classes list
                # move as you go to Classes
                
                if (label == classes[0]):
                    target_folder = images_folder + SEP + classes[0] + SEP
                    shutil.move(images_folder + SEP + file,target_folder )
                    #print(f'{file} {label} -> sort to: {target_folder}')
                elif (label == classes[1]):
                    target_folder = images_folder + SEP + classes[1] + SEP
                    shutil.move(images_folder + SEP + file,target_folder )
                    #print(f'{file} {label} -> sort to: {target_folder}')
                elif (label == classes[2]):
                    target_folder = images_folder + SEP + classes[2] + SEP
                    shutil.move(images_folder + SEP + file,target_folder )
                    #print(f'{file} {label} -> sort to: {target_folder}')
                else:
                    target_folder = images_folder + SEP + 'Unknown' + SEP
                    shutil.move(images_folder + SEP + file,target_folder )
                    #print(f'{file} {label} -> sort to: {target_folder}')
                    
                    
            Icount=Icount+1 # count each image
            
        CSV_fd.close() # o for each 10-min folder
    
    stop_time = time.time()
    total_time= ((stop_time-start_time)/60)
    print(f'Time:{round(total_time,4)} Images:{Icount}')

###########################################################
if __name__ == "__main__":
    main()




     
     
     
     



     





                      
