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
# python3 ./models/Scripts/DNN_PreSort.py --dpath=./data/rawdata/2020_10_21-short/ --mpath=./models/Model-CEFAS-Pi-CopNonDetritus_42317/ --sort --label=True

# Option to moves images into sorted folders (done by PI_Label() in c++) use
# --sort
#
# option to label each image with a TIF TAG use
# --label=True

# outputs a DNN_Results.csv file in Desc folder with the ten-minute folder.
# structure [Fname, Class, p0, p1, p2, Area, Major, Minor, GPS]
# see example in repo.



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
##import exifread ##pip install ExifRead

import cv2
import xml.etree.ElementTree as ET

#two alteratives for run time
from openvino.inference_engine import IECore

## globals
global height
global width

########## Manually set to location of folders for non-main() operation #####################
SEP='/'
DIR="/Users/culverhouse/Code/"
if sys.platform == 'win32':
  SEP='\\'
  DIR='C:'

#INrootDIR=DIR + SEP + "openvino_optimised" + SEP + "data" + SEP + "rawdata" + SEP + ""

######################### MODEL DETAILS #####################
modelname="CopNonDetritus_42317_FP32" # DSG DEC 2021 model

labelstr="CopNonDetritus_42317_Labels.xml"
#labelstr="CopNonDetritus_42317_OpenCV_Labels.xml" ## PFC 2 ways of formatting XML, the OpenCV one is the easiest to port Python3/c++"

########## CHANGE TO UPDATE MODEL LOCATION #################
#modelDIR= SEP + "openvino_optimised" + SEP + "models" + SEP + "Model-CEFAS-Pi-CopNonDetritus_42317" + SEP
############################################################


################## CLASSES used to train DNN model ########################
## note keep the orer the same as for the training regime ##

#classes = ["Copepoda", "Detritus", "NONCopepoda"]
## now read from the Labels.xml in the same location as the model
def ReadLabels(path):

    classes=[]
    try:
    # PFC USE EITHER XML FORMAT
    #    path="/Users/culverhouse/Code/openvino_optimised/models/DSG_DNN_PreSort_Model/CopNonDetritus_42317_Labels.xml"
        tree = ET.parse(path)
        root = tree.getroot()

        for child in root:
            name = child.get('Class')
            classes.append(name)
        #print(f'ReadLabels:{Classes}')
    except:
        print(f'Failed to find XML Labels {path}\n')
        
    return(classes)
    
# PFC OR USE OPENCV FILESTORAGE FORMAT
# use opencv filestorage class as its compatible with c++ & python
    #path="/Users/culverhouse/Code/openvino_optimised/models/Model-CEFAS-Pi-CopNonDetritus_42317/CopNonDetritus_42317_OpenCV_Labels.xml"
#    if os.path.exists(path):
#        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
#        fn = fs.getNode("Labels")
#        Classes=[]
#        for i in range(fn.size()):
#            Classes.append(fn.at(i).string())
#        #print(f'ReadLabels:{Classes}')
#        return(Classes)
#    else:
#        print(f'Error: no Labels file:{path}')
#        exit (-2)
            
################# Softmax() ########################
# PFC from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
# PFC need to figure how to do this insie the ONNX model
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
    
##################### CLASSIFY #####################
def classify(imageC,classes, width, height, Fname):

    try:
         # resize image and change dims to fit neural network input
         imageC=np.asarray(imageC) # make an array that resize can use
         #print(f'{width},{height},{imageC.shape}')
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
         
    except:
        print (f'Classifier failed for {Fname}\n')
        print(f'PROBS: {probs}\n')
        return (0,0)
     
############## measure image for area and fit ellipse ##################

def GetPropsFromImage(imageC, Fname):

    area=0
    d1=0
    d2=0
    
    try:
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
    except:
        print(f' Failed to find a contour in {Fname}\n')
    
    return([area, d1, d2])
    
############# TIFF META TAGS CODE #######################
# uses tiftools --> pip install tifftools

import tifftools
def read_metadata(Fpath):
    
    TAGS={'Fname':'empty','DocumentName':'empty', 'DateTime':'empty','GPSLatitudeRef':'empty',
    'GPSLatitude':'empty','GPSLongitudeRef':'empty','GPSLongitude':'empty'}

    try:
        im = Image.open(Fpath)
        info = tifftools.read_tiff(Fpath) # get meta data
        ##dump = tifftools.tiff_dump(Fpath,30)
        # also read the structure and pul out TAG items
        # for these tags
        TAGS['DateTime'] = info['ifds'][0] ['tags'].get(306) ['data'] # std TIF TAGS
        
        GPS=info['ifds'][0] ['tags'].get(34853)['ifds'] # GPS TAGS
        
        TAGS['GPSLatitudeRef'] = GPS[0][0]['tags'][1]['data']
        TAGS['GPSLatitude'] = GPS[0][0]['tags'][2]['data']
        TAGS['GPSLongitudeRef'] = GPS[0][0]['tags'][3]['data']
        TAGS['GPSLongitude'] = GPS[0][0]['tags'][4]['data']
    except:
        print(f'reading TIFTAGS failed {Fpath}\n')
    #im.close()
    return(TAGS)


# insert into TF META DATA the following tags
# Object Class as 'ImageDescription'
# 'DNN_PreSort: Model-CEFAS-Pi-CopNonDetritus_42317' as 'Software'
# 'Plankton Analytics' as 'Make'
# 'Pi-10' as 'Model'
# 'Plankton Analytics PI_Imager v1.0' as 'Artist'

def update_metadata(TEMP_FILE, Fpath, Class):
    
    try:
        im = Image.open(Fpath)
        info = tifftools.read_tiff(Fpath) # get meta data

        # get TAG stuff that is important from TIF image file.
        BIGENDIAN=info['bigEndian']
        BIGTIFF=info['bigtiff']

    #Classification
        info['ifds'][0]['tags'][tifftools.Tag.ImageDescription.value] = {
        'data': Class,
        'datatype': tifftools.Datatype.ASCII }
        
    #Classifier details
        info['ifds'][0]['tags'][tifftools.Tag.Software.value] = {
        'data': 'DNN_PreSort: Model-CEFAS-Pi-CopNonDetritus_42317',
        'datatype': tifftools.Datatype.ASCII }

    # Instrument details
    #    info['ifds'][0]['tags'][tifftools.Tag.Make.value] = {
    #    'data': 'Plankton Analytics',
    #    'datatype': tifftools.Datatype.ASCII }
    #
    #    info['ifds'][0]['tags'][tifftools.Tag.Model.value] = {
    #    'data': 'Pi-10',
    #    'datatype': tifftools.Datatype.ASCII }
    # unsure about this one
        info['ifds'][0]['tags'][tifftools.Tag.Artist.value] = {
        'data': 'Plankton Analytics PI_Imager v1.0', ## Could have Labeller here
        'datatype': tifftools.Datatype.ASCII }

        tifftools.write_tiff(info, TEMP_FILE, BIGENDIAN, BIGTIFF, allowExisting=True)
        os.rename(TEMP_FILE, Fpath)
        #im.close()
    except:
        print(f'TIFTAG write failed for {Fpath}\n')


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

    try:
        os.makedirs(folder, exist_ok=True)
        csv_fileSTR=os.path.join(folder, "DNN_Results.csv")
        #print(csv_fileSTR)
        CSVfile = open(csv_fileSTR, 'w')
        CSVwriter = csv.writer(CSVfile)
        CSVwriter.writerow([modelname]) # audit trail save the model
        
        headers=['Fname' , 'Class','Copepoda_(p)','Detritus_(p)','NONCopepoda_(p)', 'Area', 'Major', 'Minor', 'DateTime','GPSLatitudeRef','GPSLatitude','GPSLongitudeRef','GPSLongitude']
        CSVwriter.writerow(headers),
    except:
        print(f'Failed to creeate DNN_results.csv in {folder}\n')
        
    return CSVfile, CSVwriter #so we can close it
    
#### check if folder exists, list it
def listdirs(folder):
    if os.path.exists(folder):
         return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    else:
         return []
         
def usage():
    print('-d, --dpath xyz : set location of the sample, either day or 10-min folder')
    print('-m, --mpath xyz : set location of the model')
    print('-s, --sort : sort classified images into Class folders')
    print('-l, --label : label the original TIFF images with their Class label')
    print('Creates ./Desc/DNN_Results.csv in each ten-minute folder of the DAYS sample')
    print('-h, --help : this message')
    
###### MAIN runs through the folder hierarchy fining 10-minute folders with "/Images" subfolder
# assumes not much else in the folder structure
# excepting std Plankton Imager files (and ZIP files for easy teset image storing
# DNN_PreSort -fpath -sort

## example python3 ./models/Scripts/DNN_PreSort.py --dpath=./data/rawdata/2020_10_21-short/ --mpath=./models/Model-CEFAS-Pi-CopNonDetritus_42317/ --sort --label=True



def main():

    ## DNN engine vars must be global, but decl inside main()!!
    global ie_core
    global net
    global exec_net
    global input_key
    global output_key
    
    # set GPS vars
    DateTime=""
    GPSLatitudeRef=""
    GPSLatitude=""
    GPSLongitudeRef=""
    GPSLongitude=""
    
    
    ModelDIR=""
    InputDIR=""
    sort=False # set true by cmd
    LABEL=False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "dpath=", "mpath=", "sort", "label"])
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
        elif o in ("-d", "--dpath"):
            InputDIR = a
        elif o in ("-m", "--mpath"):
            ModelDIR = a
        elif o in ("-s", "--sort"):
            sort = True
        elif o in ("-l", "--label"):
            LABEL = True
        else:
            assert False, "unhandled option"
        #print (f'Args:: {InputDIR}, {ModelDIR}, {sort}')
        
    if not (os.path.exists(InputDIR)):
        assert False, "Cannot find data"
    if not (os.path.exists(ModelDIR)):
        assert False, "Cannot find model"
    
    # ################ LOAD model ####################
    # note: could set this as cmd arg
    converted_model_path = ModelDIR + modelname + ".xml"
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
        
    # then classifying images
    # measure processing time
    start_time = time.time()

    classes=ReadLabels(path = ModelDIR + labelstr)
    print(f'Classes: {classes}')

    Icount=0
    ten_min_folders=listdirs(InputDIR)

    for folder_item in ten_min_folders:
    
        try:
        
            if SEP + 'Desc' in folder_item:
                print(f'Desc-Images removed {folder_item}')
                continue ## PFC there may be an Images folder in the Desc folder, skip it
        
            images_folder=InputDIR  + folder_item + SEP + "Images" + SEP
            if not (os.path.exists(images_folder)):
                continue
                
            TEMP_FILE = images_folder + SEP + 'TEMP.tif' # used for tiff_write()
                
            Desc_folder=InputDIR  + folder_item + SEP + "Desc" + SEP
            #print(f' csv results: {Desc_folder}')

            CSV_fd, CSVwriter = Create_CSV(Desc_folder) # create target CSV results file
            shutil.copy2(ModelDIR+labelstr , Desc_folder + labelstr) ## copy Labels.xml to target
            
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

                # cannot use opencv as the Intel openvino opencv binary libraries do not have TIF lib!!
                try:
                    imageC =  Image.open(Fpath)
                except:
                    print(f' Could not open image in {Fpath}')
                
                TAGS = read_metadata(Fpath) # now uses TIFTOOLS for TAG operations
                #print(f'TAGS:{read_metadata(Fpath)}')
                DateTime = TAGS['DateTime']
                GPSLatitudeRef = TAGS['GPSLatitudeRef']
                GPSLatitude=TAGS['GPSLatitude']
                GPSLongitudeRef=TAGS['GPSLongitudeRef']
                GPSLongitude=TAGS['GPSLongitude']
            
                props=GetPropsFromImage(imageC,file) # get maj/min axes etc
                label,probs = classify(imageC,classes, width, height, Fpath) #CLASSIFY
                
                if LABEL==True:
                    update_metadata(TEMP_FILE, Fpath, label) # & update TIF with Class label etc.

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
            
        except:
            print(f' Image BROKE: {folder_item} in {Fpath}\n')
    
    stop_time = time.time()
    total_time= ((stop_time-start_time)/60)
    print(f'Time:{round(total_time,4)} Images:{Icount}')

        
    

###########################################################
if __name__ == "__main__":
    main()




     
     
     
     



     





                      
