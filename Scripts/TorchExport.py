# -*- coding: utf-8 -*-

#TorchExport.py
############################# PFC Feb 2022 ##########################
# Phil Culverhouse, Plankton Analytics Ltd.
# www.planktonanalytics.com

# run this after initialising the environment for OpenVino

import torch
import torchvision
from torchvision.transforms import functional
from PIL import Image

import os
import shutil
import numpy as np
import onnx # for checking the result

###################### MUST BE SET MANUALLY AT PRESENT *******
num_classes=3

# Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.eval()

# Loading saved model weights
#model_state_dict = torch.load(f'/Users/culverhouse/Code/openvino_optimised/models/Turing_classifier/CopNonDetritus.pth', map_location='cpu')
modelname='CopNonDetritus_42317_FP64'
## WIN10 model_state_dict = torch.load(f'c:/openvino_optimised/models/Turing_classifier/{modelname}', map_location='cpu')
model_state_dict = torch.load(f'/Users/culverhouse/Code/openvino_optimised/models/DSG_DNN_PreSort_Model/{modelname}.pth', map_location='cpu')

model.load_state_dict(model_state_dict)

# Create dummy input for the model. It will be used to run the model inside export function.
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
        model.cpu(),
        x.cpu(),
        modelname +'.onnx', # eg.'model.onnx',
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        opset_version=11, #need v11 to ensure smooth translation of PyTorch to ONNX
        verbose=False,
        input_names = ['input'],              # the model's input names
        output_names = ['output'])


#Set up things
#1. /opt/intel/openvino_2021.4.752/bin/setupvars.sh
#2. /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh onnx

#3. run model optimiser
# python3 /opt/intel/openvino_2021.4.752/deployment_tools/model_optimizer/mo.py --input_model=/Users/culverhouse/Code/openvino_optimised/models/model.onnx --data_type=FP16;

# to convert to XML and BIN files, also can specify FP resolution, default is FP32 (not FP64 AS PYTORCH)
# FP16 produces a faster classifier.
# on a Mac Book Pro (2016) approx. 30 images per second can be labelled an sorted into categories
# A GPU will be significantly faster.
