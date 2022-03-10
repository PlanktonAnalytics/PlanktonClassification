# PlanktonClassification
<b>Models, code and example data</b>

A combined effort by CEFAS, Plankton Analytics ltd. and the Turing Institute from a data Study Group in Dec. 2021 has resulted in a three category DNN model for plankton image sorting (derived by transfer learning a Resnet-50 CNN).

See https://github.com/alan-turing-institute/plankton-dsg-challenge

The images were collected using the Plankton Imager model 1 at a pixel resolution of 10 microns on board CEFAS RV Endeavour 2017-2020.

Scripts:
TorchExport.py reads a PyTorch (*.pth) model at 64bit double precision to Openvino (*.onnx) and reduced precision for faster classification
DNN_preSort.py offers a basic classification engine using the CopNonDetritus_42317_FP16 model. The model is embedded in the script.

A MAC laptop (2016 i7) runs the model and sorts the images in the predefined day sample/10-minute folder structure at a rate of about 30 images per second.
Each image has area and maj/min parameters, Date-Time and GPS coordinates extracted and saved to ./desc/DNN_reults.csv file

Phil Culverhouse

<b>Usage:</b>
Now have to specify model location. Works on Day samples, but will skip 10-min folders that do not have an Images subfolder.

By default 

modelname="CopNonDetritus_42317_FP32" # DSG DEC 2021 model, optimised by openVino by PFC. See https://zenodo.org/record/6124415#.YiRbMRMUXBn for models

labelstr="CopNonDetritus_42317_Labels.xml"

FIXED in the code, as these models are known to work.

<b>EXAMPLE: python3 DNN_PreSort.py --dpath=/Users/culverhouse/Code/openvino_optimised/data/rawdata/2020_10_21-short/ --mpath=/users/culverhouse/Code/openvino_optimised/models/Model-CEFAS-Pi-CopNonDetritus_42317/</b>

User MUST use the default path separator for linux/win10 as python attempts to add a '\' when hunting for folders... will crash!

<b>Outputs: </b>

Model:CopNonDetritus_42317_FP32

Classes: ['Copepoda', 'Detritus', 'NonCopepoda']

Time:0.0273 Images:47

and saves results to DNN_Results.csv in each 10-min folder subfolder Desc


<b>PUBLICATIONS</b>
1. Pitois SG, et al. (2021) A first approach to build and test the Copepod Mean Size and Total Abundance (CMSTA) ecological indicator using in-situ size measurements from the Plankton Imager (PI). Ecological Indicators 123 (2021) 107307. https://doi.org/10.1016/j.ecolind.2020.107307
2. Pitois SG, et al. (2018). Comparison of a cost-effective integrated plankton sampling and imaging instrument with traditional systems for mesozooplankton sampling in the Celtic Sea. Front. Mar. Sci. 5, 5. https://doi.org/10.3389/fmars.2018.00005
3. Scott J, et al.  (2021) In situ automated imaging, using the Plankton Imager, captures temporal variations in mesozooplankton using the Celtic Sea as a case study. J. Plankton Research. https://doi.org/10.1093/plankt/fbab018.


<b>INSTALLATION DETAILS</b>

There are two levels of operation available:
A) Using OpenVino to convert DNN models into ONNX and optimise the network for inference use, 
  allows TorchExport.py to run. 
   See https://pypi.org/project/torch/ and https://pypi.org/project/torchvision/
   
B) Using OpenVino just to load an optimised model and run inferences on data, using DNN_PreSort.py.
   See https://pypi.org/project/openvino/

For (A) you need to install the Torch libraries and OpenVino

pip install torch

pip install torchvision

pip install openvino

For (B) you only need to install OpenVino inference engine, but also opencv_python as a number of imaging functions use that library

pip install openvino

pip install pillow

pip install opencv-python




