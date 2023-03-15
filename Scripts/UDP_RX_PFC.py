# PFC 15.03.23
# for Pi_Imager format UDP data stream processing
# see UserGuide/UDP-data-format.pdf
# The protocol was designed for fast processing of UDP packets which can arrive out of order. There is an array of 2048 file records allocated at boot, a large ring buffer of pointers to packets, and a stack of pointers to packets, pointing to a large array of packets, also allocated at boot.

import socket
from struct import * # for unpacking bibary packet
import os
import pathlib # for creating filepaths


UDP_IP = "127.0.0.1" # change to 192.168.0.255 for live running
UDP_PORT = 5000

Fpath="/Users/culverhouse/Desktop/SurveyData/"

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

## packet header structure
## Bytes 
#  0-3 Hash:		unsigned 32 bit INT (checksum)
#  4-5 FieldIDx:	unsigned 16 bit INT
#  6-7 PartIdx:		unsigned 16 bit INT
#  8-15 UniqueID:	unsigned 64 bit INT
#  16-17 TotalParts:	unsigned 16 bit INT
#  19-20 DataSize:	unsigned 16 bit INT
#  20-21 TAG:		unsigned 16 bit INT [0: no tag;1: Name; 2: TiffIfd; 3: FileBody; 4: Tiff Image data]
#  22-24 Packing bytes - normally zero

# example
#   Hash,Field,Part,UniqueID,TotalParts,DataSize,TAG,pack1,pack2 = unpack('IHHLHHHcc', b"\x92\x11\x00\x00-\x01\x00\x00_\xd0'2\xa6\x7f\x01\x00\x03\x00<\x00\x01\x00\x00\x00")

# results in this Hash=4498, Field=301, Part=0, UniqueID=421826759479391, TotalParts=3, DataSize=60, TAG=1, pack1=b'\x00', pack2=b'\x00')

pathlib.Path(Fpath).mkdir(parents=True, exist_ok=True) # create file paths if not done

## init things
Hash=0
Field=0
Part=0
UniqueID=0
TotalParts=0
DataSize=0
TAG=0
TiffHeader=0
TiffData=0
FileBody=0
while True:
    data, addr = sock.recvfrom(8192) # buffer size is 8192 bytes max size
    Hash,Field,Part,UniqueID,TotalParts,DataSize,TAG,pack1,pack2 = unpack('IHHLHHHcc', data[0:24])
    if (TAG==0): #case 0: #illegal, corrupted packet
        print(f'TAG==0, Corrupted packet, ignoring')
        
    elif (TAG==1): # case 1: #Fname
        Fname= data[24:].decode('ascii')
        Fpath_Head_Tail=os.path.split(Fpath+Fname)
        print(f'UniqueID: {UniqueID}')
        pathlib.Path(Fpath_Head_Tail[0]).mkdir(parents=True, exist_ok=True)
        print("received message: %s" % Fname)
        TotalParts=TotalParts-1
    elif (TAG==2): # case 2: #TIFF header, fetch rest of image too
        # next is TIFF header
        # then TIFF data
        print(f'TAG: {TAG}')
        TiffHeader, addr = sock.recvfrom(8192)
        newFile.write(TiffHeader) ## DOESN'T work, TIFF file is not
        TotalParts=TotalParts-1
    elif (TAG==3): #case 3: #FileBody
        newFile = open(Fpath+Fname, "wb+")
        while (TotalParts>0):
            FileBody, addr = sock.recvfrom(8192)
            # just assume one packet of data (only for small files
            newFile.write(FileBody)
            TotalParts=TotalParts-1
            
    elif (TAG==4):
        TotalParts=TotalParts-1
        newFile = open(Fpath+Fname, "wb+")
        while (TotalParts>0):
            TiffData , addr = sock.recvfrom(8192)
            newFile.write(TiffData) ## DOESN'T work, TIFF file is not readable as a TIF for some reason.
            TotalParts=TotalParts-1
    else :
        print(f'Unknown TAG: {TAG}')
        
    


## BUT need to track Unique ID to be in sequence else pack arrived out of order

    
