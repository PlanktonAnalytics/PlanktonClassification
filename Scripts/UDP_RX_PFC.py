#!/usr/bin/env python

# PFC 15.03.23
# for Pi_Imager format UDP data stream processing
# see UserGuide/UDP-data-format.pdf
# The protocol was designed for fast processing of UDP packets which can arrive out of order. There is an array of 2048 file records allocated at boot, a large ring buffer of pointers to packets, and a stack of pointers to packets, pointing to a large array of packets, also allocated at boot.

import socket
from struct import * # for unpacking binary packets
import os
import pathlib # for creating filepaths


# UDP_IP = "127.0.0.1" # change to 192.168.0.255 for live running
UDP_IP = "0.0.0.0" # All interfaces

UDP_PORT = 5000

# Fpath="/Users/culverhouse/Desktop/SurveyData/"
Fpath="survey-data/"

pathlib.Path(Fpath).mkdir(parents=True, exist_ok=True) # create the file paths if necessary

# Implement a simple ring buffer

RING_SIZE = 2048

filenames =  ["" for x in range(RING_SIZE)] # current mapping of fields to filenames
buffers = [[0,1,2,3,4,5,6,7] for x in range(RING_SIZE)] # data packets that may arrive out of order
counts = [0 for x in range(RING_SIZE)] # number of packets received
unique_ids = [0 for x in range(RING_SIZE)] # current mapping of field indices to unique ids

#

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

# packet header structure
# Bytes
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


while True:
    data, addr = sock.recvfrom(8192) # buffer size is 8192 bytes max size

    Hash,Field,Part,UniqueID,TotalParts,DataSize,TAG,pack1,pack2 = unpack('IHHLHHHcc', data[0:24])

    print(f'{Hash},{Field},{Part},{UniqueID},{TotalParts},{DataSize},{TAG}')

    buffer = data[24:(DataSize+24)]

    if unique_ids[Field] != UniqueID :
        # This is a new UniqueID, so start over
        unique_ids[Field] = UniqueID
        counts[Field] = 0
        buffers[Field] = [0,1,2,3,4,5,6,7]

    #

    if (TAG==0):
        print(f"{TAG} NoTag - shouldn't be sent")

    elif (TAG==1):
        print(f"{TAG} Filename - the filename (first packet)")

        Fname= buffer.decode("ascii")

        print("received Fname: %s" % Fname)

        Fname = Fname.replace("\\", os.path.sep) # Convert Windows style paths to the host OS convention

        Fpath_Head_Tail=os.path.split(Fpath+Fname)
        print(f'UniqueID: {UniqueID}')
        pathlib.Path(Fpath_Head_Tail[0]).mkdir(parents=True, exist_ok=True)

        filenames[Field] = Fpath+Fname

    elif (TAG==2):
        print(f"{TAG} TiffIfd - a tiff header (second packet)")

    elif (TAG==3):
        print(f"{TAG} FileBody - ordinary file data (not a tiff file)")

    elif (TAG==4):
        print(f"{TAG} TiffBody - tiff file image data")

    else :
        print(f'Unknown TAG: {TAG}')

    #

    buffers[Field][Part]  = buffer

    counts[Field] += 1

    if (counts[Field] > TotalParts): # All packets received
        filename = filenames[Field]

        if not any([isinstance(buffers[Field][i], int) for i in range(0, TotalParts)]):
            # All packets intact
            if filename != "":
                print(f"Writing {filename} ...")
                f = open(filename, "wb+")
                for i in range(1,TotalParts):
                    f.write(buffers[Field][i])
        buffers[Field] = [0,1,2,3,4,5,6,7] # Reset, allow GC
