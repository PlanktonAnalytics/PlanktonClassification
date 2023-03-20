#!/usr/bin/env python

# PFC 15.03.23
# for Pi_Imager format UDP data stream processing
# see UserGuide/UDP-data-format.pdf
# The protocol was designed for fast processing of UDP packets which can arrive out of order. There is an array of 2048 file records allocated at boot, a large ring buffer of pointers to packets, and a stack of pointers to packets, pointing to a large array of packets, also allocated at boot.

import socket
from struct import * # for unpacking bibary packet
import os
import pathlib # for creating filepaths


# UDP_IP = "127.0.0.1" # change to 192.168.0.255 for live running
UDP_IP = "0.0.0.0" # All interfaces

UDP_PORT = 5000

# Fpath="/Users/culverhouse/Desktop/SurveyData/"
Fpath="survey-data/"


def dump(byte_array):
    hex_str = ''.join(['{:02x}'.format(b) for b in byte_array])
    print(hex_str)


# The 8-byte TIFF file header should contain the following
# information:
#
# Bytes 0-1: The byte order used within the file. Legal values are:
# "II" (4949.H) "MM" (4D4D.H). In the "II" format, byte order is
# always from the least significant byte to the most significant byte,
# for both 16-bit and 32-bit integers This is called little-endian
# byte order. In the "MM" format, byte order is always from most
# significant to least significant, for both 16-bit and 32-bit
# integers. This is called big-endian byte order.
#
# Bytes 2-3: An arbitrary but carefully chosen number (42) that
# further identifies the file as a TIFF file.The byte order depends on
# the value of Bytes 0-1.
#
# Bytes 4-7: The offset (in bytes) of the first IFD. The directory may
# be at any location in the file after the header but must begin on a
# word boundary. In particular, an Image File Directory may follow the
# image data it describes. Readers must follow the pointers wherever
# they may lead.The term byte offset is always used in this document
# to refer to a location with respect to the beginning of the TIFF
# file. The first byte of the file has an offset of 0.


def parse_tiff_header(byte_array):
    dump(byte_array)

    byte_order = False

    # Check endianness

    if byte_array[:2] == b'II':
        byte_order = "little"

    if byte_array[:2] == b'MM':
        byte_order = "big"

    if int.from_bytes(byte_array[2:4], byteorder=byte_order) != 42:
        print("Invalid check bytes")

    if byte_order:

        print("Valid TIFF file header.")
        # # Get the offset of the first IFD (Image File Directory)
        # ifd_offset = int.from_bytes(byte_array[4:8], byteorder=byte_order)

        # # Read the number of entries in the IFD
        # num_entries = int.from_bytes(byte_array[ifd_offset:ifd_offset+2], byteorder=byte_order)

        # # Loop through each IFD entry to find the width and height of the image
        # for i in range(num_entries):

        #     # Get the tag number of the IFD entry
        #     tag = int.from_bytes(byte_array[ifd_offset+2+i*12:ifd_offset+4+i*12], byteorder=byte_order)

        #     # Check if the tag corresponds to the width or height of the image
        #     if tag == 256:  # Width tag
        #         width = int.from_bytes(byte_array[ifd_offset+8+i*12:ifd_offset+12+i*12], byteorder=byte_order)
        #     elif tag == 257:  # Height tag
        #         height = int.from_bytes(byte_array[ifd_offset+8+i*12:ifd_offset+12+i*12], byteorder=byte_order)

        # print(f"Width: {width}, Height: {height}")
    else:
        print("Invalid TIFF file header.")


# Implement a naive ring buffer

RING_SIZE = 2048

filenames =  ["" for x in range(RING_SIZE)]

buffers = [[0,1,2,3,4,5,6,7] for x in range(RING_SIZE)]

counts = [0 for x in range(RING_SIZE)]

unique_ids = [0 for x in range(RING_SIZE)]


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

    print(f'{Hash},{Field},{Part},{UniqueID},{TotalParts},{DataSize},{TAG}')

    buffer = data[24:(DataSize+24)]

    if unique_ids[Field] != UniqueID :
        unique_ids[Field] = UniqueID
        counts[Field] = 0

    buffers[Field][Part]  = buffer

    counts[Field] += 1

    if counts[Field] >= TotalParts:
        filename = filenames[Field]
        if filename != "":
            newFile = open(filename, "wb+")
            for i in range(1,TotalParts):
                newFile.write(buffers[Field][i])


    if (TAG==0): #case 0: #illegal, corrupted packet
        print(f'TAG==0, Corrupted packet, ignoring')

    elif (TAG==1): # case 1: #Fname

        Fname= buffer.decode("ascii")

        print("received Fname: %s" % Fname)

        Fname = Fname.replace("\\", os.path.sep) # Convert Windows style paths to the host OS convention

        Fpath_Head_Tail=os.path.split(Fpath+Fname)
        print(f'UniqueID: {UniqueID}')
        pathlib.Path(Fpath_Head_Tail[0]).mkdir(parents=True, exist_ok=True)

        filenames[Field] = Fpath+Fname



    elif (TAG==2):
        print(f'TiFheader: {TAG}')

        dump(buffer)
        parse_tiff_header(buffer)

    elif (TAG==3):
        print(f'FileBody: {TAG}')


    elif (TAG==4):
         print(f'TiffBody: {TAG}')

    else :
        print(f'Unknown TAG: {TAG}')




## BUT need to track Unique ID to be in sequence else pack arrived out of order
