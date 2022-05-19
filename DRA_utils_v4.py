#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:41:25 2020

@author: avdonin
"""

import os as os
import numpy as np
#import array

###############################################################################
# makePeriodic_rawModel - this function creates a periodic raw model: 
# it mirrors the raw model in the desired direction and appends it to the original model.
# This function works only with 3D models. The size of the model in the direction of the operation is 2*N-1.
# FUNCTION ARGUMENTS:
#   inputRawFile - path to the input .raw file (including .raw extension!)
#   outputRawFile - path to the output .raw file (including .raw extension!)
#   dim_size - array with dimensions of the input model, e.g. [300,300,300]
#   direction - direction of the operation: 'X-', 'X+',
#                                           'Y-', 'Y+',
#                                           'Z-', 'Z+'
# EXAMPLE: create a periodic model in Z+ direction:
#          makePeriodic_rawModel('input.raw',outputZ.raw',[300,300,300],'Z+')

def makePeriodic_rawModel(inputRawFile,outputRawFile,dim_size,direction):
    import numpy as np
    # open raw file
    f = open(inputRawFile,'rb') #only opens the file for reading
    im=np.fromfile(f,dtype=np.uint8) # read data as uchar
    # according to ParaView visualization: im(zDirection,yDirection,xDirection)  
    im=im.reshape(dim_size[2],dim_size[1],dim_size[0])
    f.close()
    # generate periodic core sample
    if(direction=='Z+'):
        imPeriodic=np.concatenate((im[:-1,:,:],np.flip(im,0)),0)
    elif(direction=='Y+'):
        imPeriodic=np.concatenate((im[:,:-1,:],np.flip(im,1)),1)
    elif(direction=='X+'):
        imPeriodic=np.concatenate((im[:,:,:-1],np.flip(im,2)),2)
    elif(direction=='Z-'):
        imPeriodic=np.concatenate((np.flip(im,0),im[1:,:,:],),0)
    elif(direction=='Y-'):
        imPeriodic=np.concatenate((np.flip(im,1),im[:,1:,:],),1)
    elif(direction=='X-'):
        imPeriodic=np.concatenate((np.flip(im,2),im[:,:,1:],),2)        
    # create raw file
    f = open(outputRawFile, "wb")
    f.write(imPeriodic)
    f.close()
    print('Generated periodic model has dimensions ',[np.size(imPeriodic,2),np.size(imPeriodic,1),np.size(imPeriodic,0)])
    
    
###############################################################################
# addPoreLayers_rawModel - this function creates additional pore layers (zeros) around the raw digital rock model
# This function works only with 3D models. The size of the model increases according to the added layers
# FUNCTION ARGUMENTS:
#   inputRawFile - path to the input .raw file (including .raw extension!)
#   outputRawFile - path to the output .raw file (including .raw extension!)
#   dim_size - array with dimensions of the input model, e.g. [300,300,300]
#   direction - direction of the operation: 'X','Y','Z'
#   layers - 2d array with additional layers. First number adds layers in negative axis direction,
#            the second number adds layers in positive axis direction, e.g. [3,3]
# EXAMPLE: add 3 layers in Z direction from both sides:
#          addPoreLayers_rawModel('input.raw',outputZ.raw',[300,300,300],'Z',[3,3])

def addPoreLayers_rawModel(inputRawFile,outputRawFile,dim_size,direction,layers):
    import numpy as np
    # open raw file
    f = open(inputRawFile,'rb') #only opens the file for reading
    im=np.fromfile(f,dtype=np.uint8) # read data as uchar
    # according to ParaView visualization: im(zDirection,yDirection,xDirection)  
    im=im.reshape(dim_size[2],dim_size[1],dim_size[0])
    f.close()
    # generate additional pore layers:
    if(direction=='Z'):
        layerNegative=np.zeros([layers[0],dim_size[1],dim_size[0]],dtype='uint8')
        layerPositive=np.zeros([layers[1],dim_size[1],dim_size[0]],dtype='uint8')       
        imLayers=np.concatenate((layerNegative,im,layerPositive),0)
    if(direction=='Y'):
        layerNegative=np.zeros([dim_size[2],layers[0],dim_size[0]],dtype='uint8')
        layerPositive=np.zeros([dim_size[2],layers[1],dim_size[1]],dtype='uint8')       
        imLayers=np.concatenate((layerNegative,im,layerPositive),1)
    if(direction=='X'):
        layerNegative=np.zeros([dim_size[2],dim_size[1],layers[0]],dtype='uint8')
        layerPositive=np.zeros([dim_size[2],dim_size[1],layers[1]],dtype='uint8')       
        imLayers=np.concatenate((layerNegative,im,layerPositive),2)
    # create raw file
    f = open(outputRawFile, "wb")
    f.write(imLayers)
    f.close()
    print('Generated model with additional pore layers has dimensions ',[np.size(imLayers,2),np.size(imLayers,1),np.size(imLayers,0)])

   
###############################################################################
# read MHD file
# INPUT: filename - path to the .mhd file (including .mhd extension!)
# OUTPUT: an mhd dictionary 
# EXAMPLE: mhd_dict=read_mhd_file('input.mhd')
def read_mhd_file(filename):
    """Return a dictionary of meta data from meta header file"""
    fileIN = open(filename, "r")
    line = fileIN.readline()

    mhd_dict = {}
    tag_set1 = ['ObjectType','NDims','DimSize','ElementType','ElementDataFile']
    tag_set2 = ['BinaryData','BinaryDataByteOrderMSB','CompressedData','CompressedDataSize']
    tag_set3 = ['Offset','CenterOfRotation','AnatomicalOrientation','ElementSpacing','TransformMatrix']
    tag_set4 = ['Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']
    tag_set = []

    tag_set.extend(tag_set1)
    tag_set.extend(tag_set2)
    tag_set.extend(tag_set3)
    tag_set.extend(tag_set4)
    tag_flag = [False]*len(tag_set)
    while line:
        tags = str.split(line,'=')
        #print tags[0]
        for i in range(len(tag_set)):
            tag = tag_set[i]
            if (str.strip(tags[0]) == tag) and (not tag_flag[i]):
                #print tags[1]
                mhd_dict[tag] = str.strip(tags[1])
                tag_flag[i] = True
        line = fileIN.readline()
    #print comment
    fileIN.close()
    return mhd_dict


###############################################################################
# read RAW and MHD file given a MHD file 
# INPUT: filename - path to the .mhd file (including .mhd extension!)
# при загрузке бинарного файла .raw направоения по оси x и z поменяны местами im(zDirection,yDirection,xDirection)  
# OUTPUT: 3D image (uint8) + a mhd dictionary 
# EXAMPLE: im, mhd_dict=load_raw_data_with_mhd('input.mhd')
def load_raw_data_with_mhd(filename):
    mhd_dict=read_mhd_file(filename)
    dim_size=[int(i) for i in mhd_dict['DimSize'].split()]
    os.path.split(filename)[0]
    # Далее идет сама загрузка файла
    pathToRaw=os.path.join(os.path.split(filename)[0],mhd_dict['ElementDataFile'])
    with open(pathToRaw, 'rb') as f:
        im = np.fromfile(f, dtype=np.uint8)
    # при загрузке бинарного файла .raw направоения по оси x и z поменяны местами im(zDirection,yDirection,xDirection)  
    im = im.reshape(dim_size[2], dim_size[1], dim_size[0])
    return (im, mhd_dict)


###############################################################################
# create a dummy mhd dictionary with all required tags
# OUTPUT: an mhd dictionary 
# EXAMPLE: mhd_dict=create_mhd_dict()
def create_mhd_dict():
    mhd_dict={}
    mhd_dict['ObjectType']='Image'
    mhd_dict['NDims']='3'
    mhd_dict['ElementType']='MET_UCHAR'
    mhd_dict['Offset']='0 0 0'
    mhd_dict['ElementDataFile']=''
    mhd_dict['ElementSpacing']=''
    mhd_dict['DimSize']=''
    return mhd_dict
    
 
###############################################################################
# write MHD file
# INPUT: filename - path to the .mhd file (including .mhd extension!) that has to be created
#        mhd_dict - dictionary with keys that should be written 
# EXAMPLE: mhd_dict=create_mhd_dict(); write_mhd_file('test.mhd',mhd_dict)
def write_mhd_file(filename, mhd_dict):
    header = ''
    # do not use tags = mhd_dict.keys() because the order of tags matters
    tags = ['ObjectType','NDims','BinaryData',
       'BinaryDataByteOrderMSB','CompressedData','CompressedDataSize',
       'TransformMatrix','Offset','CenterOfRotation',
       'AnatomicalOrientation',
       'ElementSpacing',
       'DimSize',
       'ElementType',
       'ElementDataFile',
       'Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']
    for tag in tags:
        if tag in mhd_dict.keys():
            header += '%s = %s\n'%(tag,mhd_dict[tag])
    f = open(filename,'w')
    f.write(header)
    f.close()


# ###############################################################################
# write raw and MHD file
# INPUT: mhdFilename - path to the file (with .mhd extension!) that has to be created.
#        mhd_dict - dictionary with keys that should be written 
#        imFilename - path to the file (with .raw extension!) that has to be created.
#        im - image[Z,Y,X] in uint8 format that should be written 
# EXAMPLE: write_raw_data_with_mhd('test.mhd',mhd_dict,'test.raw',im)
def write_raw_data_with_mhd(mhdFilename, mhd_dict,imFilename,im):
    # save .mhd
    header =''
    # do not use tags = mhd_dict.keys() because the order of tags matters
    tags = ['ObjectType','NDims','BinaryData',
        'BinaryDataByteOrderMSB','CompressedData','CompressedDataSize',
        'TransformMatrix','Offset','CenterOfRotation',
        'AnatomicalOrientation',
        'ElementSpacing',
        'DimSize',
        'ElementType',
        'ElementDataFile',
        'Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']
    for tag in tags:
        if tag in mhd_dict.keys():
            header += '%s = %s\n'%(tag,mhd_dict[tag])
    with open(mhdFilename, 'w') as f:
        f.write(header)
    # save .raw
    im=np.ascontiguousarray(im)
    with open(imFilename, 'wb') as f:
        f.write(im)
    
    
###############################################################################
# swap axes in the .raw model. The function reads the .mhd file with the corresponding .raw file,
# swaps X and Z directions and stores the new model (.mhd + .raw)
# INPUT: mhd_file - path to the .mhd file (including .mhd extension!) that should be loaded
#        mhd_fileSwap - path to the .mhd file (including .mhd extension!) that should be created
#        swapOrder - one of three possible swap operations: 'XZ','YZ','XY'
# EXAMPLE: swapDirections('pathToFile/S5.mhd','anotherPathToFile/S5_swapXZ.mhd','XZ')
def swapDirections(mhd_file,mhd_fileSwap,swapOrder):
    import os
    import numpy as np
    # mhd_file='S5/S5.mhd'
    # mhd_fileSwap='S5/S5_swapXZ.mhd'
    mhd_dict=read_mhd_file(mhd_file)
    pathToRaw=os.path.join(os.path.dirname(mhd_file),mhd_dict['ElementDataFile'])
    dimX = int(mhd_dict['DimSize'].split()[0]) # X
    dimY = int(mhd_dict['DimSize'].split()[1]) # Y
    dimZ = int(mhd_dict['DimSize'].split()[2]) # Z
    # load raw file
    if(mhd_dict['ElementType']=='MET_UCHAR'):
       with open(pathToRaw, 'rb') as f:
           im = np.fromfile(f, dtype=np.uint8)
       # в питоне индексы не совпадаю с реальными осями -> в питоне im[Z,Y,X]!
       im = im.reshape(dimZ,dimY,dimX)
       # create new mhd file
       mhd_dictSwap=mhd_dict.copy()
       mhd_dictSwap['ElementDataFile']=os.path.splitext(os.path.basename(mhd_fileSwap))[0]+'.raw'
       # меняем местами оси:
       if(swapOrder=='XZ'):
           im = np.swapaxes(im, 0, 2)
           mhd_dictSwap['DimSize']=str(dimZ)+' '+str(dimY)+' '+str(dimX)
       elif(swapOrder=='XY'):
           im = np.swapaxes(im, 1, 2)
           mhd_dictSwap['DimSize']=str(dimY)+' '+str(dimX)+' '+str(dimZ) 
       elif(swapOrder=='YZ'):
           im = np.swapaxes(im, 0, 1)
           mhd_dictSwap['DimSize']=str(dimX)+' '+str(dimZ)+' '+str(dimY) 
       else:
           print('Error: no swap was performed, since swapOrder is set wrong!')
       write_mhd_file(mhd_fileSwap,mhd_dictSwap)
       # save new raw file Сохранение в бинарном формате uint8:
       pathToRawSwap=os.path.splitext(mhd_fileSwap)[0]+'.raw'
       with open(pathToRawSwap, "wb") as f:
           f.write(np.ascontiguousarray(im,dtype=np.uint8))
    else:
        print('Error: no model is generated, since ElementType is not MET_UCHAR!')
    
    
###############################################################################
# converts .mhd file to .xmf file. It writes scalar quantities, stored in .raw file as cell values of the cubic mesh.
# INPUT: MHDfileName - path to the .mhd file (including .mhd extension!)
#        XDMFfileName - path to the .xmf file (including .xmf extension!)
#        AttributeName - name of the field    
# OUTPUT: a stored .xmf file
# EXAMPLE: mhdToXdmf('input.mhd','output.xmf','pressure')    
def mhdToXdmf(MHDfileName,XDMFfileName,AttributeName):
    mhd=read_mhd_file(MHDfileName)
    ElementType=mhd['ElementType']
    DimSize=[int(mhd['DimSize'].split()[0]),int(mhd['DimSize'].split()[1]),int(mhd['DimSize'].split()[2])]
    ElementSpacing=[float(mhd['ElementSpacing'].split()[0]),float(mhd['ElementSpacing'].split()[1]),float(mhd['ElementSpacing'].split()[2])]
    ElementDataFile=mhd['ElementDataFile']
    NDims=int(mhd['NDims'])
    
    if(NDims!=3):
        raise Exception("NDims !=3, this function works only for NDims=3.")
    
    if((ElementType!='MET_FLOAT') & (ElementType!='MET_UCHAR') ):
        raise Exception("ElementType !=MET_FLOAT or MET_UCHAR, this function works only for ElementType MET_UCHAR and MET_FLOAT")
    
    if(ElementType=='MET_FLOAT'):
        XDMF_NumberType='Float' # 32 bit
    elif(ElementType=='MET_UCHAR'):
        XDMF_NumberType='Uchar' # unsigned 8 bit
        
    # write a .xmf file
    f=open(XDMFfileName,'wt')
    f.writelines([
            '<?xml version="1.0" ?>\n',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
            '<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n',
            '  <Domain>\n',
            '	 <Grid GridType="Uniform">\n',
            '		<Topology TopologyType="3DCORECTMesh" Dimensions="'+str(DimSize[0]+1)+' '+str(DimSize[1]+1)+' '+str(DimSize[2]+1)+'"/>\n',
            '		<Geometry GeometryType="ORIGIN_DXDYDZ">\n',
            '		  <DataItem Name="Origin" Dimensions="'+str(NDims)+'" NumberType="Float" Precision="4" Format="XML">\n',
            '				0 0 0\n',
            '			</DataItem>\n',
            '			<DataItem Name="Spacing" Dimensions="'+str(NDims)+'" NumberType="Float" Precision="4" Format="XML">\n',
            '				'+str(ElementSpacing[0])+' '+str(ElementSpacing[1])+' '+str(ElementSpacing[2])+'\n',
            '			</DataItem>\n',
            '		</Geometry>\n',
            '		<Attribute Name="'+AttributeName+'" Active="1" AttributeType="Scalar" Center="Cell">\n',
            '		  <DataItem Dimensions="'+str(DimSize[0])+' '+str(DimSize[1])+' '+str(DimSize[2])+'" DataType="'+XDMF_NumberType+'" Precision="4" Format="Binary"> '+ElementDataFile+'\n',
            '		  </DataItem>\n',
            '		</Attribute>\n',		
            ' </Grid>\n',
            '  </Domain>\n',
            '</Xdmf>\n'])
    f.close()


# This function removes all non-zero elements, that are not connected to a chosen face.
def connectivityToOneFace(im,face,conn):
    #face = 0 -> first dimension lower bound, 3 -> first dimension upper bound, 
    #       1 -> second dimension lower bound, 4 -> second dimension upper bound, 
    #       1 -> third dimension lower bound, 5 -> third dimension upper bound, 
    # conn (int) – for the 3D the options are 6 and 26, similarily for face and edge neighbors.
    
    import numpy as np
    import porespy as ps
    
    imTMP=np.zeros([np.size(im,0)+2,np.size(im,1)+2,np.size(im,2)+2],dtype=np.bool)
    imTMP[1:-1,1:-1,1:-1]=im
    if(face==0):
        imTMP[0,:,:]=imTMP[1,:,:]
    elif(face==1):
        imTMP[:,0,:]=imTMP[:,1,:]
    elif(face==2):
        imTMP[:,:,0]=imTMP[:,:,1]
    if(face==3):
        imTMP[-1,:,:]=imTMP[-2,:,:]
    elif(face==4):
        imTMP[:,-1,:]=imTMP[:,-2,:]    
    elif(face==5):
        imTMP[:,:,-1]=imTMP[:,:,-2]
    
    holes = ps.filters.find_disconnected_voxels(imTMP,conn)
    imTMP[holes] = False
    return imTMP[1:-1,1:-1,1:-1]




