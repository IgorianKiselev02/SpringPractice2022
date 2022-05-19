#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Импорт библиотек
import porespy as ps # porespy version 1.2.0
print('porespy version '+ps.__version__) 
from skimage.morphology import ball
import matplotlib.pyplot as plt # matplotlib 3.2.2
import scipy as sp # scipy version 1.5.0
print('scipy version '+sp.__version__) 
import numpy as np # numpy version 1.19.0
print('numpy version '+np.__version__) 
import openpnm as op # openpnm version 2.3.3
print('openpnm version '+op.__version__) 
import trimesh # trimesh version 3.8.4
print('trimesh version '+trimesh.__version__) 
from stl import mesh # numpy-stl 2.11.2  
#vd.embedWindow('k3d')
cm=1/2.54  # centimeters in inches
import DRA_utils_v4 as dra
import os 
import astra
import pylab
from skimage.transform import radon, iradon
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from datetime import datetime
import time
from skimage import data
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
from skimage import img_as_float
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import filters
import random
import skimage

###############################################################################
# ОГРАНИЧЕНИЯ:
# пока что скрипт работает только с одинаковым разрешением по всем трем направлениям 


###############################################################################
# ЗАГРУЗКА ЦИФРОВОЙ МОДЕЛИ КЕРНА
# Пользователь должен:
# 1) выбрать бинарный файл с цифровой моделью керна (pathToRaw),
# 2) задать размерность модели, т.е. кол-во вокселей по каждой оси (dim_size),
# 3) задать физический размер вокселя (resolution) в метрах.
# fileNameMHD='sk247/sk247.mhd'
#fileNameMHD='Berea/Berea.mhd'
fileNameMHD='Berea/Berea.mhd'

im,mhd=dra.load_raw_data_with_mhd(fileNameMHD)
dim_size =[int(mhd['DimSize'].split()[0]),int(mhd['DimSize'].split()[1]),int(mhd['DimSize'].split()[2])] # кол-во вокселей по направлению осей X, Y, Z
resolution =[float(mhd['ElementSpacing'].split()[0]),float(mhd['ElementSpacing'].split()[1]),float(mhd['ElementSpacing'].split()[2])] # физический размер вокселя в метрах
resolution=resolution[0] # пока что скрипт работает только с одинаковым разрешением по всем трем направлениям 

# старая версия кода - загрузка .raw в ручную:
# pathToRaw= 'S5/S5.raw' # путь к бинарному файлу с цифровой моделью керна
# dim_size = [300, 300, 300] # кол-во вокселей по направлению осей X, Y, Z
# resolution = 3.997e-6 # физический размер вокселя в метрах
# # Далее идет сама загрузка файла
# with open(pathToRaw, 'rb') as f:
#     im = np.fromfile(f, dtype=np.uint8)

im=np.clip(im,0,1) # ограничить значения вокселя нулем и еденицей, где 0 - поровое пространство, 1 - горная порода
# при загрузке бинарного файла .raw направоения по оси x и z поменяны местами im(zDirection,yDirection,xDirection)  
im = im.reshape(dim_size[2], dim_size[1], dim_size[0])
im = np.array(im, dtype=bool) # recast from uint8 to boolean
im = np.swapaxes(im, 0, 2) # оси меняются местами для совпадения с направлением осей файла .mhd в paraview

# При необходимости, можно обрезать цифровую модель (например для быстрого тестирования скрипта). Если обрезка не нужна, то нужно закомментировать следующую строку кода
# im = im[:100,:100,:100] # закомментировать эту строку, если обрезка модели не нужнa
dim_size=[np.size(im,0),np.size(im,1),np.size(im,2)] # не изменять эту строку, это программный код!
mhd['DimSize']=str(dim_size[0])+' '+str(dim_size[1])+' '+str(dim_size[2])
im = ~im # инвертирование True и False (False должно соответствовать горной породе) # не трогать эту строку! это программный код


###############################################################################
# ВЫБОР ВЫЧИСЛЕНИЙ ВЫБОР ВЫЧИСЛЕНИЙ ПОЛЬЗОВАТЕЛЕМ

# Визуализация цифровой модели керна в плоскости (X-Y)
visualization2D= True
iSlice=0 # выбрать номер слайса по оси Z. Стандартное значение 0, т.е. первый слайс

# Удаление изолированных пор из цифровой модели керна, 
# т.е. пор, не связанных ни с одной гранью цифровой модели образца.
remove_blind_pores = False # True или False. Стандартное значение: True
# Если remove_blind_pores=True, все последующие функции будут использовать измененную цифровую модель керна

# Выделение связного порового пространства цифровой модели керна, 
# т.е. только тех пор, которые связаны с противолежащими гранями.
# connectivity = '-'- не выделять связное поровое пространство
# connectivity = 'X','Y','Z'- связность по оси X, Y или Z
# connectivity = 'XorYorZ'- связность по (оси X) логическое И (по оси Y) логическое И (по оси Z)
# connectivity = 'XandYandZ'- связность по (оси X) логическое ИЛИ (по оси Y) логическое ИЛИ (по оси Z)
connectivity = ['-']
# Если функция connectivity исполняется, все последующие функции будут использовать измененную цифровую модель керна

# сохранение измененной модели керна:
saveNewModel = False
newModelName='newModel' # название новой модели керна


###############################################################################
# Plot options (font size  etc.)                            
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


###############################################################################
# декларация необходимых функций                          

# Функция стирает все нелулевые воксели, которые связаны с заданной гранью
# This function set to zero all non-zero elements, that are not connected to a chosen face.
def connectivityToOneFace(im,face,conn):
    #face = 0 -> first dimension lower bound, 3 -> first dimension upper bound, 
    #       1 -> second dimension lower bound, 4 -> second dimension upper bound, 
    #       1 -> third dimension lower bound, 5 -> third dimension upper bound, 
    # conn (int) – for the 3D the options are 6 and 26, similarily for face and edge neighbors.
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

#%%
###############################################################################
#Saving module
fileNameMHDSave1 = 'Berea.exp/Berea_test2.mhd'
fileNameMHDSave2 = 'Berea.exp/Berea_test2.raw'

def save_mhd(newModelName, s_im):
    mhdNew=mhd.copy()
    mhdNew['ElementDataFile']=os.path.basename(newModelName)+'.raw'
# после загрузки исходной модели керна, мы меняли направление осей с im(zDirection,yDirection,xDirection) на im(xDirection,yDirection,zDirection) 
# теперь нужно поменять направления осей обратно. Кроме того,нужно инвернировать поры и твердую фазу, чтобы было 0 - пора и 255 - твердая фаза
# Сохранение в бинарном формате uint8:
    #imToSave = np.ascontiguousarray(~np.swapaxes(s_im, 0, 2), dtype=np.uint8)
    imToSave = 255 - (np.swapaxes(255 * rescale_intensity(s_im, (0, 1)), 0, 2)).astype('uint8')
    
    print(np.max(imToSave))
    print(np.min(imToSave))
    
    dra.write_raw_data_with_mhd(newModelName+'.mhd',mhdNew,newModelName+'.raw',imToSave)
    del imToSave

#%%
#dra.swapDirections('random_walker_bc (1).mhd','S5.mhd','XZ')

def printsl(mdl):
    plt.imshow(np.swapaxes(mdl[:,:,iSlice], 0, 1), cmap = 'gray')
#%%
###############################################################################
#s014_FBP

def make_even_int(num):
    return int(num) + int(num) % 2

slice_amount = dim_size[2]

rec_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])
new_width = make_even_int(np.sqrt(dim_size[0] ** 2 + dim_size[1] ** 2) * 1.2)

start_time = datetime.now()
for cur_slice in range(0, slice_amount):
    sl = im[:,:,cur_slice] #new_model[:,:,iSlice]
    
    vol_geom = astra.create_vol_geom(dim_size[0], dim_size[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, new_width, np.linspace(0,np.pi,2400,False))

    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

    sinogram_id, sinogram = astra.create_sino(sl, proj_id)

    #pylab.imshow(sl)
    #pylab.imshow(sinogram)

    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = { 'FilterType': 'Ram-Lak' }

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    rec = astra.data2d.get(rec_id)
    
    rec_model[:,:,cur_slice] = rec
    
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

print("KT simulation woked:", datetime.now() - start_time)
#At this moment colors in rec image are inverted so 0 is solid and 1 is pores
#%%
plt.imshow(np.swapaxes(~im[:,:,iSlice], 0, 1), cmap = 'gray')
#plt.imshow(np.swapaxes(np.max(rec_model[:,:,iSlice]) - rec_model[:,:,iSlice], 0, 1), cmap = 'gray')
save_mhd("kt_s247", rec_model)
###############################################################################
#%% Contrast stretching

p2, p98 = np.percentile(rec_model, (0.5, 99.5))
r_rec_model = rescale_intensity(rec_model, in_range=(p2, p98))

#plt.hist(r_rec_model[:,:,iSlice].ravel(), bins = 256)
plt.imshow(r_rec_model[:,:,iSlice], cmap = 'gray')
save_mhd("contrast_stretching_sk247", rec_model)
#Filtration block
###############################################################################
#%% Add noise
sigma = 0.08


#Add noise: salt and pepper noise
SaP_model = random_noise(r_rec_model, mode='S&P', amount = 0.1)


#plt.imshow(SaP_model[:,:,iSlice])

#Add Gausssian blur

sigma = 2.0

blured_model = skimage.filters.gaussian(
    SaP_model, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

#random_noise(SaP_model, mode='gaussian')
#plt.imshow(blured_model[:,:,iSlice], cmap = 'gray')

save_mhd("noise+kt_sk247", rec_model)
plt.imshow(np.swapaxes(np.max(blured_model[:,:,iSlice]) - blured_model[:,:,iSlice], 0, 1), cmap = 'gray')
###############################################################################
#NLM in 3D
#%%
#sigma = 0.08
#filtrationsl = random_noise(sl, var=sigma**2)

#stretching это как раз то, что я делаю в рандом фолкере. Тут он нужен для какого выравнивани?
#выравнивать по 0 - 1

filtred_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()
for cur_slice in range(0, slice_amount):
    filtrationsl = blured_model[:,:,cur_slice].copy(order = 'C') # r_rec_model#random_noise(sl, var=sigma**2)
    sigma_est = np.mean(estimate_sigma(filtrationsl))
    #print(f'estimated noise standard deviation = {sigma_est}')

    patch_kw = dict(patch_size=7,      # 5x5 patches
                    patch_distance=11)  # 13x13 search area

    filtred_model[:,:,cur_slice] = denoise_nl_means(filtrationsl, h=0.6 * sigma_est, sigma=sigma_est,
                                     fast_mode=True, **patch_kw)
    
print("NLM filtration woked:", datetime.now() - start_time)
plt.imshow(filtred_model[:,:,iSlice])
#%%
save_mhd("flt_no_noise_sk247", filtred_model)
plt.imshow(np.swapaxes(np.max(filtred_model[:,:,iSlice]) - filtred_model[:,:,iSlice], 0, 1), cmap = 'gray')
###############################################################################
#Segmentation block
# offset optimize block

#%% all offset optimize

opt_diff = -1
opt_diff_i = -1
opt_porosity_dif = 1000
opt_porosity_i = -1
S = dim_size[0] * dim_size[1] * dim_size[2]

im_tmp = im

e_porosity = ps.metrics.porosity(im_tmp)
print(e_porosity)

imi1 = 255 - rescale_intensity(filtred_model, (0, 1)) * 255
imi1 = imi1.astype(np.uint8)

imi_res = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

for i in range(1, 51):
    cur_offset = i / 100
    targetPorosity = 0.15
    
    #plt.imshow(imi1[:,:,iSlice])
    for cur in range(0, dim_size[2]):
        imi = imi1[cur]
        
        counts, bins = np.histogram(imi,bins=255)
        Tpores=np.interp((1-cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        Tgrains=np.interp((1+cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        
        counts, bins = np.histogram(imi, bins=255)

        markers = np.zeros_like(imi)
        markers[imi <= Tpores] = 1
        markers[imi >= Tgrains] = 2

        edge_scharr = filters.scharr(imi)
        labels = watershed(edge_scharr,markers)

        segmentedImage=np.zeros_like(imi,dtype=np.uint8)
        segmentedImage[labels==2] = 1
        imi_res[cur] = 1 - segmentedImage
        
    cur_diff = (S - np.sum(np.abs(im_tmp - imi_res))) / S
    if cur_diff > opt_diff:
        opt_diff = cur_diff
        opt_diff_i = i
    cur_porosity_dif = e_porosity - ps.metrics.porosity(imi_res)
    if cur_porosity_dif < opt_porosity_dif:
        opt_porosity_dif = cur_porosity_dif
        opt_porosity_i = i
    print('Start iteration with i ' + str(i) + ' now opt persentage is(diif, porosity): (' + 
          str(opt_diff_i) + ', ' + str(opt_porosity_i) + ')')
    print('Opt porosity is: ' + str(opt_porosity_dif))
    print('Cur porosity is: ' + str(cur_porosity_dif))
    print('Opt diff is: ' + str(opt_diff))
    print('Cur diff is: ' + str(cur_diff))
    print('Offset is: ' + str(cur_offset))
    print('')
    

print(opt_diff_i)
print(opt_diff)
print(opt_porosity_i)
print(opt_porosity_dif)

w_offset = opt_diff_i / 100


print('')
print('')
print('')
print('')
print('')
print('RW')
print('')
print('')
print('')

opt_diff = -1
opt_diff_i = -1
opt_porosity_dif = 1000
opt_porosity_i = -1
S = dim_size[0] * dim_size[1] * dim_size[2]

imi_res = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

for i in range(1, 51):
    cur_offset = i / 100
    targetPorosity = 0.15
    
    #plt.imshow(imi1[:,:,iSlice])
    for cur in range(0, dim_size[2]):
        imi = imi1[cur]
        counts, bins = np.histogram(imi,bins=255)
        targetPorosity = 0.15

        Tpores=np.interp((1-cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        Tgrains=np.interp((1+cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)

        markers=np.zeros_like(imi)
        markers[imi <= Tpores] = 1
        markers[imi >= Tgrains] = 2
        
        random_walker_slice = random_walker(imi, markers, beta=10, mode='bf')
        imi_res[cur] = random_walker_slice
    
    imi_res = np.max(imi_res) - imi_res
        
        
    cur_diff = (S - np.sum(np.abs(im_tmp - imi_res))) / S
    if cur_diff > opt_diff:
        opt_diff = cur_diff
        opt_diff_i = i
    cur_porosity_dif = e_porosity - ps.metrics.porosity(imi_res)
    if cur_porosity_dif < opt_porosity_dif:
        opt_porosity_dif = cur_porosity_dif
        opt_porosity_i = i
    print('Start iteration with i ' + str(i) + ' now opt persentage is(diif, porosity): (' + 
          str(opt_diff_i) + ', ' + str(opt_porosity_i) + ')')
    print('Opt porosity is: ' + str(opt_porosity_dif))
    print('Cur porosity is: ' + str(cur_porosity_dif))
    print('Opt diff is: ' + str(opt_diff))
    print('Cur diff is: ' + str(cur_diff))
    print('Offset is: ' + str(cur_offset))
    print('')





r_offset = opt_diff_i / 100
print("now w_offset is: " + str(w_offset))
print("now r_offset is: " + str(r_offset))
#%%

r_offset = 0.01
w_offset = 0.01

#%%
opt_diff = -1
opt_diff_i = -1
opt_porosity_dif = 1000
opt_porosity_i = -1
S = dim_size[0] * dim_size[1] * 5

im_tmp = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])
im_tmp[0] = im[:,:,iSlice]
im_tmp[1] = im[:,:,37]
im_tmp[2] = im[:,:,84]
im_tmp[3] = im[:,:,148]
im_tmp[4] = im[:,:,251]


e_porosity = ps.metrics.porosity(im_tmp)

imi1 = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])
imi1[0] = 255 - rescale_intensity(filtred_model[:,:,iSlice], (0, 1)) * 255
imi1[1] = 255 - rescale_intensity(filtred_model[:,:,37], (0, 1)) * 255
imi1[2] = 255 - rescale_intensity(filtred_model[:,:,84], (0, 1)) * 255
imi1[3] = 255 - rescale_intensity(filtred_model[:,:,148], (0, 1)) * 255
imi1[4] = 255 - rescale_intensity(filtred_model[:,:,251], (0, 1)) * 255
#imi1 = imi1.astype(np.uint8)

plt.imshow(imi1[0])

imi_res = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])

for i in range(1, 51):
    cur_offset = i / 100
    targetPorosity = 0.15
    
    for cur in range(0, 5):
        imi = imi1[cur]
        counts, bins = np.histogram(imi,bins=255)
        Tpores=np.interp((1-cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        Tgrains=np.interp((1+cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        
        counts, bins = np.histogram(imi, bins=255)

        markers = np.zeros_like(imi)
        markers[imi <= Tpores] = 1
        markers[imi >= Tgrains] = 2

        edge_scharr = filters.scharr(imi)
        labels = watershed(edge_scharr,markers)

        segmentedImage=np.zeros_like(imi,dtype=np.uint8)
        segmentedImage[labels==2] = 1
        imi_res[cur] = 1 - segmentedImage
    cur_diff = (S - np.sum(np.abs(im_tmp - imi_res))) / S
    if cur_diff > opt_diff:
        opt_diff = cur_diff
        opt_diff_i = i
    cur_porosity_dif = e_porosity - ps.metrics.porosity(imi_res)
    if cur_porosity_dif < opt_porosity_dif:
        opt_porosity_dif = cur_porosity_dif
        opt_porosity_i = i


print(opt_diff_i)
print(opt_diff)
print(opt_porosity_i)
print(opt_porosity_dif)

offset = opt_diff_i / 100
print("now offset is: " + str(offset))
#%% random walker offset optimize
opt_diff = -1
opt_diff_i = -1
opt_porosity_dif = 1000
opt_porosity_i = -1
S = dim_size[0] * dim_size[1] * 5

im_tmp = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])
im_tmp[0] = im[:,:,iSlice]
im_tmp[1] = im[:,:,37]
im_tmp[2] = im[:,:,84]
im_tmp[3] = im[:,:,148]
im_tmp[4] = im[:,:,251]

e_porosity = ps.metrics.porosity(im_tmp)

imi1 = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])
imi1[0] = 255 - rescale_intensity(filtred_model[:,:,iSlice], (0, 1)) * 255
imi1[1] = 255 - rescale_intensity(filtred_model[:,:,37], (0, 1)) * 255
imi1[2] = 255 - rescale_intensity(filtred_model[:,:,84], (0, 1)) * 255
imi1[3] = 255 - rescale_intensity(filtred_model[:,:,148], (0, 1)) * 255
imi1[4] = 255 - rescale_intensity(filtred_model[:,:,251], (0, 1)) * 255
imi1 = imi1.astype(np.uint8)

imi_res = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(5)])

for i in range(1, 51):
    cur_offset = i / 100
    targetPorosity = 0.15
    
    for cur in range(0, 5):
        imi = imi1[cur]
        counts, bins = np.histogram(imi,bins=255)
        targetPorosity = 0.15

        Tpores=np.interp((1-cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
        Tgrains=np.interp((1+cur_offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)

        markers=np.zeros_like(imi)
        markers[imi <= Tpores] = 1
        markers[imi >= Tgrains] = 2
        
        random_walker_slice = random_walker(imi, markers, beta=10, mode='bf')
    
    imi_res = np.max(imi_res) - imi_res
    
    cur_diff = (S - np.sum(np.abs(im_tmp - imi_res))) / S
    if cur_diff > opt_diff:
        opt_diff = cur_diff
        opt_diff_i = i
    cur_porosity_dif = e_porosity - ps.metrics.porosity(imi_res)
    if cur_porosity_dif < opt_porosity_dif:
        opt_porosity_dif = cur_porosity_dif
        opt_porosity_i = i

print(opt_diff_i)
print(opt_diff)
print(opt_porosity_i)
print(opt_porosity_dif)

offset = opt_diff_i / 100
print("now offset is: " + str(offset))
###############################################################################
#Otsu
#%%

otsu_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()

for cur_slice in range(0, slice_amount):
    image = filtred_model[:,:,cur_slice]
    thresh = threshold_otsu(image)
    otsu_model[:,:,cur_slice] = image > thresh

    """
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 2)
    ax[1] = plt.subplot(1, 3, 3)

    ax[0].hist(image.ravel(), bins=256)
    ax[0].set_yscale('log')
    ax[0].set_title('Histogram')
    ax[0].axvline(thresh, color='r')

    ax[1].imshow(binary, cmap=plt.cm.gray)
    ax[1].set_title('Otsu segmentation')
    ax[1].axis('off')

    plt.show()
    """
    
print("Otsu segmentation woked:", datetime.now() - start_time)
plt.imshow(otsu_model[:,:,iSlice])
print(ps.metrics.porosity(otsu_model))
print(ps.metrics.porosity(im))
#%%
#save_mhd("otsu_bc", otsu_model)
#dra.swapDirections('otsu_bc.mhd','S5.mhd','XZ')
plt.imshow(np.swapaxes(np.max(otsu_model[:,:,iSlice]) - otsu_model[:,:,iSlice], 0, 1), cmap = 'gray')
"""
 Absolute permeability: 6.13551e-13 (m2), 621.681 (mD) 
 Formation factor:      25.9688
 
 ODR
 Out[86]: 0.732494
"""
###############################################################################
#Local Otsu
#%%

radius = 40

local_otsu_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()

for cur_slice in range(0, slice_amount):
    img = img_as_ubyte(rescale_intensity(filtred_model[:,:,cur_slice], (0, 1)))
    local_otsu = rank.otsu(img, disk(radius))
    #plt.imshow(img >= local_otsu, cmap=plt.cm.gray)
    local_otsu_model[:,:,cur_slice] = (img >= local_otsu)

print("Local Otsu segmentation woked:", datetime.now() - start_time)
plt.imshow(local_otsu_model[:,:,iSlice])
#%%

#save_mhd("local_otsu_bc", local_otsu_model)
#dra.swapDirections('local_otsu_bc.mhd','S5.mhd','XZ')
plt.imshow(np.swapaxes(np.max(local_otsu_model[:,:,iSlice]) - local_otsu_model[:,:,iSlice], 0, 1), cmap = 'gray')
"""
 Absolute permeability: 4.99661e-14 (m2), 50.6281 (mD) 
 Formation factor:      116.367
 
 ODR
 Out[93]: 0.515667
"""
###############################################################################
#Watershed
#%%
global_labels = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

m_min = np.min(filtred_model[:,:,iSlice])
m_max = np.max(filtred_model[:,:,iSlice])

offset=0.50#w_offset
targetPorosity = 0.15

#imi = 255 - (filtred_model[:,:,iSlice] - m_min) / (m_max - m_min) * 255

imi = 255 - rescale_intensity(filtred_model[:,:,iSlice], (0, 1)) * 255
imi = imi.astype(np.uint8)

counts, bins = np.histogram(imi,bins=255)

#plt.hist(imi.ravel(), bins=256)

Tpores=np.interp((1-offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
Tgrains=np.interp((1+offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)

counts, bins = np.histogram(imi, bins=255)

print(counts)
print(bins)

print(Tpores)
print(Tgrains)

markers = np.zeros_like(imi)
markers[imi <= Tpores] = 1
markers[imi >= Tgrains] = 2

edge_scharr = filters.scharr(imi)
labels = watershed(edge_scharr,markers)

global_labels[:,:,iSlice] = labels
 
segmentedImage=np.zeros_like(imi,dtype=np.uint8)
segmentedImage[labels==2]=255

plt.imshow(segmentedImage)

#%%
watershed_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])
global_labels = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()

m_tmp = rescale_intensity(filtred_model, (0, 1)) * 255
m_tmp = m_tmp.astype(np.uint8)

targetPorosity = ps.metrics.porosity(otsu_model)
print(targetPorosity)

for cur_slice in range(0, slice_amount):
    imagew = m_tmp[:,:,cur_slice]

    offset=0.50
    counts, bins = np.histogram(imagew,bins=255)
    #targetPorosity = ps.metrics.porosity(otsu_model)

    Tpores=np.interp((1-offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
    Tgrains=np.interp((1+offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)

    markers=np.zeros_like(imagew)
    markers[imagew <= Tpores] = 1
    markers[imagew >= Tgrains] = 2

    edge_scharr = filters.scharr(imagew)
    labels = watershed(edge_scharr,markers)

    global_labels[:,:,cur_slice] = labels
    
    segmentedImage=np.zeros_like(imagew,dtype=np.uint8)
    segmentedImage[labels==2] = 1
    
    watershed_model[:,:,cur_slice] = segmentedImage

watershed_model = watershed_model
print("Watershed segmentation woked:", datetime.now() - start_time)
plt.imshow(watershed_model[:,:,iSlice])
#plt.hist(segmentedImage.ravel(), bins=256)
#%%
#save_mhd("watershed_bc", watershed_model)
#dra.swapDirections('watershed_bc.mhd','S5.mhd','XZ')
#plt.imshow(watershed_model[:,:,iSlice])
plt.imshow(np.swapaxes(np.max(watershed_model[:,:,iSlice]) - watershed_model[:,:,iSlice], 0, 1), cmap = 'gray')

#plt.hist(watershed_model[:,:,iSlice].ravel(), bins=256)

"""
 Absolute permeability: 7.78096e-15 (m2), 7.88406 (mD) 
 Formation factor:      384.361
 
 ODR
 Out[96]: 0.38042200000000004
"""
###############################################################################
#Random walker
#%%

random_walker_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()

targetPorosity = ps.metrics.porosity(otsu_model)
print(targetPorosity)

for cur_slice in range(0, slice_amount):
    dat = 255 - m_tmp[:,:,cur_slice]
    """
    markers = np.zeros(dat.shape, dtype=np.uint)

    deviation = 0.95
    #По какому принципу мы хотим выбирать маркеры?
    markers[dat < -deviation] = 1
    markers[dat > deviation] = 2
    """
    offset=0.50
    counts, bins = np.histogram(dat,bins=255)
    #targetPorosity = ps.metrics.porosity(otsu_model)

    Tpores=np.interp((1-offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)
    Tgrains=np.interp((1+offset)*targetPorosity,np.cumsum(counts/np.sum(counts)),bins[0:-1]+np.diff(bins)/2)

    markers=np.zeros_like(dat)
    markers[dat <= Tpores] = 1
    markers[dat >= Tgrains] = 2
    
    
    random_walker_model[:,:,cur_slice] = random_walker(dat, markers, beta=10, mode='bf')
    """
    fig, axes = plt.subplots(ncols=2, figsize=(8, 3.2),sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(markers)
    ax[0].set_title('Markers')
    ax[0].axis('off')

    ax[1].imshow(labels, cmap='gray')
    ax[1].set_title('Random walker segmentation')
    ax[1].axis('off')

    plt.show()
    """
    
random_walker_model = np.max(random_walker_model) - random_walker_model
print("Random walker segmentation woked:", datetime.now() - start_time) 
plt.imshow(random_walker_model[:,:,iSlice])
#%%
#save_mhd("random_walker_bn", random_walker_model)
#dra.swapDirections('random_walker_bn.mhd','S5.mhd','XZ')
print(np.max(random_walker_model[:,:,iSlice]))
#plt.imshow(random_walker_model[:,:,iSlice])
plt.imshow(np.swapaxes(np.max(random_walker_model[:,:,iSlice]) - random_walker_model[:,:,iSlice], 0, 1), cmap = 'gray')

#plt.hist(random_walker_model[:,:,iSlice].ravel(), bins = 256)

"""
 Absolute permeability: 5.29168e-15 (m2), 5.3618 (mD) 
 Formation factor:      553.392
 
 ODR
 Out[99]: 0.35062099999999996
"""
#%%

newModelName = "test_random_walker"

mhdNew=mhd.copy()
mhdNew['ElementDataFile']=os.path.basename(newModelName)+'.raw'
# после загрузки исходной модели керна, мы меняли направление осей с im(zDirection,yDirection,xDirection) на im(xDirection,yDirection,zDirection) 
# теперь нужно поменять направления осей обратно. Кроме того,нужно инвернировать поры и твердую фазу, чтобы было 0 - пора и 255 - твердая фаза
# Сохранение в бинарном формате uint8:
#imToSave = np.ascontiguousarray(~np.swapaxes(s_im, 0, 2), dtype=np.uint8)
imToSave = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])
imToSave = imToSave.astype('uint8')

for cur_slice in range(0, slice_amount):
    imToSave[:,:,cur_slice] = 255 * random_walker_model[:,:,cur_slice] / np.max(random_walker_model[:,:,cur_slice])
       
imToSave = (np.swapaxes(imToSave, 0, 2)).astype('uint8')

dra.write_raw_data_with_mhd(newModelName+'.mhd',mhdNew,newModelName+'.raw',imToSave)
del imToSave

#save_mhd("test_random_walker", random_walker_model)
###############################################################################
#Indicator kriging
#%%
#save_mhd("im", im)
dra.swapDirections('random_walker_bn.mhd','S5.mhd','XZ')

#%%
###############################################################################
# Computations  

printsl(np.max(r_rec_model[:,:,iSlice]) - r_rec_model)
#%%
###############################################################################
#Accurancy as norm of rejection vector

V = dim_size[0] * dim_size[1] * dim_size[2]

print('Результат сегментации Otsu имеет точность относительного исходного изображения: ' 
      + str((V - np.sum(np.abs(im - otsu_model))) / V))
print('Результат сегментации local Otsu имеет точность относительного исходного изображения: ' 
      + str((V - np.sum(np.abs(im - local_otsu_model))) / V))
print('Результат сегментации Watershed имеет точность относительного исходного изображения: ' 
      + str((V - np.sum(np.abs(im - watershed_model))) / V))
print('Результат сегментации Random Walker имеет точность относительного исходного изображения: ' 
      + str((V - np.sum(np.abs(im - random_walker_model))) / V))

#%%
###############################################################################
#Total porosity

print('Общая пористость (с учетом закрытых пор) исходная: ' + str(ps.metrics.porosity(im)))
print('Общая пористость (с учетом закрытых пор) Otsu: ' + str(ps.metrics.porosity(otsu_model)))
print('Общая пористость (с учетом закрытых пор) local Otsu: ' + str(ps.metrics.porosity(local_otsu_model)))
print('Общая пористость (с учетом закрытых пор) Watershed: ' + str(ps.metrics.porosity(watershed_model)))
print('Общая пористость (с учетом закрытых пор) Random Walker: ' + str(ps.metrics.porosity(random_walker_model)))

#%%

def count_square(in_image, msg): 
#    inv_im = ~im
#    tmp = ps.tools.mesh_region(inv_im)
#    del(inv_im)
#    vertices = tmp.verts
#    faces = tmp.faces
#    del(tmp)
#    mesh_t = trimesh.Trimesh(vertices=vertices, faces=faces)
#    del(vertices)
#    del(faces)
#    S = mesh_t.area - mesh_t.convex_hull.area
#    del(mesh_t)
    in_image = in_image.astype('uint8')
    V = dim_size[0] * dim_size[1] * dim_size[2]*resolution**3
    voidArea=ps.metrics.region_surface_areas(in_image,voxel_size=resolution)
    solidArea=ps.metrics.region_surface_areas( ~in_image,voxel_size=resolution)
    S=0.5*(solidArea+voidArea-2*resolution**2*(dim_size[0]*dim_size[1]+dim_size[2]*dim_size[1]+dim_size[0]*dim_size[2])) # without area at the faces of the sample 
    print('Отношение площади поверхности к объёму для ' + msg + ': ', S / V, ' м^-1') 

count_square(otsu_model, 'Otsu')
count_square(local_otsu_model, 'local Otsu')
count_square(watershed_model / np.max(watershed_model[:,:,iSlice]), 'Watershed')
count_square(random_walker_model / np.max(random_walker_model[:,:,iSlice]), 'Random Walker')

#%%
###############################################################################
# общая пористость                          
porosity = ps.metrics.porosity(im)
#porosity = round(ps.metrics.porosity(im), 4)
print('Общая пористость (с учетом закрытых пор): ' + str(porosity))


# удаление несвязных пор 
if remove_blind_pores:
    im = ps.filters.fill_blind_pores(im)
    porosityBlind = ps.metrics.porosity(im)
    print('Пористость без учета изолированных пор: ' + str(porosityBlind))


#Выделение связного порового пространства цифровой модели керна по осям X, Y, Z:
if (connectivity == ['X']):
    imTMP1=connectivityToOneFace(im,0,6)
    imTMP2=connectivityToOneFace(im,0+3,6)
    im=np.logical_and(imTMP1,imTMP2)
    del imTMP1, imTMP2
    porosityConnected = ps.metrics.porosity(im)
    print('Связная пористость по оси X: ' + str(porosityConnected))    
if (connectivity == ['Y']):
    imTMP1=connectivityToOneFace(im,1,6)
    imTMP2=connectivityToOneFace(im,1+3,6)
    im=np.logical_and(imTMP1,imTMP2)
    del imTMP1, imTMP2
    porosityConnected = ps.metrics.porosity(im)
    print('Связная пористость по оси Y: ' + str(porosityConnected)) 
if (connectivity == ['Z']):
    imTMP1=connectivityToOneFace(im,2,6)
    imTMP2=connectivityToOneFace(im,2+3,6)
    im=np.logical_and(imTMP1,imTMP2)
    del imTMP1, imTMP2
    porosityConnected = ps.metrics.porosity(im)
    print('Связная пористость по оси Z: ' + str(porosityConnected)) 
if (connectivity == ['XorYorZ']):
    imTMP1=connectivityToOneFace(im,0,6)
    imTMP2=connectivityToOneFace(im,0+3,6)
    imX=np.logical_and(imTMP1,imTMP2)
    imTMP1=connectivityToOneFace(im,1,6)
    imTMP2=connectivityToOneFace(im,1+3,6)
    imY=np.logical_and(imTMP1,imTMP2)
    imTMP1=connectivityToOneFace(im,2,6)
    imTMP2=connectivityToOneFace(im,2+3,6)
    imZ=np.logical_and(imTMP1,imTMP2)
    im=np.logical_or(imX,imY)
    im=np.logical_or(im,imZ)
    del imTMP1, imTMP2, imX, imY, imZ
    porosityConnected = ps.metrics.porosity(im)
    print('Связная пористость по операции ось X OR ось Y OR ось Z: ' + str(porosityConnected)) 
if (connectivity == ['XandYandZ']):
    imTMP1=connectivityToOneFace(im,0,6)
    imTMP2=connectivityToOneFace(im,0+3,6)
    imX=np.logical_and(imTMP1,imTMP2)
    imTMP1=connectivityToOneFace(im,1,6)
    imTMP2=connectivityToOneFace(im,1+3,6)
    imY=np.logical_and(imTMP1,imTMP2)
    imTMP1=connectivityToOneFace(im,2,6)
    imTMP2=connectivityToOneFace(im,2+3,6)
    imZ=np.logical_and(imTMP1,imTMP2)
    im=np.logical_and(imX,imY)
    im=np.logical_and(im,imZ)    
    del imTMP1, imTMP2, imX, imY, imZ
    porosityConnected = ps.metrics.porosity(im)
    print('Связная пористость по операции ось X AND ось Y AND ось Z: ' + str(porosityConnected)) 


# Визуализация цифровой модели керна в плоскости (X-Y)
if visualization2D:
    plt.figure()
    plt.imshow(np.swapaxes(im[:,:,iSlice],0,1),cmap='binary')
    plt.xlabel('ось X')
    plt.ylabel('ось Y')
    plt.colorbar() 
    plt.title('Плоскость X-Y, iSlice='+str(iSlice)+'\n Поры 1 (черный), Твердая фаза 0 (белый)')


# сохранение измененной модели керна:
if saveNewModel:
    mhdNew=mhd.copy()
    mhdNew['ElementDataFile']=os.path.basename(newModelName)+'.raw'
    # после загрузки исходной модели керна, мы меняли направление осей с im(zDirection,yDirection,xDirection) на im(xDirection,yDirection,zDirection) 
    # теперь нужно поменять направления осей обратно. Кроме того,нужно инвернировать поры и твердую фазу, чтобы было 0 - пора и 255 - твердая фаза
    # Сохранение в бинарном формате uint8:
    imToSave = 255*np.ascontiguousarray(~np.swapaxes(im, 0, 2), dtype=np.uint8)
    dra.write_raw_data_with_mhd(newModelName+'.mhd',mhdNew,newModelName+'.raw',imToSave)
    del imToSave
    