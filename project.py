#%% Инициализация
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Импорт библиотек
import porespy as ps
print('porespy version '+ps.__version__) 
import matplotlib.pyplot as plt
import scipy as sp
print('scipy version '+sp.__version__) 
import numpy as np
print('numpy version '+np.__version__) 
import openpnm as op 
print('openpnm version '+op.__version__) 
import trimesh
print('trimesh version '+trimesh.__version__) 
import DRA_utils_v4 as dra
import os 
import astra
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from datetime import datetime
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import filters
import skimage

#%% Загрузка цифровой модели керна
###############################################################################
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
resolution=resolution[0]

im=np.clip(im,0,1) # Ограничить значения вокселя нулем и еденицей, 
# где 0 - поровое пространство, 1 - горная порода
# При загрузке бинарного файла .raw направоения по оси x и z поменяны 
# местами im(zDirection,yDirection,xDirection)  
im = im.reshape(dim_size[2], dim_size[1], dim_size[0])
im = np.array(im, dtype=bool) # recast from uint8 to boolean
im = np.swapaxes(im, 0, 2) # оси меняются местами

dim_size=[np.size(im,0),np.size(im,1),np.size(im,2)]
mhd['DimSize']=str(dim_size[0])+' '+str(dim_size[1])+' '+str(dim_size[2])
im = ~im # инвертирование True и False (False должно соответствовать горной породе)

###############################################################################
# Настройки plt                        
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
# Параметры запуска
iSlice=0
visualization2D= True
remove_blind_pores = False
connectivity = ['-']
###############################################################################
# Декларация необходимых функций                          

# Функция стирает все нелулевые воксели, которые связаны с заданной гранью
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

# Функция сохранения модели
def save_mhd(newModelName, s_im):
    mhdNew=mhd.copy()
    mhdNew['ElementDataFile']=os.path.basename(newModelName)+'.raw'
# после загрузки исходной модели керна, мы меняли направление осей с 
# im(zDirection,yDirection,xDirection) на im(xDirection,yDirection,zDirection) 
# теперь нужно поменять направления осей обратно. Кроме того,нужно 
# инвернировать поры и твердую фазу, чтобы было 0 - пора и 255 - твердая фаза
# Сохранение в бинарном формате uint8:
    #imToSave = np.ascontiguousarray(~np.swapaxes(s_im, 0, 2), dtype=np.uint8)
    imToSave = 255 - (np.swapaxes(255 * rescale_intensity(s_im, (0, 1)), 0, 2)).astype('uint8')
    
    print(np.max(imToSave))
    print(np.min(imToSave))
    
    dra.write_raw_data_with_mhd(newModelName+'.mhd',mhdNew,newModelName+'.raw',imToSave)
    del imToSave
    
#%% КТ симуляция
###############################################################################
#s014_FBP

def make_even_int(num):
    return int(num) + int(num) % 2

slice_amount = dim_size[2]

rec_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])
new_width = make_even_int(np.sqrt(dim_size[0] ** 2 + dim_size[1] ** 2) * 1.2)

start_time = datetime.now()
for cur_slice in range(0, slice_amount):
    sl = im[:,:,cur_slice]
    
    vol_geom = astra.create_vol_geom(dim_size[0], dim_size[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, new_width, np.linspace(0,np.pi,2400,False))

    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

    sinogram_id, sinogram = astra.create_sino(sl, proj_id)

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
#Тут значения пор и пород поменялись, теперь пора - это 1, порода - это 0.

#%% Сохранение результата КТ
plt.imshow(np.swapaxes(~im[:,:,iSlice], 0, 1), cmap = 'gray')
save_mhd("kt_result", rec_model)

#%% Выравнивание гистограммы

pb, pt = np.percentile(rec_model, (0.5, 99.5))
r_rec_model = rescale_intensity(rec_model, in_range=(pb, pt))

#plt.hist(r_rec_model[:,:,iSlice].ravel(), bins = 256)

#%% Сохранение результата выравнивания гистограммы
plt.imshow(r_rec_model[:,:,iSlice], cmap = 'gray')
save_mhd("contrast_stretching_result", rec_model)

#%% Добавление шумов

#Добавление фильтра S&P (Salt and Pepper noise)
SaP_model = random_noise(r_rec_model, mode ='S&P', amount = 0.1)

#plt.imshow(SaP_model[:,:,iSlice])

#Добавление Гауссовского блюра

sigma = 2.0

blured_model = skimage.filters.gaussian(
    SaP_model, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

#%% Сохранение результата КТ с добавленными шумами
save_mhd("noise+kt_result", rec_model)
plt.imshow(np.swapaxes(np.max(blured_model[:,:,iSlice]) - 
                       blured_model[:,:,iSlice], 0, 1), cmap = 'gray')

#%% Фильтрация с помощью NLM (non-local-means)
###############################################################################

filtred_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()
for cur_slice in range(0, slice_amount):
    filtrationsl = blured_model[:,:,cur_slice].copy(order = 'C') # r_rec_model
    sigma_est = np.mean(estimate_sigma(filtrationsl))
   
    patch_kw = dict(patch_size=7,      #5
                    patch_distance=11) #13

    filtred_model[:,:,cur_slice] = denoise_nl_means(filtrationsl, 
                        h = 0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    
print("NLM filtration woked:", datetime.now() - start_time)

#%% Сохранение результата фильтрации
save_mhd("flt_result", filtred_model)
plt.imshow(np.swapaxes(np.max(filtred_model[:,:,iSlice]) - 
                       filtred_model[:,:,iSlice], 0, 1), cmap = 'gray')

#%% Сегментация Otsu
###############################################################################

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
#%% Сохранение результата сегментации Otsu

save_mhd("otsu_result", otsu_model)
plt.imshow(np.swapaxes(np.max(otsu_model[:,:,iSlice]) - 
                       otsu_model[:,:,iSlice], 0, 1), cmap = 'gray')

#%% Подготовка результата сегментации Otsu для pnflow

dra.swapDirections('otsu_result.mhd','S5.mhd','XZ')

#%% Сегментация local Otsu

radius = 40

local_otsu_model = np.array([np.zeros((dim_size[0], dim_size[1])) for i in range(dim_size[2])])

start_time = datetime.now()

for cur_slice in range(0, slice_amount):
    img = img_as_ubyte(rescale_intensity(filtred_model[:,:,cur_slice], (0, 1)))
    local_otsu = rank.otsu(img, disk(radius))
    #plt.imshow(img >= local_otsu, cmap=plt.cm.gray)
    local_otsu_model[:,:,cur_slice] = (img >= local_otsu)

print("Local Otsu segmentation woked:", datetime.now() - start_time)

#%% Сохранение результата сегментации local Otsu

save_mhd("local_otsu_result", local_otsu_model)
plt.imshow(np.swapaxes(np.max(local_otsu_model[:,:,iSlice]) - 
                       local_otsu_model[:,:,iSlice], 0, 1), cmap = 'gray')

#%% Подготовка результата сегментации local Otsu для pnflow

dra.swapDirections('local_otsu_result.mhd','S5.mhd','XZ')

#%% Сегментация Watershed
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

print("Watershed segmentation woked:", datetime.now() - start_time)

#%% Сохранение результата сегментации Watershed
save_mhd("watershed_result", watershed_model)
plt.imshow(np.swapaxes(np.max(watershed_model[:,:,iSlice]) - 
                       watershed_model[:,:,iSlice], 0, 1), cmap = 'gray')

#%% Подготовка результата сегментации Watershed для pnflow

dra.swapDirections('watershed_result.mhd','S5.mhd','XZ')

#%% Сегментация Random Walker

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

#%% Сохранение результата сегментации Random Walker

save_mhd("random_walker_result", random_walker_model)
print(np.max(random_walker_model[:,:,iSlice]))
plt.imshow(np.swapaxes(np.max(random_walker_model[:,:,iSlice]) - 
                       random_walker_model[:,:,iSlice], 0, 1), cmap = 'gray')
#%% Подготовка результата сегментации Watershed для pnflow

dra.swapDirections('random_walker_result.mhd','S5.mhd','XZ')

#%% Вычисления
###############################################################################
# Точность топологии

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
# Общая пористость

print('Общая пористость (с учетом закрытых пор) исходная: ' + str(ps.metrics.porosity(im)))
print('Общая пористость (с учетом закрытых пор) Otsu: ' + str(ps.metrics.porosity(otsu_model)))
print('Общая пористость (с учетом закрытых пор) local Otsu: ' + str(ps.metrics.porosity(local_otsu_model)))
print('Общая пористость (с учетом закрытых пор) Watershed: ' + str(ps.metrics.porosity(watershed_model)))
print('Общая пористость (с учетом закрытых пор) Random Walker: ' + str(ps.metrics.porosity(random_walker_model)))

# Остальные вычисления делаются с помощью pnflow

#%% Дополнительные функции для рассчётов
###############################################################################

def count_square(in_image, msg): 
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

#%% Дополнительные функции для визуализации

# общая пористость                          
# porosity = ps.metrics.porosity(im)
# porosity = round(ps.metrics.porosity(im), 4)
# print('Общая пористость (с учетом закрытых пор): ' + str(porosity))


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
