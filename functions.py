import matplotlib.pyplot as plt
from astropy.io import fits as fits
import astropy
import plotly.express as px

import scipy
from scipy import signal ,stats
import pandas as pd

import numpy as np
from numpy.linalg import eig
from numpy import cov
 
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip
import os, math, sys, time
from astropy.stats import SigmaClip

import pathlib
from pathlib import Path

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

def dist(y,x,x_center,y_center):
    """Calculates radial distance from a center point (x_center, y_center) to a point (x, y)."""
    
    return int(math.sqrt((x-x_center)**2+(y-y_center)**2))


def sliding_median(arr, window):
    """Applies a sliding median filter with a specified window size along an array."""
    return np.nanmedian(np.lib.stride_tricks.sliding_window_view(arr, (window,)), axis=1)


def combine_LR(M_L,M_R):
    """Concatenates two matrices horizontally."""
    return np.concatenate((M_L.T, M_R.T)).T


def create_M_dist(x_center,y_center,x,y):
    """Creates a matrix where each cell contains the Euclidean distance from a center point.
    Args:
        x_center, y_center: Coordinates of the center point.
        x, y: Dimensions of the output matrix.
    Returns:
        A 2D array with distance values.
    """

    M_dist=np.zeros((y,x),dtype=float)
    for i in range(0,len(M_dist)):
        for j in range(0,len(M_dist[i])):
            M_dist[i][j]=dist(i,j,x_center,y_center)
    return M_dist 


def prepare_Mdist_from_raw(M):
    """Generates a matrix where each pixel represents its Euclidean distance from the star center.
    Args:
        M: Input 2D array representing the image.
    Returns:
        M_dist: 2D array with distance values.
    """
    dimy,dimx=M.shape

    """ Serches for the star center"""

    MedCentrim=scipy.signal.medfilt2d(M, kernel_size=3)  #median filter to eliminate bad pixels
    
    # Determine the center based on the maximum pixel value in a 200x200 central region
    cory,corx=np.where(MedCentrim==np.nanmax(MedCentrim[int(dimy/2-100):int(dimy/2+100)].T[int(dimx/2-100):int(dimx/2+100)].T))

    center_y,center_x=int(np.nanmedian(cory)),int(np.nanmedian(corx))  #if more than one max value

    M_dist=create_M_dist(center_x,center_y,dimx,dimy)      #  Można też brać do create_M_dist macierze szablony 

    return M_dist


def get_compressed_matrix_mean(liste_mat,y=1024,x=2048):
    """Compresses 3D matrices to 2D by applying a median along the third axis, if present.
    Args:
        liste_mat: List of 2D or 3D arrays.
    Returns:
        A 2D array representing the compressed matrix.
    """
    
    tot=np.zeros((len(liste_mat),y,x),dtype=float) # image convention: x horizontal, y vertical
    if liste_mat[0].shape[0]!=1 and len(liste_mat[0].shape)==3: # if a list contains many 3D cubes
        for i in range(0,len(liste_mat)):
            tot[i]=np.nanmean(liste_mat[i],axis=0)  #compresses each singular 3D matrix
    else:
        for i in range(0,len(liste_mat)):  # list contains 2D cubes
        
            tot[i]=liste_mat[i] 

    return np.nanmean(tot,axis=0)  #compresses all 2D matrix together


def get_compressed_matrix_median(liste_mat,y=1024,x=2048):
    """Compresses 3D matrices to 2D by applying a median along the third axis, if present.
    Args:
        liste_mat: List of 2D or 3D arrays.
    Returns:
        A 2D array representing the compressed matrix.
    """
    tot=np.zeros((len(liste_mat),y,x),dtype=float) # image convention: x horizontal, y vertical
    if liste_mat[0].shape[0]!=1 and len(liste_mat[0].shape)==3: #?
        for i in range(0,len(liste_mat)):
            tot[i]=np.nanmedian(liste_mat[i],axis=0)
    else:
        for i in range(0,len(liste_mat)):
        
            tot[i]=liste_mat[i]

    return np.nanmedian(tot,axis=0)


def bd_px_subs_nan(liste_indexes,Matrix):
    """Replaces pixels at specified indices in a matrix with a specified value (nan).
    Args:
        liste_indexes: Tuple of lists ([y indices], [x indices]).
        Matrix: 2D array in which to replace pixels.
    Returns:
        Matrix with bad pixels replaced.
    """

    Matrix_copy=np.copy(Matrix)
    for y, x in zip(*liste_indexes):
        Matrix_copy[y][x] = float("nan")
    """for i in range(0,len(liste_indexes[0])):
        
        y,x=liste_indexes[0][i],liste_indexes[1][i]
        Matrix_copy[y][x]= float("nan")"""

    return Matrix_copy


def bd_px_index(bad_pixel_map):
    """Finds indices of bad pixels in a bad pixel map.
    Args:
        bad_pixel_map: 2D array where 1 indicates a bad pixel.
    
    Returns:
        Tuple of lists ([y indices], [x indices]) of bad pixel locations.
    """
    return np.where(bad_pixel_map==1) 


def bd_px_subs_0(liste_indexes,Matrix):
    """Replaces pixels at specified indices in a matrix with a specified value (0).
    Args:
        liste_indexes: Tuple of lists ([y indices], [x indices]).
        Matrix: 2D array in which to replace pixels.
    Returns:
        Matrix with bad pixels replaced.
    """

    Matrix_copy=np.copy(Matrix)
    for y, x in zip(*liste_indexes):
        Matrix_copy[y][x] = 0

    return Matrix_copy


def get_bdpx_from_sky(cubes,threshold):
    """Creates a cumulative bad pixel map based on intensity thresholds in a list of sky images.
    Args:
        cubes: List of 2D arrays representing sky images.
        threshold: Intensity threshold for defining a bad pixel.
    Returns:
        A 2D array representing the bad pixel map.
    """

    y, x = cubes[0].shape
    bad_pixel_map = np.zeros((y,x))
    
    for cube in cubes:
        high_pass_filtered = scipy.signal.medfilt2d(cube, kernel_size=5)
        difference = cube - high_pass_filtered
        # Identify new bad pixels where absolute difference exceeds the threshold
        bad_pixels_y, bad_pixels_x = np.where(np.logical_and(np.abs(difference) > threshold, bad_pixel_map == 0))     #(np.abs(difference) > threshold) & (bad_pixel_map == 0))
        bad_pixel_map[bad_pixels_y, bad_pixels_x] = 1

    return bad_pixel_map



def create_new_bdpx_map(cubes, threshold=200, existing_bad_pixel_map=None):
    """Combines a new bad pixel map with an existing map.
    Args:
        cubes: List of 2D arrays representing sky images.
        threshold: Intensity threshold for defining a bad pixel.
        existing_bad_pixel_map: Optional 2D array of pre-existing bad pixels.
    Returns:
        A 2D array representing the combined bad pixel map.
    """

    new_bad_pixel_map = get_bdpx_from_sky(cubes,threshold)

    if existing_bad_pixel_map is not None:
        combined_map = np.maximum(existing_bad_pixel_map, new_bad_pixel_map)
    else:
        combined_map = new_bad_pixel_map

    return combined_map


def print_med_tot(zone, Matrix):
    """Prints statistical information about a defined background zone in the image.
    May be inappropriate if comparing PCA of the whole image and of the divided image: slices which don't contain star zone will be subtracted perfectly,
    reducing std dev even if substraction is worse in other parts of an image.
    std dev (donnes filtres): clip 3 sigma
    Args:
        zone: Binary array where 1 indicates background zone.
        matrix: 2D array representing the image."""

    masked_matrix = np.copy(Matrix)

    cor_y, cor_x=np.where(zone==0)
    for i in range(0,len(cor_y)):
        masked_matrix[cor_y[i]][cor_x[i]]=float("nan")

    result = masked_matrix.flatten()
    

    print(f"Median of background zone: {np.nanmedian(result)} \n")  #("Mediane sur la zone totale de background",np.nanmedian(result) ,"\n")  #nan!!!!!
    print(f"Std dev of background zone: {np.nanstd(result)} \n") #np.median(dev_L),
    print(f"Total flux sum squared: {np.nansum([a*a for a in result]):.2e} \n") #"Somme flux(x,y)^2: ", "{:.2e}".format(np.nansum([a*a for a in result])),np.nansum([a*a for a in result]))
   
    result=sigma_clip(result) 
    print(f"Std dev after sigma clipping: {np.nanstd(result)}\n")# ""Std dev sur la zone totale de background (donnes filtres) ",np.nanstd(result),"\n") #np.median(dev_L),


######## EXTRACTION #############################################################################################

def cubes_extraction_exptime_lin(compression='mean',bd_subs='no'):
    """
    Extract files from the "raw" dierctory. Files should be science files and sky files only. Atribution to science or sky category is 
    effectuated by checking "ESO DPR TYPE" keyword of header (should be "OBJECT" for science). 

    You may use SKY(SCIENCE)_cubes_extraction_list rather than this fonction if working on the same data (extraction without list sorting).
    See main fonction: files names list should be obtained with cubes_extraction_ at first but may be directly implamented in main function later on.
    
    INPUT: compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
    bd_subs='yes'/'no' for bad pixels substraction. If yes bad pixels map (only one) should be placed in 'bad pixels map' directory
    
    OUTPUT: np.array(cubes): 3D array (skies), np.array(cubes_science): 2D array (science) ,sky_files_list : list, science_files_list : list"""

    folder=os.path.dirname(__file__)+'\\raw\\'  
    file_list=os.listdir(folder)
    sky_files_list=[]  #contains a list of names of sky files in a same order as cubes list
    science_files_list=[]  #science files names
    liste_header=[]

    cubes=[]  #a list of sky/dark cubes
    cubes_science=[] #a list of science cubes
    liste_exp=[]
    count=0

    for i in range(0,len(file_list)):

        data=fits.getdata(folder+file_list[i])
        header=fits.getheader(folder+file_list[i])
        if "ESO DPR TYPE" in list(header.keys()):

            if header["ESO DPR TYPE"] =='OBJECT'  :  #check if file contain science
                cubes_science.append(data) 
                science_files_list.append(file_list[i]) 
                exp_sc=header['EXPTIME']  
            else:
                cubes.append(data)  # raw sky
                sky_files_list.append(file_list[i]) 
                exp=header['EXPTIME']       #ESO PRO TYPE = 'REDUCED ' pour master sky
                liste_exp.append(exp)
                header=fits.getheader(folder+file_list[i])
                liste_header.append(header)

        else:
            cubes.append(data)   # master sky
            sky_files_list.append(file_list[i]) 
            exp=header['EXPTIME'] 
            liste_exp.append(exp) 
            header=fits.getheader(folder+file_list[i])
            liste_header.append(header)

    sky_96=cubes[liste_exp.index(96)].copy()

    #flat_96=sky_96.copy().flatten()
    m_1=[np.mean(sky_96[500:550].T[300:350].T),np.mean(sky_96[500:550].T[1024+300:1024+350].T)]

    for j in range(0,len(cubes)):    #compression of sky matrices 

        if compression=='mean':
            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            print(j)
            mc_1=[np.mean(cubes[j][500:550].T[300:350].T),np.mean(cubes[j][500:550].T[1024+300:1024+350].T)]


            try:
                a,b=np.polyfit(np.array(mc_1), np.array(m_1), 1)
                    #a=np.polyfit(np.array(flat_c[i-6:i]), np.array(flat_96[i-6:i]), 0)
            except:
                pass
                
            """print(np.mean(a_li),np.mean(b_li),"mean",liste_exp[j])
            a,b=np.mean(a_li),np.mean(b_li)
            cubes[j]=(cubes[j]*a+np.full((1024,2048),b ))*(1/96)"""
            print(a,b,"mean",liste_exp[j])
            cubes[j]=(cubes[j]*a+np.full((1024,2048),b ))*(1/96)
            #cubes[j]=(cubes[j]*a)*(1/96)
        

   

        if bd_subs=='yes' and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  
            
    
    
    """fits.writeto('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir 2/PCA_K12/ciel_pose_lin2.fits',np.array(cubes[:12]),overwrite=True)
 
    sys.exit()"""






    """for j in range(0,12): #len(cubes)):    #compression of sky matrices 

        if compression=='mean':
            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            print(j)
            flat_c=cubes[j].copy().flatten()
            a_li,b_li=[],[]
            for i in range(6,len(flat_c),300):#(10000,10036,1):#(6,len(flat_c),1):
                
                try:
                    a,b=np.polyfit(np.array(flat_c[i-6:i]), np.array(flat_96[i-6:i]), 1)
                    #a=np.polyfit(np.array(flat_c[i-6:i]), np.array(flat_96[i-6:i]), 0)
                except:
                    pass

                a_li.append(a)
                #b_li.append(b)
                
            print(np.mean(a_li),np.mean(b_li),"mean",liste_exp[j])
            a,b=np.mean(a_li),np.mean(b_li)
            cubes[j]=(cubes[j]*a+np.full((1024,2048),b ))*(1/96)
            print(np.mean(a_li),"mean",liste_exp[j])
            a=np.mean(a_li)
            cubes[j]=(cubes[j]*a)*(1/96)
        

            
        elif compression=='median':
            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            flat_c=cubes[j].copy().flatten()
            a=np.polyfit(flat_c, flat_96, 1)
            cubes[j]=cubes[j]*a#+np.full((1024,2048),b )


        if bd_subs=='yes' and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  
            
    
    
    fits.writeto('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir 2/PCA_K12/ciel_pose_lin2.fits',np.array(cubes[:20]),overwrite=True)
 
    sys.exit()"""




    if compression=='mean':             #compression of science matrices 
        cubes_science=get_compressed_matrix_mean(cubes_science) #    2D matrix 
        cubes_science=1/96* cubes_science

            
    elif compression=='median':
        cubes_science=get_compressed_matrix_median(cubes_science)
 

    print("Cubes extraction done")

    
    return cubes, np.array(cubes_science), sky_files_list, science_files_list , True   #, liste_exp


def cubes_extraction_all(compression='mean',bd_subs=False):
    """
    Extracted files are normalized to a base using the formula: (matrix−sky_pose)/(matrix_time−0.83).
    Sky_pose of exposure time of 0.83s ideally includes only read noise (?).
    To fix: bad pixels map should be different for every exposure time (inefficient pca otherwise)

    Files should be science files and sky files only. Atribution to science or sky category is 
    effectuated by checking "ESO DPR TYPE" keyword of header (should be "OBJECT" for science). 

    You may use SKY(SCIENCE)_cubes_extraction_list rather than this fonction if working on the same data (extraction without list sorting).
    See main fonction: files names list should be obtained with cubes_extraction_ at first but may be directly implamented in main function later on.
    
    Args: 
        compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
        bd_subs=True/False. bad pixels substraction. If True bad pixels map (only one) should be placed in 'bad pixels map' directory
    Returns:
        cubes (np.array): 3D array (skies)
        cubes_science (np.array): 2D array (science)
        sky_files_list (list): List of sky file names
        science_files_list (list): List of science file names
        True : files will be remultiplied by the exposure time later 
    """

    print("Cubes extraction")

    folder=os.path.dirname(__file__)+'\\raw\\' #zmien raw
    sky_pose=fits.getdata(os.path.dirname(__file__)+'\sky_pose_083.fits')

    file_list=os.listdir(folder)
    sky_files_list, science_files_list = [], []
    cubes, cubes_science, liste_exp = [], [], []
    count=0

    for i in range(0,len(file_list)):

        data=fits.getdata(folder+file_list[i])
        header=fits.getheader(folder+file_list[i])

        if "ESO DPR TYPE" in list(header.keys()):

            if header["ESO DPR TYPE"] =='OBJECT'  :  #check if file contain science
                cubes_science.append(data) 
                science_files_list.append(file_list[i]) 
                exp_sc=header['EXPTIME']
 
            else:
                cubes.append(data)  # raw sky
                sky_files_list.append(file_list[i]) 
                exp=header['EXPTIME']       #ESO PRO TYPE = 'REDUCED ' pour master sky
                liste_exp.append(exp)
                

        else:
            cubes.append(data)   # master sky
            sky_files_list.append(file_list[i]) 
            exp=header['EXPTIME'] 
            liste_exp.append(exp) 

    if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)


    for j in range(0,len(cubes)):    #compression of sky matrices 


        if compression=='mean':
            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            cubes[j]=(cubes[j]-sky_pose)/(liste_exp[j]-0.83)
            #cubes[j]=(cubes[j])/(liste_exp[j]) 
            
        elif compression=='median':
            cubes[j]=get_compressed_matrix_median([cubes[j]])
            cubes[j]=(cubes[j]-sky_pose)/(liste_exp[j]-0.83)
            #cubes[j]=(cubes[j])/(liste_exp[j]) 
        
        if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  
            

    if compression=='mean':             #compression of science matrices 
        cubes_science=get_compressed_matrix_mean(cubes_science) #    2D matrix 
        cubes_science=(cubes_science-sky_pose)/(exp_sc-0.83)
        #cubes_science=(cubes_science)/(exp_sc) 
            
    elif compression=='median':
        cubes_science=get_compressed_matrix_median(cubes_science)
        cubes_science=(cubes_science-sky_pose)/(exp_sc-0.83)
        #cubes_science=(cubes_science)/(exp_sc) 

    #
    print("Cubes extraction (all) done")
    #fits.writeto('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir 2/PCA_K12/base1s_ND_4s.fits',np.array(cubes[:15]),overwrite=True)
 
    sys.exit()

    return cubes, np.array(cubes_science), sky_files_list, science_files_list ,True    #, liste_exp


def cubes_extraction_(compression='mean',bd_subs=False):
    """
    Extract files from the "raw" dierctory. Files should be science files and sky files only. Atribution to science or sky category is 
    effectuated by checking "ESO DPR TYPE" keyword of header (should be "OBJECT" for science). 

    You may use SKY(SCIENCE)_cubes_extraction_list rather than this fonction if working on the same data (extraction without list sorting).
    See main fonction: files names list should be obtained with cubes_extraction_ at first but may be directly implamented in main function later on.
    
    INPUT: compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
    bd_subs=True/False. bad pixels substraction. If True bad pixels map (only one) should be placed in 'bad pixels map' directory
    OUTPUT: np.array(cubes): 3D array (skies), np.array(cubes_science): 2D array (science) ,sky_files_list : list, science_files_list : list"""

    print("Cubes extraction")
    folder=os.path.dirname(__file__)+'\\raw\\'
    
    file_list=os.listdir(folder)
    sky_files_list=[]  #contains a list of names of sky files in a same order as cubes list
    science_files_list=[]  #science files names

    cubes=[]  #a list of sky/dark cubes
    cubes_science=[] #a list of science cubes

    for i in range(0,len(file_list)):

        data=fits.getdata(folder+file_list[i])
        header=fits.getheader(folder+file_list[i])
        if "ESO DPR TYPE" in list(header.keys()):

            if header["ESO DPR TYPE"] =='OBJECT'  :  #check if file contain science
                cubes_science.append(data) 
                science_files_list.append(file_list[i]) 
                
            else:
                cubes.append(data)  # raw sky
                sky_files_list.append(file_list[i])                                               #ESO PRO TYPE = 'REDUCED ' pour master sky

        else:
            cubes.append(data)   # master sky
            sky_files_list.append(file_list[i])   

    

    for j in range(0,len(cubes)):    #compression of sky matrices 

        if compression=='mean':
            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            
        elif compression=='median':
            cubes[j]=get_compressed_matrix_median([cubes[j]])

        if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  
            


    if compression=='mean':             #compression of science matrices 
        
        cubes_science=get_compressed_matrix_mean(cubes_science) #    2D matrix 
            
    elif compression=='median':
        cubes_science=get_compressed_matrix_median(cubes_science)

    cubes_science=bd_px_subs_0(indexes_bd,cubes_science) 

    
    print("Cubes extraction done")
    return np.array(cubes),np.array(cubes_science),sky_files_list, science_files_list   


def cubes_extraction_exptime(compression='mean',bd_subs=False):
    """
    Extract files from the "raw" dierctory. Files should be science files and sky files only (may be of diffrent exposure times). Atribution to science or sky category is 
    effectuated by checking "ESO DPR TYPE" keyword of header (should be "OBJECT" for science). 

    You may use SKY(SCIENCE)_cubes_extraction_list rather than this fonction if working on the same data (extraction without list sorting).
    See main fonction: files names list should be obtained with cubes_extraction_exptime at first but may be directly implamented in main function later on.
    
    INPUT: compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
    bd_subs=True/False. bad pixels substraction. If True bad pixels map (only one) should be placed in 'bad pixels map' directory
    OUTPUT: np.array(cubes): 3D array (skies), np.array(cubes_science): 2D array (science) ,sky_files_list : list, science_files_list : list"""

    print("Cubes extraction")
    folder=os.path.dirname(__file__)+'\\raw\\'
    
    file_list=os.listdir(folder)
    sky_files_list=[]  #contains a list of names of sky files in a same order as cubes list
    science_files_list=[]  #science files names

    cubes=[]  #a list of sky/dark cubes
    cubes_science=[] #a list of science cubes
    

    for i in range(0,len(file_list)):
        header=fits.getheader(folder+file_list[i])
        if "ESO DPR TYPE" in list(header.keys()):  
            if header["ESO DPR TYPE"] =='OBJECT':
                exp_sc=header['EXPTIME']

    #exp_sc=64
    for i in range(0,len(file_list)):

        data=fits.getdata(folder+file_list[i])
        header=fits.getheader(folder+file_list[i])
        if "ESO DPR TYPE" in list(header.keys()):

            if header["ESO DPR TYPE"] =='OBJECT'  :  #check if file contain science
                cubes_science.append(data) 
                science_files_list.append(file_list[i])    
            else:
                if exp_sc==header['EXPTIME']:
                    cubes.append(data)  # raw sky
                    
                    sky_files_list.append(file_list[i])                                               #ESO PRO TYPE = 'REDUCED ' pour master sky

        else:
            if exp_sc==header['EXPTIME']:
                cubes.append(data)   # master sky
            
                sky_files_list.append(file_list[i])   

    

    for j in range(0,len(cubes)):    #compression of sky matrices 

        if compression=='mean':

            cubes[j]=get_compressed_matrix_mean([cubes[j]])
            
        elif compression=='median':

            cubes[j]=get_compressed_matrix_median([cubes[j]])

        if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end

            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  
            


    if compression=='mean':             #compression of science matrices 
    
        cubes_science=get_compressed_matrix_mean(cubes_science) #    2D matrix 
            
    elif compression=='median':

        cubes_science=get_compressed_matrix_median(cubes_science)


    if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:
        cubes_science=bd_px_subs_0(indexes_bd,cubes_science) 

    
    print("Cubes extraction done")
   
    return np.array(cubes),np.array(cubes_science),sky_files_list, science_files_list , False  


def SKY_cubes_extraction_list(liste,compression='mean',bd_subs=False,exp_t=False):  #Do not enter bd_subs='yes' for science cubes
    """
    Extract cubes from files which names are registered in the list (files should be placed in the raw directory)
    To obtain list use first cubes_extraction_


    INPUT: compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
    bd_subs=True/False. bad pixels substraction. If True bad pixels map (only one) should be placed in 'bad pixels map' directory
    OUTPUT: np.array(cubes): 3D array (skies)"""
    print("Cubes extraction from list")

    folder=os.path.dirname(__file__)+'\\raw\\'
    cubes=[]
    for i in range(0,len(liste)):

        data=fits.getdata(folder+str(liste[i]))

        if exp_t==True:
            header=fits.getheader(folder+str(liste[i]))
            exp_sc=header['EXPTIME']

        cubes.append(data)

    for j in range(0,len(cubes)):

        if compression=='mean':
            
            cubes[j]=get_compressed_matrix_mean([cubes[j]])

        elif compression=='median':
            cubes[j]=get_compressed_matrix_median([cubes[j]])

        if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            cubes[j]=bd_px_subs_0(indexes_bd,cubes[j])  

    print("Cubes extraction done")
    return np.array(cubes)


def SCIENCE_cubes_extraction_list(liste,compression='mean',bd_subs=False, exp_t=False):  #Do not enter bd_subs='yes' for science cubes
    """
    Extract cubes from files whose names are registered in the list (files should be placed in the raw directory)
    To obtain list use first cubes_extraction_


    INPUT: compression = 'mean' / 'median'  compression of the cube of each file (not all the cubes from diffrent files)
    bd_subs=True/False. bad pixels substraction. If True bad pixels map (only one) should be placed in 'bad pixels map' directory
    OUTPUT: np.array(cubes): 2D array (science)"""
    print("Cubes extraction from list")
    folder=os.path.dirname(__file__)+'\\raw\\'
    cubes=[]
    for i in range(0,len(liste)):

        data=fits.getdata(folder+str(liste[i]))
        if exp_t==True:
            header=fits.getheader(folder+str(liste[i]))
            exp_sc=header['EXPTIME']
        
        cubes.append(data)
    
    if compression=='mean':
            
        cubes=get_compressed_matrix_mean(cubes)

    elif compression=='median':
            cubes=get_compressed_matrix_median([cubes])


    if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
        bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
        indexes_bd=bd_px_index(bd_px_cube)
        cubes=bd_px_subs_0(indexes_bd,cubes) 


    print("Cubes extraction done")
    return np.array(cubes)



def rename_files(new_file_name="sky_",folder=os.path.dirname(__file__)+'\\raw\\'):
    """ rename all the files in the given directory
    folder = r'c:/Users/...'
    new_file_name='science_'   # Only fits files"""
    
    count = 0 
    for file in os.listdir(folder):

        source = folder + file
        destination = folder + new_file_name + str(count) +".fits"
        os.rename(source, destination)
        count += 1
    print('All Files Renamed')
    return None


def collect_header():
    """ returns header from one of science matrices. All information from the header is conserved
    OUTPUT: header, h_object: str"""
    folder=os.path.dirname(__file__)+'\\raw\\' #zmiwnraww
    file_list=os.listdir(folder)
    header_s=0
    i=0
        
    while header_s==0:
        header=fits.getheader(folder+file_list[i])

        if "ESO DPR TYPE" in list(header.keys()):  

            if header["ESO DPR TYPE"] =='OBJECT'  :  #check if file contain science
                header_s=header
                h_object=header["OBJECT"]

        i=i+1

    return header,h_object


def sec_elem(b):
    """Gives second element of a list/tuple: see get_sorted_sky_list """
    return b[1]


def d_eval( a, b, type_eval,zone):
    """Evaluates resamblance between matrices by calculating distance metric
    Args:
        a: 2D science image.
        b: 2D sky image.
        type_eval: String ('std', 'med', 'square').
        zone: 2D array containingevaluation zone.

    Returns:
        A float representing the resemblance score.
    """
     # Calculate difference matrix
    M_dist=a-b  
    M_dist=combine_LR(M_dist[40:964].T[41:901].T,M_dist[40:964].T[(1024+41):(1024+901)].T)  

    if type_eval=="std":      # list sorted by the sum of  distance
        res=np.nanstd(np.square(M_dist),axis=(0,1))

    elif type_eval=="med":      # list sorted by the median of distance
        res=np.nanmedian(np.square(M_dist), axis=(0,1)) #abs

    elif type_eval=="square": # list sorted by the sum of squares of pixels values
        res=np.nansum(np.square(M_dist),axis=(0,1))
    
    return  res


def zone_subs(M,zone):
    """Replace values in the star zone with NaN.
    Args:
        M: 2D array 
        zone: 2D array where 0 indicates pixels to replace with NaN.
    Returns:
        2D array with NaNs in the specified zone.
    """

    M=np.copy(M)

    """cor_y, cor_x=np.where(zone==0)
    pos=[]
    for i in range(0,len(cor_y)):
            pos.append((cor_y[i],cor_x[i]))

    rows, cols = zip(*pos)
    M[rows, cols] = float("nan")"""
    
    cor_y, cor_x = np.where(zone == 0)
    M[cor_y, cor_x] = float("nan")
   
    return M


def get_sorted_sky_list( cubes, science, zone, files_name_list, type_eval="norme"):
    """Sort sky matrices based on resemblance to a science matrix.
    Args:
        cubes: List of sky images.
        science: 2D array
        zone: 2D array of evaluation zone.
        file_names: List of filenames for each sky image.
        type_eval: String  ('std', 'med', 'suqare').
    Returns:
        Tuple containing list of sorted sky matrices and list of corresponding sorted filenames.
    """
    #is done for both filters at once (try to devide?)
    
    science_bg=zone_subs(science.copy(),zone)  # only science background zone is used
    files_name_list=files_name_list.copy()
    file_sorted_N=[]  # list containing tuple for each sky matrix: (sky cube, parameter of resemblance, file name)
    
    for i in range(0,len(cubes)):
        
        sky_bg=zone_subs(cubes[i].copy(),zone) 

        param=d_eval(science_bg,sky_bg,type_eval,zone)   #calculate value of ressemblance parameter


        file_sorted_N.append((cubes[i], param, files_name_list[i])) 

    # sorting list by the values of resamblance parameter
    liste_sky,files_list_name_sorted=[],[]
    file_sorted_N.sort(key=sec_elem) # parameter is second element of the tuple
    
    for i in range(0,len(file_sorted_N)):
        liste_sky.append(file_sorted_N[i][0]) 
        files_list_name_sorted.append(file_sorted_N[i][2])

    print("Sky cubes sorted")

    return liste_sky, files_list_name_sorted

######### ZONE ###############################################################################################

def get_sky_smoothed(cubes,start_x, end_x, start_y, end_y, filt, kernel=11):
    """Create a smoothed sky matrix by calculating the mean of all skies in cubes.
    Args:
        cubes: 3D array of sky images.
        start_x, end_x, start_y, end_y: Integers (region of interest)
        kernel: Integer defining the kernel size for median filtering, should be odd.
        filt: "K1"/"K2"
    Returns:
        2D array representing the smoothed sky matrix.
    """

    cube=get_compressed_matrix_mean(cubes)
    if len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies. Science bd px substraction is done at the end
            
        bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
        indexes_bd=bd_px_index(bd_px_cube)
        cube=bd_px_subs_nan(indexes_bd,cube) 

    if filt=="K1":
        cube=cube.T[:1024].T[start_y:end_y].T[start_x:end_x].T
    elif filt=='K2':
        cube=cube.T[1024:].T[start_y:end_y].T[start_x:end_x].T
    CubeCentrim=scipy.signal.medfilt2d(cube, kernel_size=kernel)

    res=wrap(CubeCentrim,start_x,end_x,start_y,end_y,value="nan")

    return res


def equal_dist(r,Matrix,M_dist):
    """Extract a vector of pixels from a specified ring distance from the center.
    Args:
        r: Integer radius from center.
        Matrix: 2D array to extract pixels from.
        M_dist: Distance matrix where each cell indicates distance from center.
    Returns:
        1D array of pixel values from the specified radius.
    """
    if Matrix.shape!=M_dist.shape:
        print("equal_dist: matrices are not of the same shape")
    
    liste_indexes=np.where(M_dist==r)
    """v_eq=np.zeros(len(liste_indexes[0]))

    for h in range(0,len(liste_indexes[0])):
        v_eq[h]=Matrix[liste_indexes[0][h]][liste_indexes[1][h]]"""

    v_eq = np.array([Matrix[liste_indexes[0][i], liste_indexes[1][i]] for i in range(len(liste_indexes[0]))])

    return v_eq


def  get_zone1(zone_resc,r,M_dist, edge=40):
    """Create a signal zone mask for a single K1 or K2 matrix half-image. 0 = star signal, 1 = star signal
    Args:
        zone_resc: 2D array of 1s or pre-existing zone mask to modify.
        r: Integer radius of the central signal circle
        M_dist: Distance matrix, same dimensions as zone_resc
        edge: Integer specifying border edge 
    Returns:
        2D array 
    """
    y, x = np.where(np.logical_and((M_dist<r),(zone_resc[:,:]==1)))  #pixels that has to be assigned to the star signal zone
    
    for i in range(0,len(y)):
        zone_resc[y[i]][x[i]]=0   

    for i in range(0,edge):
        zone_resc[i][:]=0

    for i in range(zone_resc.shape[0]-edge,zone_resc.shape[0]):
        zone_resc[i][:]=0

    for i in range(0,edge):
        zone_resc.T[i][:]=0

    for i in range(zone_resc.shape[1]-edge,zone_resc.shape[1]):
        zone_resc.T[i][:]=0
  

    return zone_resc


def zone_by_dispertion(cubes, science, zone_approx, start_x, end_x, start_y, end_y,filt,r_specified=0,  lamb=2.2, n_sky=3):
    """Determine the star signal zone based on standard deviation criteria.
    Args:
        cubes: 3D array, sky images.
        science: 2D array 
        zone_approx: 2D array, approximate zone.
        start_x, end_x, start_y, end_y: Integers defining the region of interest.
        filt: "K1"/"K2"
        r_specified: Optional predefined radius for center star zone.
        lamb: Wavelength of the filter (used in radius calculation).
        n_sky: Integer specifying the number of skies for averaging.
    Returns:
        Tuple containing: 2D array defining the star signal zone; Integer radius of the star zone.
    """

    cube=get_compressed_matrix_mean(cubes[:n_sky]) # sky acceptably well adapted to science 
    

    if len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:     #bad pixels substraction from skies and science
            
        bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
        indexes_bd=bd_px_index(bd_px_cube)
        cube=bd_px_subs_nan(indexes_bd,cube) 
    
    science=bd_px_subs_nan(indexes_bd,science.copy()) 


    if filt=="K2":
        cube=cube.T[1024:].T[start_y:end_y].T[start_x:end_x].T   #reduction to the zone of intrest
        science=science.T[1024:].T[start_y:end_y].T[start_x:end_x].T

    elif filt=="K1":
        cube=cube.T[:1024].T[start_y:end_y].T[start_x:end_x].T
        science=science.T[:1024].T[start_y:end_y].T[start_x:end_x].T

    

    zone_pre=zone_approx.T[:1024].T[start_y:end_y].T[start_x:end_x].T #Approximative zone is the same for K1 and K2

    M_dist=prepare_Mdist_from_raw(science) #distance matrix of science


    first_final=science-cube   #first approximation of the reduced image
    final_smoothed=scipy.signal.medfilt2d(first_final, kernel_size=3)  #eliminates bad pixels


    final_smoothed1D=get_flatten_matrix(final_smoothed,0,end_y-start_y,0,end_x-start_x)  
    zone_pre1D=get_flatten_matrix(zone_pre,0,end_y-start_y,0,end_x-start_x)
    
    cor=np.where(zone_pre1D==0)  # Exclusion of pixels from approximative star zone

    smoothed_optim=np.delete(np.copy(final_smoothed1D),cor[0])


    result=sigma_clip(smoothed_optim) 
    std=np.nanstd(result)

    #To matrices are created: low SNR used for finding a radius of the circular center signal zone;
        # high SNR matrix used to find background sources. Informations from both matrices are than combain (to exclude background sources and star signal)
    
    y_sc,x_sc=np.where(wrap(final_smoothed,start_x,end_x,start_y,end_y,value="nan")>5*std) # collect values greater than 4 (or 5; this is arbitrary) sigma 

       
    zone_low=np.full((1024,1024),1) #zone low correspond to low SNR matrix that cannot be used as a final zone (criterium science > 4*std let pass some bg noise)
    for i in range(0,len(y_sc)):
        zone_low[y_sc[i]][x_sc[i]]=0   # 0 for star signal zone

    if r_specified==0:  #calculate approximative radius of center circle (if r not specified)
        mini=140*lamb/2.2 #minimum radius correspond to ring of fire calculated by r= 140 *lambda filter/lambda K2
        stop=286 # stop*1.35< 383 for correct PCA background zone should not be to small   283?

      
        
        for i in range(1,285):
            v_eq=equal_dist(i,zone_low,wrap(M_dist,start_x,end_x,start_y,end_y,value="nan"))#1D vector of pixels at the given r (ring pixels)


            if stop==286 and np.mean(v_eq)>0.75 and i>mini:  #75 do 4
                
            # 0.75 and 1.3 is arbitrary. Mean(v_eq) limit is so low as fixing higher limit would result in overestimation of r for K2 (presence of noise near center)
                stop=i* 1.35    ##(0.0022*i+0.72)     #z 4: 1.2

        if np.mean(v_eq)<0.75: # star zone is maximal
            stop=286*1.35

        print(filt,"Radius star zone",stop)


    else:
        stop=r_specified
        print(filt,"Radius star zone",stop)

    

    y_sc,x_sc=np.where(wrap(final_smoothed,start_x,end_x,start_y,end_y,value="nan")>6*std) #64:6
    # high SNR: only star signal is conserved. zone_high will be combine with the circle of radius determined with low SNR matrix
    zone_high=np.full((1024,1024),1)
    for i in range(0,len(y_sc)):
        zone_high[y_sc[i]][x_sc[i]]=0 

    # edge zone is masked by wrap fonction
    zone_recomb=get_zone1(zone_high.T[:1024].T[start_y:end_y].T[start_x:end_x].T, stop , M_dist, edge=0) # adding a centered circle to the high SNR zone matrix
    
    zone_=wrap(zone_recomb,start_x,end_x,start_y,end_y,value="0")

    print("Zone evaluation done",filt)
    print("zone created", filt)
    
    return zone_,stop


def get_flux_from_ct(matrix,M_dist_,r):
    """ zone_by_integration sub-function
    Integrates flux from center, contained in the circle of a given radius
    INPUT:matrix: 2D array that is integrated,  M_dist_ : 2D distance matrix ,r : radius from center
    OUTPUT: flux: integer , len(px_c): integer, number of pixels contained in area"""
    
    cor_y, cor_x=np.where(M_dist_<r)

    px_c=[]
    for i in range(0,len(cor_y)):
            px_c.append(matrix[cor_y[i]][cor_x[i]])
    
    flux=np.nansum(np.array(px_c))
    return flux, len(px_c)


def signal_zone_eval(science,dimy,dimx,larg,limit_deriv,plot_show=False):
    """zone_by_integration sub-function
    calculates star center, evaluates integrated flux and assign limit of star zone by the value of derivative of integrated flux fonction
    INPUT:science: 2D array ,dimy,dimx : dimensions science, larg: centrim width, limit_deriv: derivative value at with zone limit is fixed,
    plot_show=False : plot of integrated flux 
    OUTPUT: r_L,r_R,center_x_L,center_y_L,center_y_R,center_x_R: integers"""

    
    #cerches for star center
    centrim_L=science[int(dimy/2-larg/2):int(dimy/2+larg/2)].T [int(dimx/2-larg/2):int(dimx/2+larg/2)].T  #takes only center of image for the search area
    centrim_R=science[int(dimy/2-larg/2):int(dimy/2+larg/2)].T [int(dimx/2-larg/2)+1024:int(dimx/2+larg/2)+1024].T

    MedCentrim_L=scipy.signal.medfilt2d(centrim_L, kernel_size=3)  #eliminates bad pixels
    MedCentrim_R=scipy.signal.medfilt2d(centrim_R, kernel_size=3)
    
    #serchees only by the image center 200x200 to minimalizate bad pixels occurance
    coryL,corxL=np.where(MedCentrim_L==np.nanmax(MedCentrim_L[int(larg/2-100):int(larg/2+100)].T[int(larg/2-100):int(larg/2+100)].T))
    coryR,corxR=np.where(MedCentrim_R==np.nanmax(MedCentrim_R[int(larg/2-100):int(larg/2+100)].T[int(larg/2-100):int(larg/2+100)].T))
    center_y_L,center_x_L=int(np.nanmedian(coryL)),int(np.nanmedian(corxL))  #if more than one max value 
    center_y_R,center_x_R=int(np.nanmedian(coryR)),int(np.nanmedian(corxR)) 


    x,y=larg,larg
    M_dist_L=create_M_dist(center_x_L,center_y_L,x,y)      #  Można też brać do create_M_dist macierze szablony 
    M_dist_R=create_M_dist(center_x_R,center_y_R,x,y)

    flux_L=[]  # average intensity by a pixel at a given distance from center ??
    flux_R=[]
    ray=[]
    pt=445  #max radius for which flux is calculated 
    for i in range (0,pt):
        ray.append(i)
        flux_L.append(get_flux_from_ct(MedCentrim_L,M_dist_L,i)[0]/get_flux_from_ct(MedCentrim_L,M_dist_L,i)[1])
        flux_R.append(get_flux_from_ct(MedCentrim_R,M_dist_R,i)[0]/get_flux_from_ct(MedCentrim_R,M_dist_R,i)[1])

    if plot_show:
        plt.plot(ray, flux_L,color='blue', label='flux L')
        plt.plot(ray, flux_R,color='red', label='flux R')

        plt.title("Flux moyen par pixel en fonction du radius")
        print("Flux moyen par pixel en fonction du radius")
        plt.show()

    r_L=0
    r_R=0
    dif1=[]
    dif2=[]
    for i in range (0,pt-100):
        dif1.append((flux_L[i]-flux_L[i+100])/100)
        dif2.append((flux_R[i]-flux_R[i+100])/100)
        if r_L==0 and round((flux_L[i]-flux_L[i+100])/100,1)==limit_deriv:
            r_L=i+50
        if r_R==0 and round((flux_R[i]-flux_R[i+100])/100,1)==limit_deriv:  #nie mozna isc do dwoch cyfr
            r_R=i+50

    if plot_show:
        
        plt.plot(ray[0:445-100], dif1,color='blue', label='flux L')
        plt.plot(ray[0:445-100], dif2,color='red', label='flux R')

        plt.title("DERIVEE Flux moyen par pixel en fonction du radius")
        print("Flux moyen par pixel en fonction du radius")
        plt.show()
    
    if r_L==0:
        r_L=395

    if r_R==0:
        r_R=395
        
    r_R=r_L  #in the end right side is not calculated correctly for most cases

    return r_L,r_R,center_x_L,center_y_L,center_y_R,center_x_R


def  get_zone2(zone_resc,r_L,r_R,M_dist_L,M_dist_R, edge=40):
    """Creates a 1 and 0 matrix of a star signal zone for both K1 or K2. 0 = star signal, 1 = background signal
     INPUT:  zone_resc: 2D array can be 0s or 1s,
     r_L, r_R: integers, radius of the center circle , M_dist_L,M_dist_R : distance matrices for K1 and K2, 
     edge=40: fixes borders (0 assigned). This is not crucial as cuting with x_start , x_end... gives the same result 
    OUTPUT: 2D zone matrix  """
    
    y_left_S,x_left_S=np.where(np.logical_and((M_dist_L<r_L),(zone_resc[:,:1024]==1)))  #ce qu'il faut changer en signal (donc en 0) dans la zone gauche
    for i in range(0,len(y_left_S)):
        zone_resc[y_left_S[i]][x_left_S[i]]=0   #nie możliwe

    y_left_BG,x_left_BG=np.where(np.logical_and(M_dist_L>r_L,zone_resc[:,:1024]==0))  #ce qu'il faut changer en zone de rescaling (donc en 1) dans la zone gauche
    for i in range(0,len(y_left_BG)):
        zone_resc[y_left_BG[i]][x_left_BG[i]]=1

    y_right_S,x_right_S=np.where(np.logical_and(M_dist_R<r_R,zone_resc[:,1024:]==1))  #ce qu'il faut changer en signal (donc en 0) dans la zone gauche
    for i in range(0,len(y_right_S)):
        zone_resc[y_right_S[i]][x_right_S[i]+1024]=0  #nie możliwe

    y_right_BG,x_right_BG=np.where(np.logical_and(M_dist_R>r_R,zone_resc[:,1024:]==0))  #ce qu'il faut changer en zone de rescaling (donc en 1) dans la zone gauche
    for i in range(0,len(y_right_BG)):
        zone_resc[y_right_BG[i]][x_right_BG[i]+1024]=1

    #zone de bord
    for i in range(0,edge):
        zone_resc[i][:]=0

    for i in range(1024-edge,1024):
        zone_resc[i][:]=0


    for i in range(0,edge):
        zone_resc.T[i][:]=0

    for i in range(zone_resc.shape[1]-edge,zone_resc.shape[1]):
        zone_resc.T[i][:]=0 


    for i in range(1024-edge-30,1024+edge):  # region of passege K1:K2
        zone_resc.T[i][:]=0  

    print("Zone created")
   
    return zone_resc


def zone_by_integration(science,limit_deriv=0.7,save_="no",ratio=11/14):
    """creates zone by integration of star flux: using zone_by_dispertion may give more accurate results.
    INPUT: science: 2D array,
    limit_deriv=0.7 : parameter that defines the star zone limit: at what rate of decrease (derivate value) of avearage intensity by a pixels the star zone has its limit,
    save_="no"/"yes": whather to save zone in zone directory , 
    ratio=11/14: ratio radius K1 zone/ radiur K2 zone as radius K2 zone is mostly calculated incorrectly
    OUTPUT: zone 2D matrix """

    print("Zone evaluation")
    larg=900   #lenght of the edge of the square zone where the star is located
    dimy,dimx=1024,1024  #
     #
    r_L, r_R, center_x_L, center_y_L, center_y_R, center_x_R=signal_zone_eval(science,dimy,dimx,larg,limit_deriv,False)

    center_x_L, center_y_L=center_x_L+int(dimx/2-larg/2),center_y_L+int(dimx/2-larg/2)
    center_y_R, center_x_R=center_x_R+int(dimx/2-larg/2),center_y_R+int(dimx/2-larg/2)
    r_R=r_L*ratio
    print("Star radisu (px) K1, K2:", r_L, r_R, "\n Star center (px) K1 (x,y), K2 (x,y):",center_x_L, center_y_L, center_y_R, center_x_R)

    
    M_dist_L_2=create_M_dist(center_x_L, center_y_L,dimx,dimy)      #  Creates distance matrices of the whole K1/K2 image 
    M_dist_R_2=create_M_dist(center_x_L, center_y_L,dimx,dimy)

    edge=40
    zone=get_zone2(np.full((1024,1024),1), r_L, r_R ,M_dist_L_2, M_dist_R_2, edge) 
    if save_=="yes":
        fits.writeto(os.path.dirname(__file__)+'\\zone\\'+'ZONE_'+str(int(r_L))+'_'+str(int(r_R))+'.fits',zone,overwrite=True)

    print("Zone evaluation done")
    return zone
    

def rescaling_hor(M_adjust,science,zone,edge=40):
    """ Effectuates rescaling of the sky matrix line by line
    Args:
        M_adjust,science: 2D (1024,2048) matrices : _adjust is a sky matrix
        zone: 2D (1024,2048) zone matrix
    Returns:
        2D rescaled sky matrix  """
    dimx,dimy=science.shape

    M_coeff=np.empty((dimx,dimy),dtype=float)
    M_coeff[:]=np.NaN 

      

    cor_y, cor_x=np.where(zone==1)   # collect pixels of the background zone only

    for i in range(0,len(cor_y)):
            y,x=cor_y[i],cor_x[i]

            if M_adjust[y][x]!=0 :  
                M_coeff[y][x]=science[y][x]/M_adjust[y][x]  #Coeff is such that science[y][x]- M_adjust[y][x] =0

   
    M_med=np.nanmedian(M_coeff,axis=1) # median by each column
    
    return (M_adjust.T*M_med).T



def rescaling(M_adjust,science,zone,edge=40):
    """ Effectuates rescaling of the sky matrix column by column
    Args: 
        M_adjust,science: 2D (1024,2048) matrices : _adjust is a sky matrix
        zone: 2D (1024,2048) zone matrix
    Returns:
        2D rescaled sky matrix  """
    dimx,dimy=science.shape

    M_coeff=np.empty((dimx,dimy),dtype=float)
    M_coeff[:]=np.NaN 

      

    cor_y, cor_x=np.where(zone==1)   # collect pixels of the background zone only

    for i in range(0,len(cor_y)):
            y,x=cor_y[i],cor_x[i]

            if M_adjust[y][x]!=0 :  
                M_coeff[y][x]=science[y][x]/M_adjust[y][x]  #Coeff is such that science[y][x]- M_adjust[y][x] =0

   
    M_med=np.nanmedian(M_coeff,axis=0) # median by each column
   
    return M_adjust*M_med


#### PCA ###################################################################################################################

def get_flatten_matrix(matrix,y1,y2,x1,x2):
    """Flatten a 2D matrix to 1D and exclude border zone (as this zone intensity is highly variable; PC are less suitable for the center of the image)
    Args:
        matrix: 2D matrix , y1,y2,x1,x2: integers
    Returns:
        np.array (M.flatten()): 1D array """

    M=np.copy(matrix)
    M=M[y1:y2]
    M=M.T[x1:x2].T
   

    return np.array (M.flatten())


def compress_flatten(reconstructed,y1,y2,x1,x2):
    """reconstruct 2D matrix form  1D matrix. Border area is fixed to 0
    Args:
        reconstructed: 1D array, 
        y1,y2,x1,x2: integers
    Returns:
        2D array"""
    C1=np.zeros((1024,1024))     
    for i in range (0,y2-y1):

        a=i*(x2-x1)
        m1=combine_LR(np.zeros(x1),reconstructed[a:a+x2-x1].flatten()) 

        C1[y1+i]=combine_LR(m1,np.zeros(1024-x2)) 

    return C1


def PCA(M_sky,image_science,zone_flat,n_comp,type_coeff,mean_n):  #M_sky (nb sky x nb px)
    """perform PCA
    Args: 
        M_sky: 2D array (y,x) where each line y is a 1D flatten sky and colums x correspondes to pixels indices,
        image_science: 1D array, zone_flat: 1D array,  
        n_comp: number of components used for projection, 
        type_coeff: 'norm'/'mean'/area' : defines procedure of renormalising PC, star zone excluded
        mean_n: number of the sky images from the sorted list used for centring science 
    Returns:
        reconstructed: 1D reconstructed sky matrix ,
        M_med : coefficient of rescaling forcing 0 median on the background zone (M_med=sky reconstructed/science)

    """
    ########   Covariance - eigenvectors  ##########################################
    
    F_mean_sky = np.mean(M_sky,axis=0) #centers sky data to 0
    #F_mean_sky = np.mean(M_sky[1:],axis=0) #usun
    
    X=M_sky-F_mean_sky  #center sky data
    #X=M_sky[1:]-F_mean_sky ##usun
    
    Cov = np.dot(X,X.T)  #Covariance matrix: X*X.T is faster than X.T*X. Base is changed back to correct dim. by V = np.dot(eigenvect.T, X)
    Cov = (Cov + Cov.T) / 2   #eliminates numeric errors 
    eigenval, eigenvect = np.linalg.eigh(Cov)  
    eigenval = eigenval[::-1]  #List order is inversed: initially important eigenvectors (associeted with minimal eigenvalues - minimal variance) were at the end. 
    eigenvect = eigenvect[:, ::-1]

    eigenvect= eigenvect/np.sqrt(np.abs(eigenval)) # eigenvectors are normalized to 1


    ### Plot eigenvalues of components (variance associated to vectors)
    """tot=sum(eigenval)
    n_comp=0  #prints variance percent
    for i in range (0,len(eigenval)):
        variance_per=sum(eigenval[:i]/tot)
        #print(i,variance_per)
        if n_comp==0 and variance_per>0.99:
            
            n_comp=i+1
    print("n_comp",n_comp)
    
    plt.plot(eigenval/tot,color='red',label='eigenval')  #sky_med
    plt.legend(loc=1) #upper left
    plt.title("Valeurs propres PCA sur toute l'image HD14 (variance)") 
    plt.xlabel("Indice composante")
    plt.ylabel("% variance totale")
    plt.show()"""
    
    
    V = eigenvect[:, :]   
    V = np.dot(eigenvect.T, X)  #Back to the pixels base
    # Select the top n_comp principal components 
    V = V[:n_comp]
    """f=[]   #save principal components
    print("V dim",V.shape)
    for i in V[:15]:
        f.append(compress_flatten(i,40,964,41,901))

    fits.writeto('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir 2/PCA_K12/results/composantes_expdiff_bezpose.fits',np.array(f),overwrite=True)
    sys.exit()"""


    ########   Projection  #########################################################################
    #F_mean_science=np.mean(M_sky[:mean_n],axis=0) #centering with 1 sky: M_sky[:1]
    #image_science_cent=image_science  - F_mean_science #centers science (only few first skies are sufficiently compatible to center science on 0)

    
    alpha=np.nanmean( image_science / F_mean_sky )
    F_mean_science=np.mean(M_sky[:mean_n],axis=0) #centering with 1 sky: M_sky[:1]
    image_science_cent= image_science  - alpha * F_mean_science #centers science (only few first skies are sufficiently compatible to center science on 0)


    
    """w= sliding_median(F_mean_science, 5).tolist()
    liste=[]
    liste.append(w[0])
    liste.append(w[0])
    for i in range(0,len(w)):
        liste.append(w[i])
    liste.append(w[-1])
    liste.append(w[-1])
    image_science_cent=image_science  - np.array(liste) #centers science (only few first skies are sufficiently compatible to center science on 0)
    """
    cor=np.where(zone_flat==0) # gets coordinates of star zone (pixels to remove)
    try:
        pct=1-len(cor[0])/len(zone_flat)  # gets percent of area optimization zone/total
    except:
        pct=1

    
    im_s=np.delete(np.copy(image_science_cent),cor[0])  # removes pixels of star zone form science
   

    pc_s=np.zeros((V.shape[0],V.shape[1]-len(cor[0])))   
    for i in range(0,len(V)):
        pc_s[i]=np.delete(np.copy(V[i]),cor[0])     # removes pixels of star zone form sky    



    projection= np.dot(im_s, pc_s[:n_comp].T)  # pc   # Calculates projection coefficients on  matrices without star zone

    ### Renormalisation of projection coefficients 

    if type_coeff=="norm":
        # norm total:  sum of squares of pixels values
        norme_pc=np.dot(V[:n_comp], V[:n_comp].T)
        norme_pc_s=np.dot(pc_s[:n_comp], pc_s[:n_comp].T)   
        beta_norme=[]  #list of coefficients of renormalization |PCi|/|PCi without star zone|
        for n in range(0,len(norme_pc)):
            beta_norme.append(np.sqrt(norme_pc[n].T[n])/np.sqrt(norme_pc_s[n].T[n]))


    elif type_coeff=="mean":
        # norm total: mean of clipped pixels * number of pixels
        norme_pc_s=[]   
        norme_pc=[] 
        for i in range(0,n_comp):
            N=len(pc_s[i])  #number of pixels after removal
            N_r=len(V[i])   #number of pixels before removal
            real_s=sigma_clipped_stats(np.square(np.copy(pc_s[i])),sigma=30)[0]*N 
            real=sigma_clipped_stats(np.square(np.copy(V[i])),sigma=30)[0]*N_r
            norme_pc_s.append(real_s) 
            norme_pc.append(real)

        beta_norme=[]
        for n in range(0,len(norme_pc)):
            beta_norme.append(np.sqrt(norme_pc[n])/np.sqrt(norme_pc_s[n]))

    elif type_coeff=="area":
        beta_norme=[]
        for n in range(0,n_comp):
            beta_norme.append(1/pct)

    else:
        print("type_coeff should be: 'area', 'mean' or 'norm'")
        

    
    projection_normal=[]    # renormalised coefficients
    for i in range(0,len(pc_s)):
        projection_normal.append(projection[i]*beta_norme[i])

    ### Plot coefficients of projection
    """plt.plot([abs(i) for i in projection_normal],color='purple',label='coeff projection (renormalises)')  #sky_med
    plt.legend(loc=1) #upper left
    plt.title("Coeff de projection renormalises PCA sur image totale HD14") 
    plt.xlabel("Indice composante")
    plt.ylabel("Valeur coeff")
    plt.show() """
    
    reconstructed = np.dot(np.array(projection_normal).T, V[:n_comp]) + F_mean_science  #Multiplies PC by coefficients and uncenteres

    ### Calculs of rescaling coefficiants
    M_coeff=np.zeros(len(im_s))  #first coefficients for each pixels are calculated


    sc=np.delete(np.copy(image_science),cor[0]) #  an uncentered science matrix without star zone
    M_adjust=np.delete(np.copy(reconstructed),cor[0]) # reconstructed sky matrix without star zone
    for x in range(0,len(sc)):
        M_coeff[x]=sc[x]/M_adjust[x]

    M_med=np.nanmedian(M_coeff) # M_med is an integer

    return reconstructed, M_med


def wrap(matrix,start_x,end_x,start_y,end_y,value="nan"):
    """replace border area of final matrix by nans"""

    M=np.zeros((1024,1024))
    M[:start_y]=float(value)
    M[end_y:]=float(value)
    M.T[:start_x]=float(value)
    
    if matrix.shape==(1024,1024):
        M[start_y:end_y].T[start_x:end_x]=matrix[start_y:end_y].T[start_x:end_x]
        M.T[end_x:]=float(value)
    elif M[start_y:end_y].T[start_x:end_x].shape==matrix.T.shape:
        M[start_y:end_y].T[start_x:end_x]=matrix.T
        M.T[end_x:]=float(value)
    else:
        print("Couldn't wrap matrix correctly")
        M=matrix

    return M


def division_sliding_rect(M,width_rect,i):
    """divide matrix into vertical rectangles for sliding rectangles PCA
    Args: 2D array ,width_rect : number of pixels of the width of the rectangle , i : current lag of first rectangle from the left border of the matrix
    OUTPUT: list of  1D arrays - image slices"""
    a=M.shape[1]  #x dimension
    n=(a-i)/width_rect

    if int(n)>n:
        m=int(n)-1  # m: number of full size rectangles
    else:
        m=int(n)

    liste_rect=[]
    if M.T[0:i].T.flatten().shape[0]!=0:  # collect a first rectangle (from left) of reduced size
        liste_rect.append(M.T[0:i].T.flatten())

    for j_rect in range(0,m):
       
        liste_rect.append(M.T[i+width_rect*j_rect:i+width_rect*(j_rect+1)].T.flatten())

    if m*width_rect+i<a:
       
        liste_rect.append(M.T[m*width_rect+i:].T.flatten())

    return liste_rect


def merge_sliding_rect(liste,M,width_rect,i):
    """Merges rectangular slices of a matrix into an image
    INPUT: liste: list of 1D arrays of image slices
    M : initial matrix that has been devided, width_rect: value of recnagle width, i:  loop step
    OUTPUT: A: 2D matrix"""
    h_tot,a=M.shape[0],M.shape[1]  # h : y dim  a: x dim
    n=(a-i)/width_rect
    if int(n)>n:
        full_rect=int(n)-1  # number of full rectangles

    else:
        full_rect=int(n)

    width0=i   #width of the first reduced size rectangle (could be 0 if first rectangle is of the full size)
    if int(n)==n: ##co  i==0 and 
        width_last=a-((full_rect-1)*width_rect+i)   #width of the first reduced size rectangle ####
    else:
        width_last=a-((full_rect)*width_rect+i)


    

    l_f=[[] for i in range(0,len(liste))]  # list of 2D np array (rectangles): each  list correspond to one rectangle
    M_f=np.zeros(M.shape)

    for n_rect in range(0,len(liste)):
        liste[n_rect]=liste[n_rect].tolist() #should be list not a np array

    for h in range(0,h_tot): # for each "row"

        nb_first=0

        if width0!=0:
            l_f[0].append(liste[0][h*width0:width0*(h+1)]) #transcription of the first rectangle (if the size is reduced)
            nb_first=1 # full rectangle transcription start from the second list element


        for m in range(nb_first,len(liste)-1):
            l_f[m].append(liste[m][h*width_rect:(h+1)*width_rect])   #transcription of all full rectangles

        l_f[-1].append(liste[-1][h*width_last:(h+1)*width_last])  #transcription of the last rectangle


    
    for m in range(0,len(l_f)): #should be np array not a list

        l_f[m]=np.array(l_f[m])


    A=combine_LR(l_f[0],l_f[1])
    for i in range(2,len(l_f)):
        A=combine_LR(A,l_f[i])

   
    return A


def pca_sliding_rectangle( zone, science, cubes, width_rect, n_comp, start_x, end_x, start_y, end_y, filt, type_coeff, step, mean_n, margins, path_save='0',resc_rect=False):
    """Calculates principal components on sliding rectangular slices of width width_rect (either left or part of the image K1/K2)
    Rectangles are beeing shifted, each time PCA is calculated for each rectangle. In the end all obtained matrices of differant rectangles configuration are averaged
    INPUT: zone: 2D array , science: 2D array , cubes : 3D skies array, width_rect: width of rectangle in pixels, n_comp: number of PCA components to project on, 
    start_x, end_x, start_y, end_y, : integers - border area to exclude, filt: 'K1/'K2' left part or right part of the image is reduced, 
    type_coeff: 'norm'/'mean'/area' : defines procedure of renormalising PC, star zone excluded, path_save: path. if not '0' reconstructed sky will be saved, 
    resc=True/False: rescaling of each slice, forcing total median of final image slice to be 0. Not recommended for rectangular division
    OUTPUT: rec_sky: 2D matrix of reconstructed sky"""
    cubes_t=cubes.copy()
    #sky=fits.getdata('c:/Users/klara/OneDrive/Pulpit/Stage fits/AB auriga/RES DLA P/save_sky_hip201.fits')
    
    #np.insert(cubes, 0 , sky, axis=0)   #usun
    
    science_t=science.copy()
   
    zone_t=zone.copy()


    if filt=="K2":
        zone_c=zone_t.T[1024:].T[start_y:end_y].T[start_x:end_x].T
        science_c=science_t.T[1024:].T[start_y:end_y].T[start_x:end_x].T
 
    elif filt=="K1":
        zone_c=zone_t.T[:1024].T[start_y:end_y].T[start_x:end_x].T
        science_c=science_t.T[:1024].T[start_y:end_y].T[start_x:end_x].T
    
 
    else:
        print("Enter filter: 'K1' or 'K2'")

   
    matrices_liste=[]
    
    for i_px in range(0,width_rect,step): # for each PCA rectangles are shifted by one pixel. Precess goes on until rectangle border end up in the same place as at i=0 

        liste_zone=division_sliding_rect(zone_c,width_rect,i_px)
        liste_sc=division_sliding_rect(science_c,width_rect,i_px)
        #print("division rectangle",len(liste_sc), liste_sc[0].shape,liste_sc[-1].shape)
        cubes_liste=[[] for i in range(0,len(liste_sc))]  #list of list of devided cubes (if n rectangles, list contains n lists)
        cubes_t2=[]

        for g in range(0,len(cubes_t)):

            if filt=="K2":
                cubes_t2.append(cubes_t[g].T[1024:].T[start_y:end_y].T[start_x:end_x].T) 
            elif filt=="K1":
                cubes_t2.append(cubes_t[g].T[:1024].T[start_y:end_y].T[start_x:end_x].T)

            liste_rect=division_sliding_rect(cubes_t2[g],width_rect,i_px)

            for j in range(0,len(cubes_liste)):
                cubes_liste[j].append(np.array(liste_rect[j])) # each rectangle is atributed to the correct cubes_liste sub list
       
        liste_fin=[]
        
        # 4 differant boundry cases. 1st case: first and last rectangle are of full size (of width_rect)
        if liste_sc[0].shape==liste_sc[1].shape==liste_sc[-1].shape:  #PCA is calculated on all the rectangles
            #print("case 1")
            for i_rect in range (0,len(liste_sc)):
      
                new,coeff=PCA(cubes_liste[i_rect],liste_sc[i_rect],liste_zone[i_rect],n_comp,type_coeff, mean_n)#[0]
                if not resc_rect:
                    coeff=1
                liste_fin.append(new* coeff)

            M_f=merge_sliding_rect(liste_fin,science_c,width_rect,i_px)
            matrices_liste.append(M_f)

        #2nd case: first rectangle is smaller than middle rectangles, PCA will not be calculated for this rectangle
        elif liste_sc[0].shape!=liste_sc[1].shape and liste_sc[1].shape==liste_sc[-1].shape:  #PCA is calculated on all the rectangles apart the first (smaller rectangle)
            #print("case 2")
            liste_fin.append(np.full(liste_sc[0].shape,float("nan")))
            for i_rect in range (1,len(liste_sc)):
                new,coeff=PCA(cubes_liste[i_rect],liste_sc[i_rect],liste_zone[i_rect],n_comp,type_coeff,mean_n)#[0]
                if not resc_rect:
                    coeff=1
                liste_fin.append(new* coeff)

            M_f=merge_sliding_rect(liste_fin,science_c,width_rect,i_px)
            matrices_liste.append(M_f)

        #3rd case: last rectangle is smaller than middle rectangles, PCA will not be calculated for this rectangle
        elif liste_sc[0].shape==liste_sc[1].shape and liste_sc[1].shape!=liste_sc[-1].shape:  #PCA is calculated on all the rectangles apart the last (smaller rectangle)
            #print("case 3")
            for i_rect in range (0,len(liste_sc)-1):
                new,coeff=PCA(cubes_liste[i_rect],liste_sc[i_rect],liste_zone[i_rect],n_comp,type_coeff,mean_n)#[0]
                if not resc_rect:
                    coeff=1
                liste_fin.append(new* coeff)
            liste_fin.append(np.full(liste_sc[-1].shape,float("nan")))

            M_f=merge_sliding_rect(liste_fin,science_c,width_rect,i_px)
            matrices_liste.append(M_f)

        #both last and first  rectangle are too small
        else:  #PCA is calculated on all the rectangles apart the last (smaller rectangle)
            #print("case 4")
            liste_fin.append(np.full(liste_sc[0].shape,float("nan")))
            for i_rect in range (1,len(liste_sc)-1):
                new,coeff=PCA(cubes_liste[i_rect],liste_sc[i_rect],liste_zone[i_rect],n_comp,type_coeff,mean_n)#[0]
                if not resc_rect:
                    coeff=1
                liste_fin.append(new* coeff)
            liste_fin.append(np.full(liste_sc[-1].shape,float("nan")))

            M_f=merge_sliding_rect(liste_fin,science_c,width_rect,i_px)
            
            matrices_liste.append(M_f)
        print(filt, "PCA done:", i_px, "rectangle")
        #print("PCA finished: slice", i )

    
    print("PCA finished",filt)

    M_sky_combined=get_compressed_matrix_mean(matrices_liste,y=matrices_liste[0].shape[0],x=matrices_liste[0].shape[1])
    

    if margins: #pca is effectuated independently on marginal area
        M_sky_combined=margins_pca( M_sky_combined, zone_t, science_t, cubes_t, filt, start_y,end_y,start_x,end_x,n_comp,type_coeff, mean_n)
    else: #marginal area is replaced with nans
        M_sky_combined=wrap(M_sky_combined,start_x,end_x,start_y,end_y,value="nan")

    return M_sky_combined


def flatten_margins(M,filt,start_y,end_y,start_x,end_x):
    """ flatten border region to 1D matrices
    Args: 
        M: 2D array, filt: 'K1'/'K2'
        start_y,end_y,start_x,end_x: integers
    Returns:
        list of 1D arrays"""

    if filt=="K2":
        M=M.T[1024:].T
    else:
        M=M.T[:1024].T

    M_up=M[:start_y].flatten()  
    M_down=M[end_y:].flatten()
    M_left=M[start_y:end_y].T[:start_x].T.flatten()
    M_right=M[start_y:end_y].T[end_x:].T.flatten()

    margins_K12=compress_margins(np.zeros((end_y-start_y,end_x-start_x)),[M_up,M_down,M_left,M_right],start_y,end_y,start_x,end_x)
    
    return [M_up,M_down,M_left,M_right]


def compress_margins(M_sky_combined, liste_fin, start_y, end_y, start_x, end_x):
    """combines reconstructed central area and reconstructed margins
    Args:
        M_sky_combined: 2D array, 
        liste_fin: list of 4 1D arrays of margin areas, 
        start_y, end_y, start_x, end_x : integers
    Returns:
        recombined matrix"""

    C1=np.zeros((1024,1024))  #924 od 41 do 901  jest 860


    for i in range (0,start_y): #up
        a=i*1024
        C1[i]=liste_fin[0][a:a+1024]

    
    for i in range (0,1024-end_y): #down
        a=i*1024
        C1[end_y+i]=liste_fin[1][a:a+1024]
       
    
    for i in range (start_y,end_y): #left
        a=(i-start_y)*start_x
        #sys.exit()
        b=(i-start_y)*(1024-end_x)  # brakuje dokładnie 40 linii
        c=i-start_y

        C1[i]=combine_LR(liste_fin[2][a:a+start_x],combine_LR(M_sky_combined[c],liste_fin[3][b:b+(1024-end_x)]))


    return C1


def margins_pca(M_sky_combined, zone_t, science_t, cubes_t, filt, start_y,end_y,start_x,end_x,n_comp,type_coeff, mean_n):
    """effectuates pca of margins region. Pca is effectuated independently of the pca of central area.
    Args: 
        M_sky_combined: 2D array of the cenral area, zone_t: 2D array, 
        science_t: 2D array, cubes_t: 2D array, filt: 'K1'/'K2', 
        start_y,end_y,start_x,end_x: integers ,
        n_comp: number of used principal components,
        type_coeff: type of method evaluating projection coefficients in pca, 
        mean_n: number of sky images used for centring science 
    Returns: 
        recombined image of reconstructed cenrtral area and margins  """

    if filt=="K2": #on K1 or on K2

        liste_fin=[]
        cubes_liste=[[],[],[],[]]
        zone_liste=flatten_margins(np.ones(zone_t.shape),"K2",start_y,end_y,start_x,end_x)
        science_liste=flatten_margins(science_t,"K2",start_y,end_y,start_x,end_x)

        for g in range(0,len(cubes_t)):
            rep=flatten_margins(cubes_t[g],"K2",start_y,end_y,start_x,end_x)
            for i in range(0,4):
                cubes_liste[i].append(rep[i])

        for j in range(0,4):
            new,coeff=PCA( cubes_liste[j], science_liste[j], zone_liste[j], n_comp, type_coeff, mean_n)
            liste_fin.append(new*coeff)

    elif filt=="K1":
       
        liste_fin=[]
        cubes_liste=[[],[],[],[]]
        zone_liste=flatten_margins(np.ones(zone_t.shape),"K1",start_y,end_y,start_x,end_x)


        science_liste=flatten_margins(science_t,"K1",start_y,end_y,start_x,end_x)

        for g in range(0,len(cubes_t)):
            rep=flatten_margins(cubes_t[g],"K1",start_y,end_y,start_x,end_x)
            for i in range(0,4):
                cubes_liste[i].append(rep[i])


        for j in range(0,4):
          
            #print(cubes_liste[j][i].shape,science_liste[i].shape)
            new,coeff=PCA( cubes_liste[j], science_liste[j], zone_liste[j], n_comp, type_coeff, mean_n)
        
            liste_fin.append(new*coeff)


    margins_comp=compress_margins(M_sky_combined,liste_fin,start_y,end_y,start_x,end_x)
    
    """margins_K1=compress_margins(liste_fin,start_y,end_y,start_x,end_x)

        margins_K12=combine_LR(margins_K1,margins_K2)
        fits.writeto('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir/funcs/results/marg.fits',margins_K12,overwrite=True)
    sys.exi()"""
    
    return margins_comp



def exe_pca( science, zone, cubes_inp, n_comp, start_x, end_x, 
             start_y, end_y, path_save_sky, path_save_final, header, type_coeff, type_div, n_rect, width_rect, margins, diff_exptime,
             resc_vert, bd_subs, save_, mean_n, select_auto, step):
    """Executes PCA.  
    Args: science: 2D array, zone : 2D array     cubes_inp: ordered list of sky images (2D arrays) used for PCA
    n_comp: number of components used for projection coefficients calculation
    start_x, end_x, start_y, end_y:  defines zone of intrest on which PCA is effectuated 
    path_save_sky, path_save_final: paths where final files will be saved 
    type_coeff: projection coefficients are renormalized by pc's norm: coeff*|PC|/|PC without star zone| 
        'norm': |PC without star zone| is obtain by a sum of the square of each pixel
        'mean': |PC without star zone| is obtain by a sum of the average squared value of a pixel 
        'area': |PC|/|PC without star zone|= PC pixels number/PC without star zone pixels number
    type_div: type of division
        'rectangle': divistion by vertical rectangles  n_rect: number of rectangles: must be divisor of end_x-start_x
        'rectangle hor': division by horizontal symmetry axe and in vertical rectangles   n_rect: total number of rectangles 
        'tringle': by 8 triangles with commmon vertex in the image center
        'siliding_rect': divistion by vertical rectangles  n_rect: number of rectangles: must be divisor of end_x-start_x

        if other: PCA on the whole image (K1/K2 separetly)
    n_rect: number of rectangles if type_div='rectangle',
    width_rect: width of rectangles if type_div='salding_rect', 
    margins: True/False. True: pca is calculated on margins area, 
    diff_exptime: True/False. True: science and sky files should be renormalised to the initial exposure time,
    resc_vert: True/ False. True: reconstructed sky image is rescaled to science column by column , 
    bd_subs: True/False. True: bad pixels are replaced by nans in the final image, 
    save_: True/False, 
    mean_n: int, number of sky files used for centring science,
    select_auto=True/False, 
    step=10"""

    cubes=cubes_inp.copy()
    ##############################################################
    if select_auto:   #evaluates the optimal mean_n (number of sky matrices used for centring science) value
        print('mean_n check')
        plot_all=True
        mean_n=centring_science(zone, science, cubes, width_rect, n_comp, bd_subs, resc_vert, start_x, end_x, start_y, end_y, type_coeff, step, mean_n, margins, plot_all, path_save='0',resc=False)
    
    if type_div=="sliding_rect":
        
        new_sky_pca_L=pca_sliding_rectangle( zone, science, cubes, width_rect, n_comp, start_x, end_x, start_y, end_y, 'K1', type_coeff,step, mean_n, margins, path_save='0', resc_rect=False)
        new_sky_pca_R=pca_sliding_rectangle( zone, science, cubes, width_rect, n_comp, start_x, end_x, start_y, end_y, 'K2', type_coeff, step,mean_n, margins, path_save='0',resc_rect=False)


    else:   #PCA calculated on whole image (1024,1024)
        sc_K1=get_flatten_matrix(science.T[:1024].T,start_y, end_y, start_x, end_x)
        sc_K2=get_flatten_matrix(science.T[1024:].T,start_y, end_y, start_x, end_x)

        image_matrix_K1=np.array([get_flatten_matrix(cubes[i].T[:1024].T,start_y, end_y, start_x, end_x) for i in range(0,len(cubes))])
        image_matrix_K2=np.array([get_flatten_matrix(cubes[i].T[1024:].T,start_y, end_y, start_x, end_x) for i in range(0,len(cubes))])
 

        zone_K1=get_flatten_matrix(zone.T[:1024].T,start_y, end_y, start_x, end_x)
        zone_K2=get_flatten_matrix(zone.T[1024:].T,start_y, end_y, start_x, end_x)

        new_sky_pca_L=compress_flatten(PCA(image_matrix_K1,sc_K1,zone_K1,n_comp,type_coeff, mean_n)[0],start_y, end_y, start_x, end_x)
        new_sky_pca_L=wrap(new_sky_pca_L,start_x,end_x,start_y,end_y,value="nan")
        print("PCA K1 done")
        new_sky_pca_R=compress_flatten(PCA(image_matrix_K2,sc_K2,zone_K2,n_comp,type_coeff, mean_n)[0],start_y, end_y, start_x, end_x)
        new_sky_pca_R=wrap(new_sky_pca_R,start_x,end_x,start_y,end_y,value="nan")
        print("PCA K2 done")
   
    ##########################
    
    new_sky_pca=combine_LR(new_sky_pca_L,new_sky_pca_R)
    
    if diff_exptime:
        new_sky_pca=(header['EXPTIME'])*new_sky_pca #+sky_pose
        science=(header['EXPTIME'])*science #+sky_pose

    if resc_vert:

        #If pca was effectuated also on the margins. 
        A,B,C=np.ones((end_y-start_y,start_x)),np.ones((end_y-start_y,1024-end_x+start_x)),np.ones((end_y-start_y,1024-end_x))
        middle_z=combine_LR(A,combine_LR(zone[start_y:end_y].T[start_x:end_x].T,combine_LR(B,combine_LR(zone[start_y:end_y].T[start_x+1024:end_x+1024].T,C))))
        zone_resc=np.concatenate((zone[:start_y],middle_z,zone[end_y:]))
        new_sky_pca=rescaling(new_sky_pca,science,zone_resc,edge=40)
        
        print("Rescaling by vertical coefficients")

    if save_:  #exptime?
        fits.writeto(str(path_save_sky),new_sky_pca,overwrite=True)

    final_pca=science-new_sky_pca 

    if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:
        print("bd px subs")
        bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
        indexes_bd=bd_px_index(bd_px_cube)
        final_pca=bd_px_subs_nan(indexes_bd,final_pca) 
    else:
        print("bad pixels are not substracted")
    
    if save_:
        fits.writeto(str(path_save_final),final_pca,header,overwrite=True)

    print("####### PARAMETRES #######","\n")
    print(type_coeff, type_div, n_rect, path_save_final,"\n")
    print_med_tot(zone,final_pca)
    print("\n")
    return final_pca



def centring_science(zone, science, cubes, width_rect, n_comp, bd_subs, resc_vert, start_x, end_x, start_y, end_y, type_coeff, step, mean_n, margins, plot_all, path_save='0',resc=False):
    
    med_val=[]
    dev_val=[]

    for n in range (1,6): #pca with science cetntered with 1, 2... sky images
        mean_n=n
        new_sky_pca_L_0=pca_sliding_rectangle( zone, science, cubes, width_rect, n_comp, start_x, end_x, start_y, end_y, 'K1', type_coeff, step, mean_n, margins, path_save='0')
        new_sky_pca_R_0=pca_sliding_rectangle( zone, science, cubes, width_rect, n_comp, start_x, end_x, start_y, end_y, 'K2', type_coeff, step, mean_n, margins, path_save='0')
        new_sky_pca_0=combine_LR(new_sky_pca_L_0,new_sky_pca_R_0)

        if resc_vert:
            new_sky_pca=rescaling(new_sky_pca_0,science,zone,edge=40)

        final_pca=science-new_sky_pca 

        if bd_subs and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:
            print("bd px subs")
            bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
            indexes_bd=bd_px_index(bd_px_cube)
            final_pca=bd_px_subs_nan(indexes_bd,final_pca)

        med_val.append(dev_on_subspaces(final_pca,zone)[0]) #median of the 50x50 px cubes in the background zone
        dev_val.append(dev_on_subspaces(final_pca,zone)[1])
        print( n, "MED:", med_val[-1])
        fits.writeto('c:/Users/klara/OneDrive/Pulpit/Stage fits/AB auriga/RES DLA P/save_sky_hip20'+str(n)+'.fits',new_sky_pca,overwrite=True)
        fits.writeto('c:/Users/klara/OneDrive/Pulpit/Stage fits/AB auriga/RES DLA P/save_hip20'+str(n)+'.fits',final_pca,overwrite=True)
        

    rep=len(med_val)
    i=0
    while rep==len(med_val) and i<len(med_val)-1:  
        #parameter is std dev of medians of cubes 50x50px of reduced science
        #mean_n will stop incresing if parameter gets 1.8 bigger when the next sky in the sorted list is included in the science centring
        print("tu", med_val[i], med_val[i+1],1.8*med_val[i] < med_val[i+1])
        if 1.8*med_val[i] < med_val[i+1]:
            rep=i
        i=i+1

    print("selected mean_n (number of sky images used for centering): ", rep)

    if plot_all:    
        plt.plot([n for n in range(1,len(med_val)+1)],med_val,color='red')   
        plt.title("Std dev of the medians of  50 x 50 pixels squares in the background zone")
        plt.xlabel("Number of sky images used for centring science")
        plt.show()

        plt.plot([n for n in range(1,len(med_val)+1)],dev_val,color='red')
        plt.xlabel("Number of sky images used for centring science")
        plt.title("Std dev in the background zone")
        plt.show()

    return rep

        

def mean_result_files(science,liste_sky,path_save,header=None, bd_subs='yes'):
    """ calculate mean of few  reconstructed sky files and substract it from science
    files must be placed in results directory
    science: 2D array, liste_sky: list of sky files names ,path_save : path where reduced science image will be saved ,bd_subs='yes'"""
    folder=os.path.dirname(__file__)+'\\results\\'

    cubes=[]
    for i in range (0,len(liste_sky)):
        data=fits.getdata(folder+str(liste_sky[i]))
        cubes.append(data)
    cube=get_compressed_matrix_mean(cubes)
    final_pca=science-cube
    

    if bd_subs=='yes' and len(os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\'))==1:
        print("bd px subs")
        bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
        indexes_bd=bd_px_index(bd_px_cube)
        final_pca=bd_px_subs_nan(indexes_bd,final_pca) 

    if header!=None:
        fits.writeto(str(path_save),final_pca, header, overwrite=True)
    else:
        fits.writeto(str(path_save),final_pca, overwrite=True)

    return final_pca
    

def dev_on_subspaces(M_1,zone):
    M1_s=zone_subs(M_1.copy(),zone)
    med1=[]
    flux=[]
    n=40
    m=20 #sur y
    for i in range(0,n):
        for j in range (0,m):
            med1.append(np.nanmedian(M1_s[j*int(1024/m):int(1024/m)*(1+j)].T[int(2048/n)*i:int(2048/n)*(i+1)].T))
            flux.append(np.nansum([a*a for a  in M1_s[j*int(1024/m):int(1024/m)*(1+j)].T[int(2048/n)*i:int(2048/n)*(i+1)].T]))
    print("M1",np.nanstd(med1))
    
    #print("min/max:",[i/max(flux) for i in flux])
    """M=np.copy(Matrix)

    cor_y, cor_x=np.where(zone==0)
    for i in range(0,len(cor_y)):
        M[cor_y[i]][cor_x[i]]=float("nan")

    result = M.flatten()"""
    return np.nanstd(med1),np.nanstd(M1_s)
 



