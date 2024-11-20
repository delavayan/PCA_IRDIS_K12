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

import functions
from functions import get_bdpx_from_sky, cubes_extraction_ , get_sorted_sky_list , zone_by_integration, SCIENCE_cubes_extraction_list, SKY_cubes_extraction_list,bd_px_subs_nan,bd_px_index, prepare_Mdist_from_raw, create_new_bdpx_map, rename_files
from functions import exe_pca, cubes_extraction_exptime_lin, get_sky_smoothed, get_compressed_matrix_mean, zone_by_dispertion, combine_LR, print_med_tot,collect_header,get_zone2, rescaling, cubes_extraction_exptime, cubes_extraction_all


def main():
  
    start_time = time.perf_counter()

    header, h_object = collect_header()   # h_objet is object name used for reduced science file name

    start_x=41  #PC will be calculated on the matrix M [start_y: end_y] [start_x: end_x] for both K1 and K2
    end_x=901 
    start_y=40  
    end_y=964  

    
    ### extraction of all files placed in raw directory :
    # if sky files should be of the same exposure time as science:
    #cubes, science, sky_files_list, science_files_list, diff_exptime = cubes_extraction_exptime(compression='mean',bd_subs=True)  # cubes: (nb sky, y dim, x dim)   science: ( y dim, x dim)
    
    # if sky files of different exposure time should be normalized to the common base:
    cubes, science, sky_files_list, science_files_list, diff_exptime = cubes_extraction_all(compression='mean',bd_subs=True)

    # Normalisation to the same base by linear regression (in construction):
    #cubes, science, sky_files_list, science_files_list, diff_exptime=cubes_extraction_exptime_lin(compression='mean',bd_subs='yes')
 
    ### extraction with files names lists
    """files_list_sorted=['sky_27.fits', 'sky_88.fits', 'sky_590.fits', 'sky_257.fits', 'sky_557.fits', 'sky_644.fits', 'sky_357.fits', 'sky_90.fits', 'sky_335.fits', 'sky_660.fits', 'sky_640.fits', 'sky_22.fits', 'sky_457.fits', 'sky_26.fits', 'sky_75.fits', 'sky_20.fits', 'sky_105.fits', 'sky_652.fits', 'sky_468.fits', 'sky_10.fits', 'sky_668.fits', 'sky_423.fits', 'sky_346.fits', 'sky_83.fits', 'sky_24.fits', 'sky_639.fits', 'sky_58.fits', 'sky_179.fits', 'sky_97.fits', 'sky_70.fits', 'sky_401.fits', 'sky_5.fits', 'sky_103.fits', 'sky_157.fits', 'sky_623.fits', 'sky_113.fits', 'sky_501.fits', 'sky_94.fits', 'sky_301.fits', 'sky_98.fits', 'sky_86.fits', 'sky_64.fits', 'sky_645.fits', 'sky_77.fits', 'sky_651.fits', 'sky_646.fits', 'sky_89.fits', 'sky_643.fits', 'sky_25.fits', 'sky_446.fits', 'sky_6.fits', 'sky_312.fits', 'sky_2.fits', 'sky_102.fits', 'sky_29.fits', 
        'sky_235.fits', 'sky_612.fits', 'sky_57.fits', 'sky_17.fits', 'sky_11.fits', 'sky_101.fits', 'sky_80.fits', 'sky_224.fits', 'sky_641.fits', 'sky_16.fits', 'sky_665.fits', 'sky_534.fits', 'sky_479.fits', 'sky_290.fits', 'sky_545.fits', 'sky_35.fits', 'sky_279.fits', 'sky_650.fits', 'sky_33.fits', 'sky_92.fits', 'sky_12.fits', 'sky_135.fits', 'sky_87.fits', 'sky_38.fits', 'sky_19.fits', 
        'sky_653.fits', 'sky_637.fits', 'sky_104.fits', 'sky_96.fits', 'sky_190.fits', 'sky_663.fits', 'sky_13.fits', 'sky_657.fits', 'sky_46.fits', 'sky_490.fits', 'sky_655.fits', 'sky_412.fits', 'sky_642.fits', 'sky_661.fits', 'sky_32.fits', 'sky_656.fits', 'sky_379.fits', 'sky_647.fits', 'sky_30.fits', 'sky_246.fits', 'sky_21.fits', 'sky_124.fits', 'sky_81.fits', 'sky_84.fits', 'sky_666.fits', 'sky_568.fits', 'sky_50.fits', 'sky_168.fits', 'sky_654.fits', 'sky_45.fits', 'sky_223.fits', 'sky_146.fits', 'sky_523.fits', 'sky_201.fits', 'sky_51.fits', 'sky_636.fits', 'sky_268.fits', 'sky_648.fits', 'sky_434.fits', 'sky_85.fits', 'sky_635.fits', 'sky_7.fits', 'sky_445.fits', 'sky_100.fits', 'sky_93.fits', 'sky_601.fits', 'sky_579.fits', 'sky_662.fits', 'sky_9.fits', 'sky_79.fits', 'sky_61.fits', 'sky_659.fits', 'sky_68.fits', 'sky_323.fits', 'sky_669.fits', 'sky_658.fits', 'sky_4.fits', 'sky_8.fits', 'sky_78.fits', 'sky_3.fits', 'sky_39.fits', 'sky_622.fits', 'sky_554.fits', 'sky_95.fits', 'sky_111.fits', 'sky_276.fits', 'sky_536.fits', 'sky_438.fits', 'sky_386.fits', 'sky_114.fits', 'sky_595.fits', 'sky_463.fits', 'sky_297.fits', 'sky_165.fits', 'sky_578.fits', 'sky_397.fits', 'sky_416.fits', 'sky_209.fits', 'sky_270.fits', 'sky_548.fits', 'sky_245.fits', 'sky_594.fits', 'sky_547.fits', 'sky_266.fits', 'sky_119.fits', 'sky_593.fits', 'sky_573.fits', 'sky_196.fits', 'sky_143.fits', 'sky_170.fits', 'sky_162.fits', 'sky_577.fits', 'sky_152.fits', 'sky_271.fits', 'sky_293.fits', 'sky_581.fits', 'sky_194.fits', 'sky_524.fits', 'sky_322.fits', 'sky_204.fits', 'sky_382.fits', 'sky_580.fits', 'sky_368.fits', 'sky_338.fits', 'sky_249.fits', 'sky_561.fits', 'sky_164.fits', 'sky_544.fits', 'sky_559.fits', 'sky_613.fits', 'sky_631.fits', 'sky_203.fits', 'sky_138.fits', 'sky_563.fits', 'sky_315.fits', 'sky_62.fits', 'sky_273.fits', 'sky_495.fits', 'sky_431.fits', 'sky_159.fits', 'sky_160.fits', 'sky_549.fits', 'sky_378.fits', 'sky_232.fits', 'sky_550.fits', 'sky_309.fits', 'sky_188.fits', 'sky_260.fits', 'sky_565.fits', 'sky_429.fits', 'sky_596.fits', 'sky_288.fits', 'sky_155.fits', 'sky_474.fits', 'sky_163.fits', 'sky_633.fits', 'sky_144.fits', 'sky_624.fits', 'sky_500.fits', 'sky_543.fits', 'sky_592.fits', 'sky_605.fits', 'sky_365.fits', 'sky_202.fits', 'sky_117.fits', 'sky_110.fits', 'sky_219.fits', 'sky_175.fits', 'sky_200.fits', 'sky_220.fits', 'sky_571.fits', 'sky_477.fits', 'sky_277.fits', 'sky_329.fits', 'sky_496.fits', 'sky_173.fits', 'sky_494.fits', 'sky_487.fits', 'sky_511.fits', 'sky_462.fits', 'sky_177.fits', 'sky_265.fits', 'sky_411.fits', 'sky_363.fits', 'sky_350.fits', 'sky_358.fits', 'sky_402.fits', 'sky_562.fits', 'sky_600.fits', 'sky_421.fits', 'sky_14.fits', 'sky_15.fits', 'sky_606.fits', 'sky_63.fits', 'sky_76.fits', 'sky_464.fits', 'sky_55.fits', 'sky_43.fits', 'sky_47.fits', 'sky_41.fits', 'sky_0.fits', 'sky_56.fits', 'sky_49.fits', 'sky_91.fits', 'sky_634.fits', 'sky_18.fits', 'sky_512.fits', 'sky_1.fits', 'sky_649.fits', 'sky_334.fits', 'sky_54.fits', 'sky_65.fits', 'sky_31.fits', 'sky_466.fits', 'sky_74.fits', 'sky_53.fits', 'sky_44.fits', 'sky_73.fits', 'sky_667.fits', 'sky_71.fits', 'sky_638.fits', 'sky_52.fits', 'sky_72.fits', 'sky_99.fits', 'sky_66.fits', 'sky_212.fits', 'sky_67.fits', 'sky_390.fits', 'sky_60.fits', 'sky_69.fits', 'sky_42.fits', 'sky_48.fits', 'sky_23.fits', 'sky_37.fits', 'sky_82.fits', 'sky_664.fits', 'sky_556.fits', 'sky_40.fits', 'sky_242.fits', 'sky_141.fits', 'sky_120.fits', 'sky_417.fits', 'sky_247.fits', 'sky_28.fits', 'sky_521.fits', 'sky_34.fits', 'sky_112.fits', 'sky_618.fits', 'sky_115.fits', 'sky_36.fits', 'sky_560.fits', 'sky_59.fits', 'sky_142.fits', 'sky_611.fits', 'sky_369.fits', 'sky_370.fits', 'sky_310.fits', 'sky_118.fits', 'sky_507.fits', 'sky_538.fits', 'sky_364.fits', 'sky_193.fits', 'sky_597.fits', 'sky_558.fits', 'sky_253.fits', 'sky_498.fits', 'sky_441.fits', 'sky_251.fits', 'sky_254.fits', 'sky_108.fits', 'sky_629.fits', 'sky_227.fits', 'sky_214.fits', 
        'sky_234.fits', 'sky_256.fits', 'sky_215.fits', 'sky_400.fits', 'sky_180.fits', 'sky_231.fits', 'sky_248.fits', 'sky_585.fits', 'sky_128.fits', 'sky_443.fits', 'sky_161.fits', 'sky_123.fits', 'sky_267.fits', 'sky_499.fits', 'sky_178.fits', 'sky_221.fits', 'sky_620.fits', 'sky_546.fits', 'sky_255.fits', 'sky_304.fits', 'sky_467.fits', 'sky_367.fits', 'sky_540.fits', 'sky_460.fits', 'sky_191.fits', 'sky_583.fits', 'sky_148.fits', 'sky_630.fits', 'sky_206.fits', 'sky_347.fits', 'sky_582.fits', 'sky_602.fits', 'sky_427.fits', 'sky_588.fits', 'sky_586.fits', 'sky_440.fits', 'sky_442.fits', 'sky_607.fits', 'sky_158.fits', 'sky_625.fits', 'sky_497.fits', 'sky_129.fits', 'sky_426.fits', 'sky_459.fits', 'sky_509.fits', 'sky_149.fits', 'sky_156.fits', 'sky_226.fits', 'sky_199.fits', 'sky_313.fits', 'sky_228.fits', 'sky_419.fits', 'sky_233.fits', 'sky_437.fits', 'sky_552.fits', 'sky_471.fits', 'sky_505.fits', 'sky_130.fits', 'sky_281.fits', 'sky_432.fits', 'sky_627.fits', 'sky_205.fits', 'sky_274.fits', 'sky_608.fits', 'sky_504.fits', 'sky_469.fits', 'sky_244.fits', 'sky_570.fits', 'sky_564.fits', 'sky_551.fits', 'sky_243.fits', 'sky_553.fits', 'sky_252.fits', 'sky_283.fits', 'sky_377.fits', 'sky_424.fits', 'sky_584.fits', 'sky_452.fits', 'sky_603.fits', 'sky_453.fits', 'sky_456.fits', 'sky_458.fits', 'sky_176.fits', 'sky_121.fits', 'sky_529.fits', 'sky_359.fits', 'sky_451.fits', 'sky_626.fits', 'sky_192.fits', 'sky_314.fits', 'sky_354.fits', 'sky_514.fits', 'sky_295.fits', 'sky_237.fits', 'sky_316.fits', 'sky_238.fits', 'sky_361.fits', 
        'sky_513.fits', 'sky_516.fits', 'sky_278.fits', 'sky_520.fits', 'sky_184.fits', 'sky_153.fits', 'sky_355.fits', 'sky_407.fits', 'sky_531.fits', 'sky_518.fits', 'sky_349.fits', 'sky_344.fits', 'sky_517.fits', 'sky_519.fits', 'sky_619.fits', 'sky_616.fits', 'sky_211.fits', 'sky_422.fits', 'sky_106.fits', 'sky_420.fits', 'sky_528.fits', 'sky_542.fits', 'sky_535.fits', 'sky_527.fits', 'sky_197.fits', 'sky_210.fits', 'sky_171.fits', 'sky_461.fits', 'sky_398.fits', 'sky_621.fits', 'sky_166.fits', 'sky_450.fits', 'sky_448.fits', 'sky_418.fits', 'sky_345.fits', 'sky_352.fits', 'sky_275.fits', 'sky_380.fits', 'sky_483.fits', 'sky_139.fits', 'sky_408.fits', 'sky_403.fits', 'sky_137.fits', 'sky_435.fits', 'sky_151.fits', 'sky_615.fits', 'sky_470.fits', 'sky_404.fits', 'sky_472.fits', 'sky_198.fits', 'sky_433.fits', 'sky_510.fits', 'sky_150.fits', 'sky_207.fits', 'sky_280.fits', 'sky_366.fits', 'sky_491.fits', 'sky_216.fits', 'sky_628.fits', 'sky_294.fits', 'sky_476.fits', 'sky_572.fits', 'sky_604.fits', 'sky_308.fits', 'sky_343.fits', 'sky_486.fits', 'sky_305.fits', 'sky_614.fits', 'sky_480.fits', 'sky_181.fits', 'sky_182.fits', 'sky_339.fits', 'sky_292.fits', 'sky_263.fits', 'sky_289.fits', 'sky_632.fits', 'sky_589.fits', 'sky_576.fits', 'sky_320.fits', 'sky_174.fits', 'sky_302.fits', 'sky_373.fits', 'sky_609.fits', 'sky_475.fits', 'sky_300.fits', 'sky_485.fits', 'sky_444.fits', 'sky_413.fits', 'sky_303.fits', 'sky_610.fits', 'sky_482.fits', 'sky_116.fits', 'sky_311.fits', 'sky_218.fits', 'sky_430.fits', 'sky_481.fits', 'sky_306.fits', 
        'sky_109.fits', 'sky_296.fits', 'sky_406.fits', 'sky_250.fits', 'sky_473.fits', 'sky_415.fits', 'sky_348.fits', 'sky_140.fits', 'sky_307.fits', 'sky_225.fits', 'sky_282.fits', 'sky_439.fits', 'sky_287.fits', 'sky_330.fits', 'sky_291.fits', 'sky_384.fits', 'sky_360.fits', 'sky_374.fits', 'sky_478.fits', 'sky_591.fits', 'sky_241.fits', 'sky_414.fits', 'sky_409.fits', 'sky_376.fits', 'sky_222.fits', 'sky_399.fits', 'sky_353.fits', 'sky_375.fits', 'sky_258.fits', 'sky_566.fits', 'sky_449.fits', 'sky_324.fits', 'sky_484.fits', 'sky_447.fits', 'sky_598.fits', 'sky_599.fits', 'sky_186.fits', 'sky_284.fits', 'sky_261.fits', 'sky_122.fits', 'sky_371.fits', 'sky_362.fits', 'sky_454.fits', 'sky_617.fits', 'sky_195.fits', 'sky_208.fits', 'sky_515.fits', 'sky_381.fits', 'sky_217.fits', 'sky_264.fits', 'sky_525.fits', 'sky_522.fits', 'sky_342.fits', 'sky_187.fits', 'sky_341.fits', 'sky_530.fits', 'sky_327.fits', 'sky_131.fits', 'sky_272.fits', 'sky_172.fits', 'sky_285.fits', 'sky_286.fits', 'sky_132.fits', 'sky_555.fits', 'sky_489.fits', 'sky_337.fits', 'sky_298.fits', 'sky_319.fits', 'sky_526.fits', 'sky_136.fits', 'sky_325.fits', 'sky_488.fits', 'sky_539.fits', 'sky_541.fits', 'sky_340.fits', 'sky_332.fits', 'sky_333.fits', 'sky_299.fits', 'sky_185.fits', 'sky_317.fits', 'sky_502.fits', 'sky_318.fits', 'sky_107.fits', 'sky_147.fits', 'sky_321.fits', 'sky_183.fits', 'sky_133.fits', 'sky_189.fits', 'sky_336.fits', 'sky_154.fits', 'sky_410.fits', 'sky_331.fits', 'sky_492.fits', 'sky_493.fits', 'sky_356.fits', 'sky_425.fits', 'sky_236.fits', 
        'sky_326.fits', 'sky_503.fits', 'sky_506.fits', 'sky_259.fits', 'sky_465.fits', 'sky_395.fits', 'sky_213.fits', 'sky_385.fits', 'sky_167.fits', 'sky_229.fits', 'sky_532.fits', 'sky_145.fits', 'sky_393.fits', 'sky_230.fits', 'sky_389.fits', 'sky_134.fits', 'sky_239.fits', 'sky_372.fits', 'sky_436.fits', 'sky_391.fits', 'sky_240.fits', 'sky_574.fits', 'sky_351.fits', 'sky_569.fits', 'sky_567.fits', 'sky_405.fits', 'sky_125.fits', 'sky_127.fits', 'sky_537.fits', 'sky_262.fits', 'sky_575.fits', 'sky_383.fits', 'sky_587.fits', 'sky_455.fits', 'sky_428.fits', 'sky_508.fits', 'sky_126.fits', 'sky_396.fits', 'sky_388.fits', 'sky_392.fits', 'sky_269.fits', 'sky_169.fits', 'sky_394.fits', 'sky_533.fits', 'sky_387.fits', 'sky_328.fits']
         
    
    science_files_list=['SPHER.2019-11-25T06-44-20.835IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T06-53-14.291IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T07-11-01.375IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T07-19-49.733IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T07-24-17.187IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T07-33-07.891IRD_SCIENCE_DBI_RAW.fits', 'SPHER.2019-11-25T07-41-57.984IRD_SCIENCE_DBI_RAW.fits']
    science=SCIENCE_cubes_extraction_list(science_files_list,compression='mean',bd_subs='yes')
    cubes=SKY_cubes_extraction_list(files_list_sorted ,compression='mean',bd_subs='yes')"""
 


    ### Zone evaluation ###################################################################################################
    #if zone is already saved: enter file name
    #zone=fits.getdata('C:/Users/klara/OneDrive/Pulpit/SPH_files/dir 2/PCA_K12/zone/ZONE_V AB Aur_363_290.fits')
                      #ZONE_HD 83443_283_218.fits')
                      #ZONE_HD 147911_292_225.fits')
                      #ZONE_MWC_758_341_262.fits')

    #first arbitrary approximation of the zone in order to sort skies (sorted skies are used for the precise zone approximation)
    r_approx=250
    zone_approx=get_zone2(np.zeros((1024,2048)),r_approx,r_approx,prepare_Mdist_from_raw(science.T[:1024].T),prepare_Mdist_from_raw(science.T[1024:].T), edge=40)
    cubes_approx, files_list_sorted_approx=get_sorted_sky_list( cubes, science, zone_approx, sky_files_list, type_eval="square") #type_eval="med"
    print(files_list_sorted_approx)
    
  
    #precise zone approximation
    zone_L,r_L=zone_by_dispertion(cubes_approx, science, zone_approx, start_x, end_x, start_y, end_y,"K1", lamb=2.1)
    r_R_specified=r_L*0.8  # as K2 center star zone obtained with is zone_by_dispertion usually too big r_R is specified as a fraction of K1 circle zone
    zone_R, r_R=zone_by_dispertion(cubes_approx, science, zone_approx, start_x, end_x, start_y, end_y,"K2", r_R_specified, lamb=2.2)

    zone=combine_LR(zone_L,zone_R) #glues two matrices horizontaly
    fits.writeto(os.path.dirname(__file__)+'\\zone\\'+'ZONE_'+h_object+'_'+str(int(r_L))+'_'+str(int(r_R))+'.fits',zone,overwrite=True)
   

    ####################################################################################################################################
    
    #Final sorting of sky files
    cubes, files_list_sorted=get_sorted_sky_list( cubes, science, zone, sky_files_list, type_eval="square") #type_eval="med"
    print("SKY FILES SORTED: ",files_list_sorted,'\n SCIENCE FILES LIST:', science_files_list)
    
    #################### Vertical rescaling ##########################################################################################
    """nb_sky=4
    print("Used skies (rescaling):",nb_sky)

    colect_sky=[]
    for j in range(0,nb_sky):
        new_sky=rescaling(cubes[j],science,zone)    
        colect_sky.append(new_sky)

    sky_f=get_compressed_matrix_mean(colect_sky)
    final=science-sky_f

    # replace bad pixels prior attribyted values (0) with NaNs
    bd_px_cube=fits.getdata(os.path.dirname(__file__)+'\\bad pixels map\\'+os.listdir(os.path.dirname(__file__)+'\\bad pixels map\\')[0])
    indexes_bd=bd_px_index(bd_px_cube)
    bd_px_subs_nan(liste_indexes,final) 
    

    fits.writeto(os.path.dirname(__file__)+'\\results\\'+'REDUCED SCIENCE_rescaling'+'.fits',final,overwrite=True)
    print_med_tot(zone,final)
    sys.exit()"""
    ### mean of pca files  ################################################################################################
    """path_save=os.path.dirname(__file__)+'\\results\\'+'REDUCED SCIENCE_HD8 mean 3.fits'
    liste_files=['SKY_HD8 mean rescrectangle_22_c nb_150_n comp_20.fits','SKY_HD8 mean resctriangle_22_c nb_150_n comp_20.fits','SKY_HD8 mean rescrectanglee_16_c nb_150_n comp_20.fits']
    final_pca=mean_result_files(science,liste_files,path_save)
    print_med_tot(zone,final_pca)
    sys.exit()"""

    ### pca - sky reconstruction #############################################################################################

    ### parameters 
    n_cubes=100   
    #number of cubes used for pca. Should be about 50% of the total number of skyimages of the same exposure time
    # for small number of sky images (<80) should be more than 50%


    n_comp=2 # 2/3

    type_div="sliding_rect"  # Divison type: "sliding rect" "rectangle", "triangle" , "rectangle hor", else ('0'): PCA on the whole image
    n_rect=22  # important only if type_div="rectangle"/"rectangle hor",    n_rect must be a divisor of end_x-start_x
    type_coeff="mean" # "norm" , "mean", "area"
    width_rect=90 # important only if type_div="sliding rect" . number of pixels of single rectangle width
   
    path_save_sky=os.path.dirname(__file__)+'\\results\\'+'SKY__ '+h_object+'_'+type_div+'_'+str(n_rect)+"_cubes nb_"+str(n_cubes)+"_n comp_"+str(n_comp)+'_'+str(width_rect)+'.fits'
    path_save_final=os.path.dirname(__file__)+'\\results\\'+'REDUCED SCIENCE__ '+h_object+'_'+type_div+'_'+str(n_rect)+"_c nb_"+str(n_cubes)+"_n comp_"+str(n_comp)+'_'+str(width_rect)+'.fits'
    
    resc_vert = True       # True: reconstructed sky image is rescaled to science column by column.
    margins = True        # True: pca is effectuated on margins area. Otherwise margins are replaced with nans. 
    mean_n = 5       #number of sky images using for centring science during pc calculation. mean_n>=1
    select_auto = False      # if True: evaluates what number of sky images used for centring science during pca is optimal 
    # select_auto : in construction, criterium used in mean_result_files may be insufficient 
    step = 10
    bd_subs = True
    save_ = True

    cubes=np.array(cubes[:n_cubes])
  
    M_1=exe_pca( science, zone, cubes, n_comp, start_x, end_x, start_y, end_y, path_save_sky,
            path_save_final, header, type_coeff, type_div, n_rect, width_rect ,margins, diff_exptime, 
            resc_vert, bd_subs, save_, mean_n, select_auto, step)
    
    end_time = time.perf_counter()
    print("Elapsed time: ", end_time - start_time)
 


if __name__ == '__main__':
    main()


