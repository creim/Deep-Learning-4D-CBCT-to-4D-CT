'''Vergleichs-Metriken'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage import color
from skimage.transform import resize
import scipy
import glob


target_img_path = '/home/creim/Desktop/CycleGAN-Tensorflow-2/datasets/Train-Varian-Neu-Test-Thorax-9/test-CT/*.png'
input_img_path = '/home/creim/Desktop/CycleGAN-Tensorflow-2/datasets/Train-Varian-Neu-Test-Thorax-9/test-CBCT/*.png'
generated_img_path= '/home/creim/Desktop/CycleGAN-Tensorflow-2/output5-experiments/final_experiment/Train-Varian-Neu-Test-Thorax-9/samples_testing/A2B/*.png'



#Creating list of arrays of target images
target_img = []
for img in sorted(glob.glob(target_img_path)):
    #normalize img
    image = plt.imread(img)
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    image = resize(image, (512,512))
    """image_min = np.percentile(image, 5)
    image[image < image_min] = image_min
    image_max = np.percentile(image, 95)
    image[image > image_max] = image_max
    """
    target_img.append(image)
#Creating list of arrays of input images
input_img = []
for img in sorted(glob.glob(input_img_path)):
     image = plt.imread(img)
     if len(image.shape) > 2:
         image = color.rgb2gray(image)
     image = resize(image, (512,512))
     """image_min = np.percentile(image, 5)
     image[image < image_min] = image_min
     image_max = np.percentile(image, 95)
     image[image > image_max] = image_max
     image = (image - image_min)/(image_max-image_min)"""
     input_img.append(image)
  

#Creating list of arrays of generated test images
generated_img = []
for img in sorted(glob.glob(generated_img_path)):
    image = plt.imread(img)
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    image = resize(image, (256,256))
    """image_min = np.percentile(image, 5)
    image[image < image_min] = image_min
    image_max = np.percentile(image, 95)
    image[image > image_max] = image_max
    image = (image - image_min)/(image_max-image_min)"""
    generated_img.append(image)

def MSE_Hounsfield(target_img, styled_img):
    '''Computing the Mean Squared Error - in Hounsfield Units, to get some physical evaluation in there'''
    #if pixel values between (0,1) convert generated and reference CT png slice images to HU [lower bound, upper bound]
    """
    def rescale_to_HU(values: np.ndarray, input_range: tuple, output_range: tuple):
        in_min, in_max = input_range
        out_min, out_max = output_range
        return (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    #Hounsfield Units as measured in original Reference CT
    output_range = (-1024, 3071)
    target_img = rescale_to_HU(target_img, (np.min(target_img), np.max(target_img)), output_range)
    styled_img = rescale_to_HU(styled_img, (np.min(styled_img), np.max(styled_img)), output_range)
    """
    
    error = np.sum((target_img - styled_img)**2)
    mse = np.divide(error, styled_img.shape[0] * styled_img.shape[1] *styled_img.shape[2])
    return mse



def MSE_Hounsfield_seg(target_img, styled_img):
    '''Computing the Mean Squared Error - in Hounsfield Units, to get some physical evaluation in there'''
    #Input is already in Hounsfield Units so no rescaling

    
    error = np.sum((target_img - styled_img)**2)
    mse = np.divide(error, styled_img.shape[0] * styled_img.shape[1])
    
    return mse


def NCC(target_img, styled_img):
    '''Computing the Normalized Cross correlation of two images
    which is mathematically the cosine of angle between 
    flattened out image arrays as 1D-vectors
    NCC -> 1 is great'''    
     
    def rescale_to_0_1(values: np.ndarray, input_range: tuple, output_range: tuple ):
            in_min, in_max = input_range
            out_min, out_max = output_range
            return (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
         
    input_range_target = (np.min(target_img), np.max(target_img))
    input_range_styled = (np.min(styled_img), np.max(styled_img))
    
    output_range = (0, 1)
    
    target_img = rescale_to_0_1(target_img, input_range_target, output_range)
    styled_img = rescale_to_0_1(styled_img, input_range_styled, output_range)
    
    
    flat_target = np.reshape(target_img, [-1])
    flat_styled = np.reshape(styled_img, [-1])
    
    dot_product = np.dot(flat_target, flat_styled)
    norm_product = np.linalg.norm(flat_target) * np.linalg.norm(flat_styled)
    ncc = dot_product / norm_product
    return ncc



def SSIM(target_img, styled_img):
    '''Computing the Structural Similarity Index SSIM 
    using the skimage.metrics package; SSIM -> 1 is great'''
    
    def rescale_to_0_1(values: np.ndarray, input_range: tuple, output_range: tuple ):
        in_min, in_max = input_range
        out_min, out_max = output_range
        return (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    
    input_range_target = (np.min(target_img), np.max(target_img))
    input_range_styled = (np.min(styled_img), np.max(styled_img))
    output_range = (0,1)
    target_img = rescale_to_0_1(target_img, input_range_target, output_range)
    styled_img = rescale_to_0_1(styled_img, input_range_styled, output_range)
    
    ssim = compare_ssim(target_img, styled_img, multichannel=False)
    return ssim   

def SNR(img_arr):
     """As defined: mean divided by std"""
     mean = np.mean(img_arr)
     std = np.std(img_arr)
     return mean/std
 
'''
Standardabweichung berechnen der mittleren Metriken
Patienten DATEN !!
cbct_data = patient_gen_ct_9_r2018016
ct_data = patient_ct_r2018016

mean_mse = MSE_Hounsfield(ct_data, cbct_data)**0.5
mean_ncc = NCC(ct_data, cbct_data) 
mean_ssim = SSIM(ct_data, cbct_data)

slices_cbct = []
slices_ct = []
for i in range(71):
    slices_cbct.append(np.squeeze(cbct_data[i:i+1, :, :], axis = 0))
    slices_ct.append(np.squeeze(ct_data[i:i+1, :, :], axis = 0))
    
variance_mse = 0
for ct, cbct in zip(slices_ct, slices_cbct):
     variance_mse += (MSE_Hounsfield_seg(ct, cbct)**0.5 - mean_mse)**2
std_mse = np.sqrt(variance_mse/len(slices_cbct))
print(mean_mse)
print(std_mse)

variance_ncc = 0
for ct, cbct in zip(slices_ct, slices_cbct):
     variance_ncc += (NCC(ct, cbct)- mean_ncc)**2
std_ncc = np.sqrt(variance_ncc/len(slices_cbct))
print(mean_ncc)
print(std_ncc)

variance_ssim = 0
for ct, cbct in zip(slices_ct, slices_cbct):
    variance_ssim += (SSIM(ct, cbct)- mean_ssim)**2
std_ssim = np.sqrt(variance_ssim/len(slices_cbct))
print(mean_ssim)
print(std_ssim)


PHANTOM DATEN LOCH / VOLL TUMOR SLICES

cbct_data_loch = thorax_1_cbct_loch
 ct_data_loch = ref_ct_loch
 cbct_data_voll = thorax_1_cbct_voll
 ct_data_voll = ref_ct_voll 

mean_mse_loch = MSE_Hounsfield(ct_data_loch, cbct_data_loch)**0.5
     ...: mean_ncc_loch = NCC(ct_data_loch, cbct_data_loch) 
     ...: mean_ssim_loch = SSIM(ct_data_loch, cbct_data_loch)
     ...: 
     ...: mean_mse_voll = MSE_Hounsfield(ct_data_voll, cbct_data_voll)**0.5
     ...: mean_ncc_voll = NCC(ct_data_voll, cbct_data_voll)
     ...: mean_ssim_voll = SSIM(ct_data_voll, cbct_data_voll)
     ...: 
     ...: slices_cbct_loch = []
     ...: slices_ct_loch = []
     ...: 
     ...: slices_cbct_voll = []
     ...: slices_ct_voll = []
     ...: 
     ...: for i in range(15):
     ...:     slices_cbct_loch.append(np.squeeze(cbct_data_loch[i:i+1, :, :], axis = 0))
     ...:     slices_ct_loch.append(np.squeeze(ct_data_loch[i:i+1, :, :], axis = 0))
     ...: 
     ...: for i in range(12):
     ...:     slices_cbct_voll.append(np.squeeze(cbct_data_voll[i:i+1, :, :], axis =0))
     ...:     slices_ct_voll.append(np.squeeze(ct_data_voll[i:i+1, :, :], axis = 0))
     ...:     
     ...: variance_mse = 0
     ...: for ct, cbct in zip(slices_ct_loch, slices_cbct_loch):
     ...:      variance_mse += (MSE_Hounsfield_seg(ct, cbct)**0.5 - mean_mse_loch)**2
     ...: std_mse = np.sqrt(variance_mse/len(slices_cbct_loch))
     ...: print("Loch:",mean_mse_loch)
     ...: print("Loch:",std_mse)
     ...: 
     ...: variance_mse = 0
     ...: for ct, cbct in zip(slices_ct_voll, slices_cbct_voll):
     ...:      variance_mse += (MSE_Hounsfield_seg(ct, cbct)**0.5 - mean_mse_voll)**2
     ...: std_mse = np.sqrt(variance_mse/len(slices_cbct_voll))
     ...: print("Voll:",mean_mse_voll)
     ...: print("Voll:",std_mse)
     ...: 
     ...: 
     ...: variance_ncc = 0
     ...: for ct, cbct in zip(slices_ct_loch, slices_cbct_loch):
     ...:      variance_ncc += (NCC(ct, cbct)- mean_ncc_loch)**2
     ...: std_ncc = np.sqrt(variance_ncc/len(slices_cbct_loch))
     ...: print("Loch:",mean_ncc_loch)
     ...: print("Loch:",std_ncc)
     ...: 
     ...: variance_ncc = 0
     ...: for ct, cbct in zip(slices_ct_voll, slices_cbct_voll):
     ...:      variance_ncc += (NCC(ct, cbct)- mean_ncc_voll)**2
     ...: std_ncc = np.sqrt(variance_ncc/len(slices_cbct_voll))
     ...: print("Voll:", mean_ncc_voll)
     ...: print("Voll:",std_ncc)
     ...: 
     ...: variance_ssim = 0
     ...: for ct, cbct in zip(slices_ct_loch, slices_cbct_loch):
     ...:     variance_ssim += (SSIM(ct, cbct)- mean_ssim_loch)**2
     ...: std_ssim = np.sqrt(variance_ssim/len(slices_cbct_loch))
     ...: print("Loch:",mean_ssim_loch)
     ...: print("Loch:",std_ssim)
     ...: 
     ...: variance_ssim = 0
     ...: for ct, cbct in zip(slices_ct_voll, slices_cbct_voll):
     ...:     variance_ssim += (SSIM(ct, cbct)- mean_ssim_voll)**2
     ...: std_ssim = np.sqrt(variance_ssim/len(slices_cbct_voll))
print("Voll:", mean_ssim_voll)
print("Voll:", std_ssim)


PHANTOM DATEN SEGMENTIERT !!

 cbct_data_loch = thorax_1_gen_ct_loch
     ...: ct_data_loch = ref_ct_loch
     ...: cbct_data_voll = thorax_1_gen_ct_voll
     ...: ct_data_voll = ref_ct_voll
     ...: 
     ...: loch_seg = lochtumor_slices_seg
     ...: voll_seg = volltumor_slices_seg
     ...: 
     ...: cbct_loch_seg_all = cbct_data_loch[loch_seg ==1]
     ...: cbct_voll_seg_all = cbct_data_voll[voll_seg ==1]
     ...: 
     ...: ct_loch_seg_all = ref_ct_loch[loch_seg ==1]
     ...: ct_voll_seg_all = ref_ct_voll[voll_seg ==1]
     ...: 
     ...: seg_slices_loch = []
     ...: for i in range(15):
     ...:       seg_slices_loch.append(np.squeeze(loch_seg[i:i+1, :, :], axis=0))
     ...: 
     ...: seg_slices_voll = []
     ...: for i in range(12):
     ...:       seg_slices_voll.append(np.squeeze(voll_seg[i:i+1, :, :], axis=0))  
     ...: 
     ...: 
     ...: slices_cbct_loch = []
     ...: slices_ct_loch = []
     ...: slices_cbct_voll = []
     ...: slices_ct_voll = []
     ...: 
     ...: for i in range(15):
     ...:     slices_cbct_loch.append(np.squeeze(cbct_data_loch[i:i+1, :, :], axis = 0))
     ...:     slices_ct_loch.append(np.squeeze(ct_data_loch[i:i+1, :, :], axis = 0))
     ...: 
     ...: for i in range(12):
     ...:     slices_cbct_voll.append(np.squeeze(cbct_data_voll[i:i+1, :, :], axis =0))
     ...:     slices_ct_voll.append(np.squeeze(ct_data_voll[i:i+1, :, :], axis = 0))
     ...: 
     ...: seg_cbct_slices_loch = []
     ...: seg_cbct_slices_voll = []
     ...: 
     ...: seg_ct_slices_loch = []
     ...: seg_ct_slices_voll = []
     ...: 
     ...: for cbct,seg in zip(slices_cbct_loch, seg_slices_loch):
     ...:     seg_cbct_slices_loch.append(cbct[seg == 1])
     ...: 
     ...: for cbct,seg in zip(slices_cbct_voll, seg_slices_voll):
     ...:     seg_cbct_slices_voll.append(cbct[seg == 1])
     ...: 
     ...: for ct,seg in zip(slices_ct_loch, seg_slices_loch):
     ...:     seg_ct_slices_loch.append(ct[seg == 1])
     ...: 
     ...: for ct,seg in zip(slices_ct_voll, seg_slices_voll):
     ...:     seg_ct_slices_voll.append(ct[seg == 1])
     ...: 
     ...: mean_mse_loch = MSE_Hounsfield_seg(ct_loch_seg_all, cbct_loch_seg_all)**0.5
     ...: mean_ncc_loch = NCC(ct_loch_seg_all, cbct_loch_seg_all) 
     ...: mean_ssim_loch = SSIM(ct_loch_seg_all, cbct_loch_seg_all)
     ...: 
     ...: mean_mse_voll = MSE_Hounsfield_seg(ct_voll_seg_all, cbct_voll_seg_all)**0.5
     ...: mean_ncc_voll = NCC(ct_voll_seg_all, cbct_voll_seg_all)
     ...: mean_ssim_voll = SSIM(ct_voll_seg_all, cbct_voll_seg_all)
     ...: 
     ...: variance_mse = 0
     ...: for ct, cbct in zip(seg_ct_slices_loch, seg_cbct_slices_loch):
     ...:      variance_mse += (MSE_Hounsfield_seg(ct, cbct)**0.5 - mean_mse_loch)**2
     ...: std_mse = np.sqrt(variance_mse/len(slices_cbct_loch))
     ...: print("MSE Loch:",mean_mse_loch)
     ...: print("MSE Loch:",std_mse)
     ...: 
     ...: variance_mse = 0
     ...: for ct, cbct in zip(seg_ct_slices_voll, seg_cbct_slices_voll):
     ...:      variance_mse += (MSE_Hounsfield_seg(ct, cbct)**0.5 - mean_mse_voll)**2
     ...: std_mse = np.sqrt(variance_mse/len(slices_cbct_voll))
     ...: print("MSE Voll:",mean_mse_voll)
     ...: print("MSE Voll:",std_mse)
     ...: 
     ...: 
     ...: variance_ncc = 0
     ...: for ct, cbct in zip(seg_ct_slices_loch, seg_cbct_slices_loch):
     ...:      variance_ncc += (NCC(ct, cbct)- mean_ncc_loch)**2
     ...: std_ncc = np.sqrt(variance_ncc/len(slices_cbct_loch))
     ...: print("NCC Loch:",mean_ncc_loch)
     ...: print("NCC Loch:",std_ncc)
     ...: 
     ...: variance_ncc = 0
     ...: for ct, cbct in zip(seg_ct_slices_voll, seg_cbct_slices_voll):
     ...:      variance_ncc += (NCC(ct, cbct)- mean_ncc_voll)**2
     ...: std_ncc = np.sqrt(variance_ncc/len(slices_cbct_voll))
     ...: print("NCC Voll:", mean_ncc_voll)
     ...: print("NCC Voll:",std_ncc)
     ...: 
     ...: variance_ssim = 0
     ...: for ct, cbct in zip(seg_ct_slices_loch, seg_cbct_slices_loch):
     ...:     variance_ssim += (SSIM(ct, cbct)- mean_ssim_loch)**2
     ...: std_ssim = np.sqrt(variance_ssim/len(slices_cbct_loch))
     ...: print("SSIM Loch:",mean_ssim_loch)
     ...: print("SSIM Loch:",std_ssim)
     ...: 
     ...: variance_ssim = 0
     ...: for ct, cbct in zip(seg_ct_slices_voll, seg_cbct_slices_voll):
     ...:     variance_ssim += (SSIM(ct, cbct)- mean_ssim_voll)**2
     ...: std_ssim = np.sqrt(variance_ssim/len(slices_cbct_voll))
     ...: print("SSIM Voll:", mean_ssim_voll)
     ...: print("SSIM Voll:", std_ssim)

'''
 
#Creating difference images as contour plots
#input is a 2D diff tensor in Hounsfield units, if not squeeze out the third axis and plot using code below
''' plt.imshow(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage('/home/creim/Desktop/CycleGAN-Tensorflow-2/output5-experiments/final_experiment/Train-Thorax-10-mit-1-Tumor-Test-mit-2/samples_testing/Thorax-10-Train-mit-1-Tumor-Lochtumor-Slices.nii'))[0:1, :, :], axis=0), origin='lower', cmap='gray')
     ...: plt.title('Generated CT for Thorax 10')
     ...: plt.colorbar().ax.set_ylabel('Hounsfield Units')'''

#pix2pix Resultat Vergleich zwischen input cbct->target ct und gen ct->target ct
#Cycle GAN resultat vergleich zwischen input cbct-> target ct und gen ct->target ct
MSE_values1 = []
NCC_values1 = []
SSIM_values1 = []

'''Cycle-GAN-Auswertung zwischen zwei Listen von tar_img, gen_img mit einer Länge von 27 Einträgen: 
3 x 5 Lochtumor und 3 x 4 Volltumor)
   pix2pix Auswertung zwischen zwei Listen von tar_img, input_img und gen_img mit einer Länge von 3 Einträgen
'''
"""
for tar_img, inp_img in zip(target_img, input_img):       
     MSE_values1.append(MSE_Hounsfield(tar_img, inp_img))
     NCC_values1.append(NCC(tar_img, inp_img))
     SSIM_values1.append(SSIM(tar_img, inp_img))

print('CycleGAN Input-> Referenz:')
print('MSE mit Loch:', ((MSE_values1[0]+MSE_values1[1]+MSE_values1[2]+MSE_values1[3]+MSE_values1[4]+MSE_values1[9]+MSE_values1[10]+MSE_values1[11]+MSE_values1[12]+MSE_values1[13]+MSE_values1[18]+MSE_values1[19]+MSE_values1[20]+MSE_values1[21]+MSE_values1[22])/15)**0.5)
print('MSE mit Volltumoren:', ((MSE_values1[5]+MSE_values1[6]+MSE_values1[7]+MSE_values1[8]+MSE_values1[14]+MSE_values1[15]+MSE_values1[16]+MSE_values1[17]+MSE_values1[23]+MSE_values1[24]+MSE_values1[25]+MSE_values1[26])/12)**0.5)
print('NCC mit Loch:', (NCC_values1[0]+NCC_values1[1]+NCC_values1[2]+NCC_values1[3]+NCC_values1[4]+NCC_values1[9]+NCC_values1[10]+NCC_values1[11]+NCC_values1[12]+NCC_values1[13]+NCC_values1[18]+NCC_values1[19]+NCC_values1[20]+NCC_values1[21]+NCC_values1[22])/15)
print('NCC mit Volltumoren:', (NCC_values1[5]+NCC_values1[6]+NCC_values1[7]+NCC_values1[8]+NCC_values1[14]+NCC_values1[15]+NCC_values1[16]+NCC_values1[17]+NCC_values1[23]+NCC_values1[24]+NCC_values1[25]+NCC_values1[26])/12)
print('SSIM mit Loch:', (SSIM_values1[0]+SSIM_values1[1]+SSIM_values1[2]+SSIM_values1[3]+SSIM_values1[4]+SSIM_values1[9]+SSIM_values1[10]+SSIM_values1[11]+SSIM_values1[12]+SSIM_values1[13]+SSIM_values1[18]+SSIM_values1[19]+SSIM_values1[20]+SSIM_values1[21]+SSIM_values1[22])/15)
print('SSIM mit VOlltumoren:', (SSIM_values1[5]+SSIM_values1[6]+SSIM_values1[7]+SSIM_values1[8]+SSIM_values1[14]+SSIM_values1[15]+SSIM_values1[16]+SSIM_values1[17]+SSIM_values1[23]+SSIM_values1[24]+SSIM_values1[25]+SSIM_values1[26])/12)


print('pix2pix-Input->Referenz:')
print('Lochtumor-MSE:', MSE_values1[0]**0.5)
print('Volltumor-MSE:', MSE_values1[2]**0.5)
print('Lochtumor-NCC:', NCC_values1[0])
print('Volltumor-NCC:', NCC_values1[2])
print('Lochtumor-SSIM:', SSIM_values1[0])
print('Volltumor-SSIM:', SSIM_values1[2])
"""


MSE_values = []
NCC_values = []
SSIM_values = []
"""
for tar_img, gen_img in zip(target_img, generated_img):       
    MSE_values.append(MSE_Hounsfield(tar_img, gen_img))
    NCC_values.append(NCC(tar_img, gen_img))
    SSIM_values.append(SSIM(tar_img, gen_img))


print('Cycle GAN Generiert -> Referenz:')
print('MSE mit Loch:', ((MSE_values[0]+MSE_values[1]+MSE_values[2]+MSE_values[3]+MSE_values[4]+MSE_values[9]+MSE_values[10]+MSE_values[11]+MSE_values[12]+MSE_values[13]+MSE_values[18]+MSE_values[19]+MSE_values[20]+MSE_values[21]+MSE_values[22])/15)**0.5)
print('MSE mit Volltumoren:', ((MSE_values[5]+MSE_values[6]+MSE_values[7]+MSE_values[8]+MSE_values[14]+MSE_values[15]+MSE_values[16]+MSE_values[17]+MSE_values[23]+MSE_values[24]+MSE_values[25]+MSE_values[26])/12)**0.5)
print('NCC mit Loch:', (NCC_values[0]+NCC_values[1]+NCC_values[2]+NCC_values[3]+NCC_values[4]+NCC_values[9]+NCC_values[10]+NCC_values[11]+NCC_values[12]+NCC_values[13]+NCC_values[18]+NCC_values[19]+NCC_values[20]+NCC_values[21]+NCC_values[22])/15)
print('NCC mit Volltumoren:', (NCC_values[5]+NCC_values[6]+NCC_values[7]+NCC_values[8]+NCC_values[14]+NCC_values[15]+NCC_values[16]+NCC_values[17]+NCC_values[23]+NCC_values[24]+NCC_values[25]+NCC_values[26])/12)
print('SSIM mit Loch:', (SSIM_values[0]+SSIM_values[1]+SSIM_values[2]+SSIM_values[3]+SSIM_values[4]+SSIM_values[9]+SSIM_values[10]+SSIM_values[11]+SSIM_values[12]+SSIM_values[13]+SSIM_values[18]+SSIM_values[19]+SSIM_values[20]+SSIM_values[21]+SSIM_values[22])/15)
print('SSIM mit VOlltumoren:', (SSIM_values[5]+SSIM_values[6]+SSIM_values[7]+SSIM_values[8]+SSIM_values[14]+SSIM_values[15]+SSIM_values[16]+SSIM_values[17]+SSIM_values[23]+SSIM_values[24]+SSIM_values[25]+SSIM_values[26])/12)


print('pix2pix-Generiert->Referenz:')
print('Lochtumor-MSE:', MSE_values[0]**0.5)
print('Volltumor-MSE:', MSE_values[2]**0.5)
print('Lochtumor-NCC:', NCC_values[0])
print('Volltumor-NCC:', NCC_values[2])
print('Lochtumor-SSIM:', SSIM_values[0])
print('Volltumor-SSIM:', SSIM_values[2])
"""





