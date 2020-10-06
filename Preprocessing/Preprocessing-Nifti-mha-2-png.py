#Data preprocessing
#Generating List of labeled CT-CBCT images and saving it into Folder for Classification task
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf



def load_and_save_img_from_nii(image_path, Img_Name, saving_path, fileextension): 
    #Read Image from image-path
    img = sitk.ReadImage(image_path)
    #Getting a numpy array
    img = sitk.GetArrayFromImage(img)
    
    a,b,c = np.shape(img)      
                                           
    image_slices = []
    
     #Iterieren über Slice-Zahl Achse
    for i in range(a):
        image_slices.append(np.squeeze(img[i:i+1, :, :], axis=0)) 
    #for CBCT .nii have shape 464,250,464 / CT.nii have shape 143,512,512, adjust the slicing operation and for loop parameter accordingly
    #Slices have shape HeightxWidth - Luminance Info -> GrayScale
    
    labels = [Img_Name + str(id) for id in range(len(image_slices))]
    image_mins = []
    image_maxs = []
    for i in range(len(image_slices)):
        image_min = np.percentile(image_slices[i], 1)
        image_max = np.percentile(image_slices[i], 99.9)
        image_mins.append(image_min)
        image_maxs.append(image_max)

    upper_bound = np.mean(image_maxs)
    print(upper_bound)
    lower_bound = np.mean(image_mins)
    print(lower_bound)
    for i in range(len(image_slices)):
        
        image_slices[i][image_slices[i] < lower_bound] = lower_bound
        image_slices[i][image_slices[i] > upper_bound] = upper_bound
        
        image_slices[i] = (image_slices[i] -lower_bound)/(upper_bound-lower_bound)
        
        filename = saving_path+labels[i]+fileextension
        plt.imsave(filename, image_slices[i], cmap='gray', vmin=0, vmax=1)
        
    return None 

load_and_save_img_from_nii('/home/creim/Desktop/patient_data/neu/R2018036/4dct/bin_10.nii',
                   'CT-R2018036-upper-lower-normiert-Bin-10-', '/home/creim/Desktop/patient_data/neu/R2018036/4dct/Bin-10/', '.png')



#For 4D mha to 3D nii conversion - frederics code
def split_4d_and_save(image_path, Img_Name, saving_path):
    
    image = sitk.ReadImage(image_path)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()

    origin_3d = origin[:-1]
    spacing_3d = spacing[:-1]
    direction_3d = [direction[i] for i in [0, 1, 2, 4, 5, 6, 8, 9, 10]]

    image_arr = sitk.GetArrayFromImage(image)
    
    #Splitting
    images_3d = []
    for i in range(image_arr.shape[0]):
        image_arr_3d = image_arr[i]
        image_3d = sitk.GetImageFromArray(image_arr_3d)
        image_3d.SetOrigin(origin_3d)
        image_3d.SetSpacing(spacing_3d)
        image_3d.SetDirection(direction_3d)
        images_3d.append(image_3d)
    
    label = [Img_Name + str(id) for id in range(len(images_3d))]
    fileextension = '.nii'
    #Saving
    for i in range(len(images_3d)):
        filename = saving_path+label[i]+fileextension
        sitk.WriteImage(images_3d[i], filename)
    return images_3d

#split_4d_and_save('/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/4d_rooster_10_0002_00005.mha', '4D-CBCT-bin-', '/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/')

'''
img_3d = split_4d(sitk.ReadImage('/home/creim/Desktop/Messung1-05-02/CBCT/2020_02_05_session_1/binned/4d_rooster_10_0002_00005.mha'))
labels_bins = ['CBCT-bin-'+str(id) for id in range(len(img_3d))]
for i in range(len(img_3d)):
        path = '/home/creim/Desktop/Messung1-05-02/CBCT/2020_02_05_session_1/binned/'
        filelabel = labels_bins[i]
        fileextension = '.nii'
        filename = path+filelabel+fileextension
        sitk.WriteImage(img_3d[i], filename)
'''

#Für standard 3D niftis
def load_img_nii(image):
    img = sitk.ReadImage(image)
    img = sitk.GetArrayFromImage(img) # getting a numpy array
    a,b,c = np.shape(img)
    image_slices = []
    for i in range(a):
        image_slices.append(np.squeeze(img[i:i+1,:,:], axis=0))
    '''def preprocess(img):
       vmin, vmax = np.amin(img), np.amax(img)
       img = (img - vmin) / (vmax - vmin)
       return img'''
    #Normalisiert die pixel werte auf floats zwischen 0 und 1    
    '''gray_image_slices = []
    for i in range(len(image_slices)):
        gray_image_slices.append(image_slices[i])'''
    '''if cropping is needed
    def crop_data(image_slice):
        width = 1
        height = 1
        crop_image_slice = image_slice[:,height: -height, width: -width,:]
        return crop_image_slice
    cropped_2D_slices = []
    for i in range(len(rgb_image_slices)):
        cropped_2D_slices.append(crop_data(np.reshape(rgb_image_slices[i], (1,512,512,3))))
    '''
    return image_slices


#Saving Images for Classfication/Feature Learning of a CNN
'''
CT_image_arrays = load_img_nii('/home/creim/Desktop/Messung1-05-02/CT/ZZZ_CIRS_LUNGE_R2020005/THORAX-PHANTOM-MESSUNG-1-05-02-2020/RESP_3_0_B31F_AVERAGE_CT_0002/RESP_3_0_B31F_AVERAGE_CT_0002_03_Resp_Gating_Normal_20200205160618_2.nii')
print(len(CT_image_arrays))

#Creating a list of labels
labels_CT = ['CT-image-'+str(id) for id in range(len(CT_image_arrays))]

#Label + Bild aus zwei Listen macht es möglich...
for i in range(len(CT_image_arrays)):
    saving_path = '/home/creim/Desktop/Messung1-05-02/CT/Average Gray Scale mit Spacing 1,1,1/'
    filelabel = labels_CT[i]
    fileextension = '.png'
    filename = saving_path+filelabel+fileextension
    plt.imsave(filename, CT_image_arrays[i], cmap ='gray')
'''
'''
CBCT_images = load_img_nii('/home/creim/Desktop/Style Transfer/Registrated4DImageNewTry.mha')
# Saving the Slices as individual Images for Data Gathering
print(len(CBCT_images))
print(CBCT_images[80].shape)

labels_CBCT = ['Registrated-4D-CBCT-image-'+str(id) for id in range(len(CBCT_images))]
#Label + Bild aus zwei Listen macht es möglich...
for i in range(len(CBCT_images)):
    path = '/home/creim/Desktop/Style Transfer/Registrated 4D CBCT Images2/'
    filelabel = labels_CBCT[i]
    fileextension = '.png'
    filename = path+filelabel+fileextension
    plt.imsave(filename, CBCT_images[i], cmap='gray')
'''