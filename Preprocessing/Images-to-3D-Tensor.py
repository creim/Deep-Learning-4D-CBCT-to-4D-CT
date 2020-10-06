#Combining Slice Images to 3D Tensor with Correct Hounsfield Units

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
from skimage import color
from skimage.transform import resize


slice_images = []
file_ext = '.nii'
dataset = 'Test-CBCT-R2018036-Bin-5'
img_dir = '/home/creim/Desktop/CycleGAN-Tensorflow-2/datasets/Train-R2017025-Test-R2017026/test-CBCT-R2018036-Bin-5/*.png'
output_dir = '/home/creim/Desktop/CycleGAN-Tensorflow-2/datasets/Train-R2017025-Test-R2017026/' + dataset + '.nii'


for img in sorted(glob.glob(img_dir)):
    image = plt.imread(img)
    #image = tf.image.decode_png(image, channels=1)
    image = color.rgb2gray(image)
    image = resize(image, (512,512))
    #Convert to Hounsfield Unit range between -1024,3071
    def rescale_to_HU(values: np.ndarray, input_range: tuple, output_range: tuple):
        in_min, in_max = input_range
        out_min, out_max = output_range
        return (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    #Hounsfield Units as measured in original Reference CT
    output_range = (-1024, 824.66)
    image = rescale_to_HU(image, (np.min(image), np.max(image)), output_range)
    slice_images.append(image)
    
#for img in slice_images:
   # norm = np.linalg.norm(img)
    #np.divide(img, norm)
    
tensor_3d = np.stack(slice_images, axis=0)
print(tensor_3d.shape)
sitk.WriteImage(sitk.GetImageFromArray(tensor_3d), output_dir)
    
    
    
