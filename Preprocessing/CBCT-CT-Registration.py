#Image Registration
#Average CBCT Registration to AverageCT Image

import SimpleITK as sitk

# If the environment variable SIMPLE_ITK_MEMORY_CONSTRAINED_ENVIRONMENT is set, this will override the ReadImage
# function so that it also resamples the image to a smaller size (testing environment is memory constrained).
#%run setup_for_testing

import os
import numpy as np

from ipywidgets import interact, fixed


# This is the registration configuration which we use in all cases. The only parameter that we vary 
# is the initial_transform. 
def multires_registration(fixed_image, moving_image, initial_transform):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return (final_transform, registration_method.GetMetricValue())

#average CT - 002
fixed_image = sitk.ReadImage('/home/creim/Desktop/patient_data/neu/R2018036/4dct/avg.nii', sitk.sitkFloat32)

#average CBCT - von Messprotokoll X
moving_image = sitk.ReadImage('/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/avg_fdk.mha', sitk.sitkFloat32)


#Center both images before transform - geometrically
initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
#compute avg transform
#avg_transform,_ = multires_registration(fixed_image, moving_image, initial_transform)

def save_transform_and_image(transform, fixed_image, moving_image, outputfile_prefix):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.
    
    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to 
                                    outputfile_prefix.mha.
    """                 
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(transform)
    sitk.WriteImage(resample.Execute(moving_image), outputfile_prefix+'.mha')
    sitk.WriteTransform(transform, outputfile_prefix+'.tfm')

#avg_CT_alt_moving = sitk.ReadImage('/home/creim/Desktop/Messungen-Bild-Daten-mit-variierenden-Strahlparametern/4D CT Referenz - alt/ZZZ_CIRS_LUNGE_R2020005/THORAX-PHANTOM-MESSUNG-1-05-02-2020/RESP_3_0_B31F_AVERAGE_CT_0002/RESP_3_0_B31F_AVERAGE_CT_0002_03_Resp_Gating_Normal_20200205160618_2.nii', sitk.sitkFloat32)
#avg_CT_neu_fixed = sitk.ReadImage('/home/creim/Desktop/Messungen-Bild-Daten-mit-variierenden-Strahlparametern/4D-CT-Referenz-neu/THORAX_03_RESP_GATING_NORMAL_ERWACHSENER_20200701_182846_270000/AVERAGE_CT/RESP_3_0_B31F_AVERAGE_CT_0003_03_Resp_Gating_Normal_20200701182846_3.nii', sitk.sitkFloat32)

#save_transform_and_image(avg_transform, fixed_image, moving_image, '/home/creim/Desktop/Messungen-Bild-Daten-mit-variierenden-Strahlparametern/4D CT Referenz - alt/ZZZ_CIRS_LUNGE_R2020005/THORAX-PHANTOM-MESSUNG-1-05-02-2020/RESP_3_0_B31F_100_EX_-_100_IN_HERZPHASE_100%_0003/Reg-CT-alt-Bin-0-100%'  )

CT_4D_fixed_image = sitk.ReadImage('/home/creim/Desktop/patient_data/neu/R2018036/4dct/bin_09.nii', sitk.sitkFloat32)
CBCT_4D_moving_image = sitk.ReadImage('/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/4D-CBCT-bin-9.nii', sitk.sitkFloat32)

save_transform_and_image(sitk.ReadTransform('/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/Reg-4D-CBCT-bin-0.tfm'), CT_4D_fixed_image, CBCT_4D_moving_image, '/home/creim/Desktop/patient_data/neu/R2018036/4dcbct/Reg-4D-CBCT-bin-9') 