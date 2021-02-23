"""
    Pipeline for performing deformable registration on data. Data are provided in the Nifti format (.nii.gz).

    Requirements: FSL, DRAMMS

    For installing FSL visit: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
    For installing DRAMMS visit: https://www.med.upenn.edu/sbia/dramms.html


 """

import os
from os.path import join as jp
from os.path import dirname as dn
from glob import glob
import time
from datetime import timedelta
from scripts.registration.dramms import affine, deform,  combine, inv_field, transf_label


# Path variables for atlas/reference image
ref_path = os.path.abspath('/.../')
ref_mri = jp(ref_path, '....nii.gz')
ref_label = jp(ref_path, '...nii.gz')

# Path variables for registered images
data_path = os.path.abspath('/.../')
data = sorted(glob(os.path.join(data_path, '*')))
data_names = sorted(os.listdir(data_path))

# Path variables for saving registered dara
reg_folder = os.path.abspath('/.../')

# Registration
start_time = time.time()

for i in range(0, len(data_names)):

    # Create patient folder for saving results
    name = data_names[i]
    new_folder = os.path.join(reg_folder, name)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    # target_mri: the target MRI we want to register
    target_mri = jp(data[i], ''.join((data_names[i], '.nii.gz')))
    target_labelmap = jp(data[i], ''.join((data_names[i], '_labelmap.nii.gz')))

    # affine_mri: the target MRI with affine registration to the atlas space
    affine_mri = jp(new_folder, ''.join((data_names[i], '_affine.nii.gz')))

    # deform_mri: the target MRI with deformable registration to the atlas space
    deform_mri = jp(new_folder, ''.join((data_names[i], '_deform.nii.gz')))
    deform_labelmap = jp(new_folder, ''.join((data_names[i], '_deform_labelmap.nii.gz')))

    # transformation matrices
    affine_mat = jp(dn(affine_mri), ''.join((data_names[i], '_affine.mat')))
    def_transf = jp(dn(deform_mri), 'def_transf.nii.gz')
    total_transf = jp(dn(deform_mri), ''.join((data_names[i], '_total_transf.nii.gz')))
    inv_tot_transf = jp(dn(deform_mri), ''.join((data_names[i], '_inv_total_transf.nii.gz')))


    if 1:  # Affine registration of target image

        affine(target_mri, affine_mri, ref_mri)


    if 1:  # Deformable registration of target image (deform the affine registered target mri)

        deform(affine_mri, deform_mri, ref_mri)


    if 1:  # Combine affine + deformable transformation of target image

        combine(affine_mat, def_transf, total_transf, target_mri, affine_mri)

    if 1:  # Apply total (affine+deformable) transform to labelmap

        transf_label(target_labelmap, total_transf, deform_labelmap, ref_mri)

    if 1:  # Invert target image transform

        inv_field(total_transf, inv_tot_transf)

    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print('Registration of patient ', data_names[i], '(#', i+1, ') is finished')

    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')

elapsed_time = time.time() - start_time
print('\nElapsed time: ', str(timedelta(seconds=elapsed_time)))
print('\nRegistration of data is finished :)')


