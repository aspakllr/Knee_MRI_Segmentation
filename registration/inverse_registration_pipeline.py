"""
    Pipeline for inverting the predicted labelmaps (i.e. the outputs of the CNN),
    in the original subject space (i.e. before registration).

    Inverse deformable transformation is calculated using DRAMMS.

"""

import os
from os.path import join as jp
import numpy as np
from glob import glob
import nibabel as nib
from nilearn.image import new_img_like
import time
from scripts.registration.dramms import transf_label


# Path variables for CNN predictions
cnn_path = '/.../'

# Path variables for original data
data_path = '/.../'
data = sorted(glob(os.path.join(data_path, '*')))
data_names = sorted(os.listdir(data_path))

# Load predicted labelmaps (i.e. output of CNN)
cnn_labelmaps = np.load(jp(cnn_path, 'predicted_masks.npy'))  # shape (num_of_images, 64, 128, 160, num_of_labels)

start_time = time.time()

for i in range(0, len(data)):

    target = data_names[i]
    print('Patient', target)

    if 1:  # Convert predictions to labelmaps

        predicted_cnn = cnn_labelmaps[i]

        # Convert predicted softmax labelmap into binary
        predicted_bin = np.where(predicted_cnn > 0.5, 1, 0)

        # Convert from one-hot back into labelmaps with values  [0 1 2 3 4 5]
        predicted_labelmap = np.argmax(predicted_cnn, axis=-1)

        # Save predicted output as nifti
        reg_labelmap = nib.load(jp(data_path, target, ''.join((target, '_deform.nii.gz'))))

        predicted_labelmap_nii = new_img_like(reg_labelmap, predicted_labelmap, copy_header=True)
        predicted_labelmap_nii.to_filename(jp(data_path, target, ''.join((target, '_predicted_labelmap.nii.gz'))))


    if 1:  # Invert predicted labelmap to original image space

        predicted_labelmap = jp(data_path, target, ''.join((target, '_predicted_labelmap.nii.gz')))
        reference_mri = jp(data_path, target, ''.join((target, '.nii.gz')))
        inverse_labelmap = jp(data_path, target, ''.join((target, '_predicted_labelmap_inv.nii.gz')))
        inv_tot_transf = jp(data_path, target, ''.join((target, '_inv_total_transf.nii.gz')))

        # Invert labelmap
        transf_label(predicted_labelmap, inv_tot_transf, inverse_labelmap, reference_mri)

        print('Predicted output is inverted to original image space')

    if 1:  # Invert deform mris to original image space

        deform_mri = jp(data_path, target, ''.join((target, '_deform.nii.gz')))
        inverse_mri = jp(data_path, target, ''.join((target, '_inverse.nii.gz')))
        inv_tot_transf = jp(data_path, target, ''.join((target, '_inv_total_transf.nii.gz')))

        # Invert mri
        transf_label(deform_mri, inv_tot_transf, inverse_mri, reference_mri)

        print('Deformed MRI is inverted to original image space')
