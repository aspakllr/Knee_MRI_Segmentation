"""
    Pipeline for label fusion.

    Requirements: PICSL

    For installing PICSL visit: https://www.nitrc.org/projects/picsl_malf
 """

import os
from os.path import join as jp
from scripts.label_fusion.picsl import segmentation
from scripts.label_fusion.labelmaps import mask2labelmap as lab
from scripts.label_fusion.morph_oper import morph
from os.path import dirname as dn
import numpy as np
import time

# Path variables
data_path = '/.../'
data_names = sorted(os.listdir(data_path))


for pat in range(0, len(data_names)):

    # Load target MRI, i.e. the image we want to calculate the segmentation
    patient = data_names[pat]
    target_mri = jp(data_path, patient, ''.join((patient, '.nii.gz')))

    # warped_label: the predicted labelmaps from each atlas/reference space, that are projected back to the original 	   subject space
    warped_label = np.array([
        jp('/.../Registration_1/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz'))),
        jp('/.../Registration_2/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz'))),
        jp('/.../Registration_3/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz'))),
        jp('/.../Registration_4/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz'))),
        jp('/.../Registration_5/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz'))),
        jp('/.../Registration_6/...', patient,  ''.join((patient, '_predicted_labelmap_inv.nii.gz')))
        ])

    # warped_mri: the mris from each atlas/reference space, that are projected back to the original subject space
    warped_mri = np.array([
        jp('/.../Registration_1/...', patient, ''.join((patient, '_inverse.nii.gz'))),
        jp('/.../Registration_2/...', patient, ''.join((patient, '_inverse.nii.gz'))),
        jp('/.../Registration_3/...', patient, ''.join((patient, '_inverse.nii.gz'))),
        jp('/.../Registration_4/...', patient, ''.join((patient, '_inverse.nii.gz'))),
        jp('/.../Registration_5/...', patient, ''.join((patient, '_inverse.nii.gz'))),
        jp('/.../Registration_6/...', patient, ''.join((patient, '_inverse.nii.gz')))
        ])

    # Create patient folder for saving results
    lbl_fusion_folder = jp( '/.../', patient)
    os.mkdir(lbl_fusion_folder)

    # out_seg_picsl: fused labelmap
    out_seg_picsl = jp(lbl_fusion_folder, ''.join((patient, '_fused_labelmap.nii.gz')))

    # Label fusion
    start = time.time()
    segmentation(warped_mri, out_seg_picsl, target_mri, warped_label)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print('')
    print(pat, '-Label fusion of patient', data_names[pat], 'is finished')
    print('Time elapsed:')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print('********************************************************************************************************************************************************************************************')


    # Enhance label fusion with filters

    # segmentation masks
    masks = {
        1: 'femur.nii.gz',
        2: 'tibia.nii.gz',
        3: 'femoral_cartilage.nii.gz',
        4: 'lateral_tibial_cartilage.nii.gz',
        5: 'medial_tibial_cartilage.nii.gz',
           }

    # Create folder for saving results after filtering
    morph_output = jp(lbl_fusion_folder, 'filtered')
    if not os.path.exists(morph_output):
        os.mkdir(morph_output)

    fused_labelmap = jp(lbl_fusion_folder, ''.join((patient, '_fused_labelmap.nii.gz')))
    filtered_labelmap = jp(morph_output, '_filtered_labelmap.nii.gz')

    morph(fused_labelmap, dn(filtered_labelmap), masks)
    lab(filtered_labelmap, masks)
