import nibabel as nib
import numpy as np
import os
from os.path import join as jp
from os.path import dirname as dn
# from collections import OrderedDict


def mask2labelmap(label_path, masks):
    """This function combines the different segmentation files
    to create a single labelmap to use for the segmentation procedure.

    Parameters: 
    label_path (str): Absolute name of the labelmap that will be created.
                Masks lie in the same folder as the labelmap.
    masks (dict): dictionary of labels and mask names that specifies which
           masks will be combined and with which label value each one.

    Returns:
    -----------
    """

    mask_labels = list(masks.keys())

    # base_file is the segmenation mask file upon which we add the labels
    # of the other masks

    input_data_path = dn(label_path)

    base_name = jp(input_data_path, masks[mask_labels[0]])
    base_file = nib.load(base_name)
    base_data = base_file.get_data()

    label_value = mask_labels[0]

    """
    There are some label values possibly due to an error or interpolation
    that belong to the background but do not have a value precisely 0. So
    we assume that all these voxels belong to the background and have
    value less than 0.5.
    """

    base_data[base_data >= 0.8] = label_value
    base_data[base_data < 0.8] = 0

    for j in range(1, len(mask_labels)):

        # Load the segmenation masks

        seg_name = jp(input_data_path, masks[mask_labels[j]])
        seg_file = nib.load(seg_name)

        # Get the segmentation mask values

        seg_data = seg_file.get_data()

        label_value = mask_labels[j]

        # Replace with the new label value and make the values that should
        # be zero equal to zero.

        seg_data[seg_data >= 0.8] = label_value
        seg_data[seg_data < 0.8] = 0

        # Check for overlap of masks

        overlap = np.logical_and(base_data, seg_data)

        # Add the new label value to the base mask

        base_data = np.add(base_data, seg_data)

        # and replace overlap region with the new label

        base_data[overlap == 1] = label_value

    # Create a new nifty file containing the new labelmap

        labelmap = nib.Nifti1Image(
            base_data, base_file.affine, base_file.header, base_file.extra,
            base_file.file_map)

        # Save the new labelmap to the same folders with the input
        # data

        nib.save(labelmap, label_path)


