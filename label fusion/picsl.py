import os
from os.path import dirname
import collections as col
import shutil
import numpy as np

jp = os.path.join

abs_picsl = os.path.abspath('../picsl/src')


def segmentation(input_path_mri, output_path_seg, target_path_mri,
                 input_path_label):
    """Creates the Joint Label Fusion file for the target image.

    Parameters:
    input_path_mri (numpy-array): list of absolute paths of the warped mri
    output_path_seg (str): absolute path of the output segmentation
    target_path_mri (str): absolute path of the mri we want to segment
    input_path_label (numpy-array): list of absolute paths of the warped
                                    labelmaps

    Returns:
    -----------
    """

    command_array = []

    modalities = input_path_mri.ndim

    # command_array is an array where each element is a command line argument
    # so if we display the command_array array it will be more readable

    command_array.append('jointfusion 3 ' + str(modalities) + ' -g \\')

    for i in range(len(input_path_mri)):

        # We write the warped volumes files to the script files.
        # If there are more than one modalities we write all of them to the
        # script file.

        if modalities is not 1:
            for j in range(modalities):

                command_array.append('\n' + input_path_mri[i][j] + ' \\')
        else:
            command_array.append('\n' + input_path_mri[i] + ' \\')

    command_array.append('\n-l \\')

    for i in range(len(input_path_label)):

        # We write the Warped labelmap files to the script.

        command_array.append('\n' + input_path_label[i] + ' \\')

        # Writing parameters

    command_array.append('\n-m Joint[0.1,2] \\')
    command_array.append('\n-rp 2x2x2 \\')
    command_array.append('\n-rs 3x3x3 \\')

    # We write the target to segment volume to the script file
    # And if there are more than one modalities we write them also

    command_array.append('\n-tg \\')

    if modalities is not 1:

        for i in range(len(target_path_mri)):
            command_array.append('\n' + target_path_mri[i] + ' \\')
    else:
        command_array.append('\n' + target_path_mri + ' \\')

    output_path = dirname(output_path_seg)

    command_array.append('\n-p ' + jp(output_path,
                                  'Segm_Joint_posterior%04d.nii.gz') + ' \\')
    command_array.append('\n' + output_path_seg)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(command_array)
    # execute the command after the command_array array is converted into a
    # string
    command_string = ''.join(command_array)
    os.system(jp(abs_picsl, command_string))


