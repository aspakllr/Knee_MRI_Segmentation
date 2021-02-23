import os
from os.path import join as jp
from os.path import dirname as dn
from os.path import exists
from nilearn.image import load_img

abs_dramms = os.path.abspath('../dramms/bin')


def affine(input_file, output_file, ref_file, search_direction=0):
    """This function finds an affine transformation using flirt. It calls the
    flirt program through the commmand line using the parameter -dof 6 that
    stands for rigid registration.

    Parameters:
    input_file (str): absolute path of the MRI file we want to register.
    output_file (str): absolute path of the registered MRI file name. In this
                       path the transformation file is also saved with file
                       extension .mat.
    ref_file (str): absolute path of the MRI file used as reference for the
                    registration (atlas space.)
    search_direction (int): the direction of search (default=0)
                            for left knees set to 90

    Returns:
    -----------
    """

    if not exists(dn(output_file)):
        os.makedirs(dn(output_file))

    search_direction = str(search_direction)
    dof = str(12)  # alternatives: 6, 7
    cost = 'normcorr'  # alternatives: normmi, leastsq

    subcmd = ' -searchrx -' + search_direction + ' ' + search_direction + \
        ' -searchry -' + search_direction + ' ' + search_direction + \
        ' -searchrz -' + search_direction + ' ' + search_direction

    command = 'flirt -in ' + input_file + ' -ref ' + ref_file + ' -out ' + \
        output_file + ' -omat ' + output_file[:-7] + '.mat -bins 256' + \
        ' -cost ' + cost + subcmd + ' -dof ' + dof + '  -interp trilinear'
    os.system(command)

def apply_transf_with_applywarp(input_file, output_file, affine_matrix, ref_file):
    """This function applies an existing transformation to an image with spline interpolation

         Parameters:
         input_file (str): absolute path of the image we want to register
         output_file (str): absolute path of the registered image
         affine_matrix (str): absolute path of the transformation matrix
         ref_file (str): reference file that determines the size of the output volume-it's contents are not used

         Returns:
         -----------
         """

    command = ' applywarp -i ' + input_file + ' -o ' + output_file + ' -r ' + ref_file + ' --premat=' + affine_matrix + ' --interp=spline '
    os.system(command)


def apply_transf_to_labelmap(input_labelmap, output_labelmap, affine_matrix, ref_file):
    """This function applies an existing transformation to a labelmap with nearest neighbor interpolation

      Parameters:
      input_labelmap (str): absolute path of the labelmap we want to register
      output_labelmap (str): absolute path of the registered labelmap
      affine_matrix (str): absolute path of the transformation matrix
      ref_file (str): reference file that determines the size of the output volume-it's contents are not used

      Returns:
      -----------
      """

    command = ' flirt -in ' + input_labelmap + ' -ref ' + ref_file + ' -out ' + output_labelmap + ' -init ' + affine_matrix + ' -applyxfm ' + '  -interp nearestneighbour '
    os.system(command)


def deform(input_file, output_file, ref_file):
    """This function finds a deformable transformation using dramms. It calls
    the dramms program through the commmand line.

    Parameters:
    input_file (str): absolute path of the MRI file we want to register. It
                      must have been registered firstly using an affine or
                      rigid transformation.
    output_file (str): absolute path of the registered MRI file name. In this
                       path the transformation file is also saved with file
                       extension .nii.gz.
    ref_file (str): absolute path of the MRI file used as reference for the
                    registration (atlas space.)

    Returns:
    -----------
    """
    out_fol = dn(output_file)
    output_def = jp(out_fol, 'def_transf.nii.gz')

    command = abs_dramms + '/dramms -S ' + input_file + ' -T ' + ref_file + \
        ' -O ' + output_file + ' -D ' + output_def + ' -w 1 -a 0 -v -v'
    os.system(command)


def combine(input_aff, input_def, out_transf, original_mri, aff_mri):
    """This function combines 1 affine and 1 def transformation into a single
    file. The MRIs before the affine transformation and after the affine must
    also be provided.

    Parameters:
    input_aff (str): absolute path of the affine transformation file (.mat)
                     that we want to combine.
    input_def (str): absolute path of the deformable transformation file
                     (.nii.gz) that we want to combine.
    out_transf (str): absolute path of the combined transformation file
                     (.nii.gz).
    original_mri (str): absolute path of the MRI file before it was registered
                        with the affine transformation.
    aff_mri (str): absolute path of the MRI file after it was registered with
                   the affine transformation but not registered with the deform
                   transformation.

    Returns:
    -----------
    """

    command = abs_dramms + '/dramms-combine -c -f ' + original_mri + \
        ' -t ' + aff_mri + ' ' + input_aff + ' ' + input_def + \
        ' ' + out_transf

    os.system(command)


def inv_field(input_field, output_field):
    """This function calculates the inverse transformation of a deformable
    tranformation (deformation field with extension .nii.gz).

    Parameters:
    input_field (str): absolute path of the deformation field that we want to
                       invert.
    output_field (str): absolute path of the inverse deformation field.

    Returns:
    -----------
    """

    command = abs_dramms + '/dramms-defop -i ' + input_field + ' ' + \
        output_field

    os.system(command)


def transf_label(input_label, input_field, output_label, template_file):
    """This function applies a transformation to a labelmap.

    Parameters:
    input_label (str): absolute path of the labelmap file that we want to
                       transform.
    input_field (str): absolute path of the transformation that we want to
                       apply to the labelmap.
    output_label (str): absolute path of the transformed labelmap.

    Returns:
    -------------
    """

    if not exists(dn(output_label)):
        os.makedirs(dn(output_label))

    command = abs_dramms + '/dramms-warp ' + input_label + ' ' + \
        input_field + ' ' + output_label + ' -t ' + template_file + ' -n'

    os.system(command)


def resample(input_file, output_file, template_file, voxel_size):
    """This function changes the dimension and voxel size of a new image to
    match those of the target image, so that the inverse transformation can
    be calculated correctly.

    Parameters:
    input_file (str): absolute path of the target image.
    output_file (str): absolute path of the resulted resampled image.
    tempalte_file (str): absolute path of the template file used for
                         registration.
    voxel_size (array): the voxel size of the template image.

    Returns:
    -------------
    """

    if not exists(dn(output_file)):
        os.makedirs(dn(output_file))

    temp_img = load_img(template_file)
    dim = temp_img.shape
    command = abs_dramms + '/dramms-imgop -p ' + str(voxel_size[0]) + ',' + \
        str(voxel_size[1]) + ',' + str(voxel_size[2]) + ' -d ' + \
        str(dim[0]) + ',' + str(dim[1]) + ',' + str(dim[2]) + ' ' + \
        input_file + ' ' + output_file

    os.system(command)


def thresh(input, output):
    """This function binarizes (0, 1) a Nifti image using a threshold

       Parameters:
       input (str): absolute path of the nifti image to binarize
       output (dict): absolute path of the binarized nifti image

       Returns:
       -----------
       """
    command = ' fslmaths ' + input + ' -thr 0.5 ' + ' -bin ' + output
    os.system(command)
