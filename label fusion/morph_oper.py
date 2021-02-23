from nilearn.image import load_img, new_img_like
from nilearn.image import largest_connected_component_img as lcc
from scipy.ndimage.morphology import binary_closing
import os
import numpy as np
from scipy.signal import medfilt

jp = os.path.join
abs_path = os.path.abspath('../data')


def morph(labelmap_path, output_path, masks):
    """This function applies morphological operations on a labelmap and saves
    the results to separated masks.

    Parameters:
    labelmap_path (str): absolute path of the input labelmap
    output_path (str): absolute path to save the generated masks
    masks (dict): dictionary of label and masks names

    Returns:
    -----------
    """

    label = list(masks.keys())

    labelmap = load_img(labelmap_path)
    labelmap_d = np.round(labelmap.get_data())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(len(label)):
        cloned_d = np.copy(labelmap_d)
        cloned_d[cloned_d > label[i]] = 0
        cloned_d[cloned_d < label[i]] = 0

        # binary closing
        cloned_d = binary_closing(cloned_d, structure=np.ones((3, 3, 3)))

        # median filter
        cloned_d = medfilt(cloned_d, 3)

        temp_img = new_img_like(labelmap, cloned_d, copy_header=True)

        try:
            # keep largest connected component (remove isolated pieces)
            largest_cc = lcc(temp_img)
            largest_cc.to_filename(jp(output_path, masks[label[i]]))
        except:
            temp_img.to_filename(jp(output_path, masks[label[i]]))
