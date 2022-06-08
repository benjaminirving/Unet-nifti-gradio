"""

Hello world nifti

"""

import numpy as np
import nibabel as nib
import gradio as gr
from processing import process_case

def sepia(input_img, dropdown):

    # Loading a nifti in the traditional sense
    slice1_num =  150
    segmentation = process_case(input_img.name)
    slice1 = np.squeeze(segmentation[:, :, slice1_num])
    slice1_float = (slice1 - np.min(slice1)) /(np.max(slice1) - np.min(slice1)) 

    return "mesh_test.obj", slice1_float, slice1_float

demo = gr.Interface(sepia, ["file", gr.Dropdown(['UNET', 'UNETR'])], [gr.Model3D(), "image", "image"], live=False, 
    title="Organ segmentation with UNET/UNETR")

demo.launch(share=False)
