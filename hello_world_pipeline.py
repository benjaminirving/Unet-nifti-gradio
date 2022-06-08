"""

Test pipeline for loading a CT scan

"""

import numpy as np
import nibabel as nib
import gradio as gr
from processing import process_case
from time import time

def segment_organs(input_img, dropdown):

    # Loading a nifti in the traditional sense
    slice1_num =  150

    t1 = time()
    segmentation, im_path = process_case(input_img.name)
    t2 = time()
    print((t2-t1)/60)
    slice1 = np.squeeze(segmentation[:, :, slice1_num])
    slice1_float = (slice1 - np.min(slice1)) /(np.max(slice1) - np.min(slice1)) 

    return "mesh_test.obj", im_path

demo = gr.Interface(segment_organs, 
    ["file", gr.Dropdown(['UNET', 'UNETR'], label="Model")], 
    [gr.Model3D(), "image"], 
    live=False, 
    title="Organ segmentation with UNET/UNETR")

demo.launch(share=False)
