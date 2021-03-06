"""

Hello world nifti

"""

import numpy as np
import nibabel as nib
import gradio as gr

def sepia(input_img, dropdown):

    # Loading a nifti in the traditional sense
    img = nib.load(input_img.name)
    data = img.get_fdata()
    slice1 = np.squeeze(data[:, :, 200])
    slice1_float = (slice1 - np.min(slice1)) /(np.max(slice1) - np.min(slice1)) 

    return slice1_float, slice1_float, "mesh_test.obj"

demo = gr.Interface(sepia, ["file", gr.Dropdown(['UNET', 'UNETR'])], ["image", "image", gr.Model3D()], live=False, 
    title="Organ segmentation with UNET/UNETR")

demo.launch(share=False)
