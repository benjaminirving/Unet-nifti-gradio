"""

Main file for processing a particular case using a trained model

"""
import os
import numpy as np

import tempfile
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, BasicUNet

from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch

import trimesh
from skimage import measure
from skimage import morphology
from matplotlib.colors import ColorConverter

print_config()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

def generate_mesh_from_seg(segmentation: torch.tensor):
    """Generate a mesh from the output segmentaton

    Args:
        segmentaton (torch.tensor): volumetric segmentaton
    """

    # Add cleanup of segmentation 
    segmentation_numpy = segmentation.detach().cpu().numpy().astype(int) 
    seg_label = morphology.label(segmentation_numpy)
    seg_label = morphology.remove_small_objects(seg_label, 10000)
    segmentation_numpy[seg_label == 0]=0
    # sn = np.flipud(segmentation_numpy)
    sn = np.rot90(segmentation_numpy, k=3, axes=(1,2))
    sn = np.rot90(sn, k=1, axes=(0,2))

    print("Running marching cubes")

    verts, faces, normals, values = measure.marching_cubes(sn, 0, step_size=2)
    jet = plt.get_cmap('jet')
    rgb = jet(values/values.max())
    print("Completed marching cubes")
    surf_mesh = trimesh.Trimesh(verts, faces, vertex_colors=rgb, validate=True)
    # surf_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(segmentation_numpy)
    print("Completed trimesh")
    # is_smooth = True
    # if is_smooth:
    #     trimesh.smoothing.filter_taubin(surf_mesh, iterations=10)
    # print("Completed smoothing")
    surf_mesh.export('mesh_test.obj')
    print("completed export")

    return segmentation_numpy, trimesh

def process_case(case_path: str, model_type: str = 'unet'):
    """
    Process a single case using MONAI

    Input: Case path

    Output: Segmentation
    """

    # Define a dataloader for a single case
    val_files = [{'image': case_path}]
    val_ds = Dataset(
    data=val_files, transform=val_transforms)
    # val_loader = DataLoader(
    #     val_ds, batch_size=1, shuffle=False)

    # Define the model
    if model_type == 'unet':
        # Define a UNET
        print("Using UNET")
        model = BasicUNet(
            in_channels=1,
            out_channels=14,
        ).to(device)

        model_name_epoch = "models/unet_btcv_segmentation24500.pth" 
        model.load_state_dict(torch.load(model_name_epoch, map_location=device))
        model.eval()

    elif model_type == 'unetr':
        # Load UNETR
        model = UNETR(
            in_channels=1,
            out_channels=14,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

        model_name_epoch = "models/unetr_btcv_segmentation11000.pth" 
        model.load_state_dict(torch.load(model_name_epoch, map_location=device))
        model.eval()
    else:
        print('No such model')
        return 0
    
    #  Extracting info and running model
    img = val_ds[0]["image"].to(device)
    print(f"Images shape: {img.shape}")
    img_name = os.path.split(val_ds[0]["image_meta_dict"]["filename_or_obj"])[1]
    val_inputs = torch.unsqueeze(img, 1)
    print('Run model')
    # val_outputs = sliding_window_inference(
    #     val_inputs, (96, 96, 96), 1, model, overlap=0.8
    # )
    # slice1 = 134
    slice1 = 150
    segmentation = model(val_inputs)
    # Rotate the image to match
    segmentation = torch.rot90(torch.argmax(segmentation, dim=1).squeeze(), dims=[0, 1])
    img = torch.rot90(img.squeeze(), dims=[0, 1])


    segmentation, trimesh = generate_mesh_from_seg(segmentation)
    masked = np.ma.masked_where(segmentation == 0, segmentation)

    im_path = "test.jpg"
    print('Model finished - running test plot')
    plt.imshow(img.cpu().numpy()[:, :, slice1], cmap="gray")
    plt.imshow(masked[:, :, slice1])
    plt.axis('off')
    plt.savefig(im_path, bbox_inches='tight')

    return segmentation, im_path


if __name__ == "__main__":
    """
    Run on a test case
    """

    path1 = 'test_data/liver_118.nii.gz'
    # path1 = '/home/ben/Code/gradio_test/test_data/btcv_imagesTs_img0002.nii.gz'
    seg = process_case(path1, model_type="unet")

