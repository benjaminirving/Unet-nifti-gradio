"""

Main file for processing a particular case using a trained model

"""
import os

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
        model = BasicUNet(
            in_channels=1,
            out_channels=14,
        ).to(device)

        model_name_epoch = "models/unet_btcv_segmentation24500.pth" 
        model.load_state_dict(torch.load(model_name_epoch, map_location=device))
        model.eval()

    elif model_type == 'unetr':
        # Load UNETR
        x=1
    else:
        print('No such model')
        return 0
    
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

    val_outputs = model(val_inputs)
    print('Model finished')
    plt.figure("check", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice1], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("output")
    plt.imshow(
        torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice1]
    )
    plt.savefig('test.jpg')


if __name__ == "__main__":
    """
    Run on a test case
    """

    path1 = 'test_data/liver_118.nii.gz'
    # path1 = '/home/ben/Code/gradio_test/test_data/btcv_imagesTs_img0002.nii.gz'
    process_case(path1)

