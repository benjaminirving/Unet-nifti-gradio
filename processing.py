"""

Main file for processing a particular case using a trained model

"""
import tempfile
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
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
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False)

    # Define the model
    if model_type == 'unet':
        # Define a UNET
        model = BasicUNet(
            in_channels=1,
            out_channels=14,
        ).to(device)

        model_name_epoch = "models/unet_btcv_segmentation24500.pth" 
        model.load_state_dict(torch.load(model_name_epoch))
        model.eval()



    elif model_type == 'unetr':
        # Load UNETR
        x=1
    else:
        print('No such model')
        return 0









if __name__ == "__main__":
    """
    Run on a test case
    """

    path1 = 'test_data/liver_118.liver_118.nii.gz'
    process_case(path1)

