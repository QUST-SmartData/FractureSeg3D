# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.engines import get_devices_spec
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, VNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
)

# model_name = 'UNet'
# model_name = 'ResUNet'
# model_name = 'VNet'
# model_name = 'AttentionUnet'
# model_name = 'SwinUNETR'
model_name = 'SwinUNETR_Res_v2'
# model_name = 'DSCNet'
# model_name = 'SegMamba'
# model_name = 'SwinUMamba3D'

dataset = 'Carbonate-richShale'
# dataset = 'MancosShale'

loss_name = 'Dice'
# loss_name = 'Cavity'

roi_size = (128, 128, 128)
# roi_size = (64, 64, 64)

data_root_path = f'/opt/data/private/datasets/paper_data/{dataset}/3d'
valid_data_list_path = os.path.join(data_root_path, 'valid.txt')

out_dir = f'{dataset}_{model_name}_{loss_name}'
saved_best_model_name = os.path.join(out_dir, "best_metric_model_segmentation_3d_dict.pth")

pred_dir = os.path.join(out_dir, 'pred_evaluation')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == 'UNet':
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
elif model_name == 'VNet':
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    ).to(device)
elif model_name == 'AttentionUnet':
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
elif model_name == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=1,
    ).to(device)
else:
    model = None
    exit(0)


def read_list_files(parent_path, file_list_path):
    with open(file_list_path, 'r') as f:
        filenames = f.read().splitlines()

    file_paths = [os.path.join(parent_path, filename) for filename in filenames]

    return file_paths


def main():
    # monai.config.print_config()

    valid_images = sorted(read_list_files(os.path.join(data_root_path, 'images'), valid_data_list_path))
    valid_segs = sorted(read_list_files(os.path.join(data_root_path, 'labels'), valid_data_list_path))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(valid_images, valid_segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir=pred_dir, output_ext=".nii.gz", output_postfix="seg", separate_folder=False)
    # load trained model dict
    model.load_state_dict(torch.load(saved_best_model_name))

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # define sliding window size and batch size for windows inference
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            for val_output in val_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    main()
