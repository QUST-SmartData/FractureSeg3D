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

import monai
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, VNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
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
test_data_list_path = os.path.join(data_root_path, 'test.txt')

out_dir = f'{dataset}_{model_name}_{loss_name}'
saved_best_model_name = os.path.join(out_dir, "best_metric_model_segmentation_3d_dict.pth")

pred_dir = os.path.join(out_dir, 'pred_inference')

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

    # data file list
    test_images = sorted(read_list_files(os.path.join(data_root_path, 'images'), test_data_list_path))
    test_files = [{"img": img} for img in test_images]

    # define pre transforms
    pre_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            ScaleIntensityd(keys="img"),
        ]
    )
    # define dataset and dataloader
    dataset = Dataset(data=test_files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
    # define post transforms
    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=pre_transforms,
                orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            AsDiscreted(keys="pred", threshold=0.5),
            SaveImaged(keys="pred", output_dir=pred_dir, output_postfix="seg",
                       resample=False, separate_folder=False),
        ]
    )

    # load trained model dict
    model.load_state_dict(torch.load(saved_best_model_name))

    model.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=roi_size, sw_batch_size=4,
                                                 predictor=model)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]


if __name__ == "__main__":
    main()
