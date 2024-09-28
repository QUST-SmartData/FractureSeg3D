import os
import random

import nibabel as nib
from tqdm import tqdm


def read_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine


def save_nii(cropped_data, affine, output_path):
    cropped_img = nib.Nifti1Image(cropped_data, affine)
    nib.save(cropped_img, output_path)
    # print(f'saved {output_path}!')


def crop_and_save(input_file, output_dir, crop_size, stride):
    data, affine = read_nii(input_file)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cropped_files = []
    index = 0
    # Iterate over the data with the given stride
    for start_z in range(0, data.shape[2] - crop_size[2] + 1, stride[2]):
        for start_y in range(0, data.shape[1] - crop_size[1] + 1, stride[1]):
            for start_x in range(0, data.shape[0] - crop_size[0] + 1, stride[0]):
                # Calculate the end coordinate for each dimension
                end_x = start_x + crop_size[0]
                end_y = start_y + crop_size[1]
                end_z = start_z + crop_size[2]

                # Perform the actual cropping
                crop_slices = [slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z)]
                cropped_data = data[tuple(crop_slices)]

                # Create a unique filename for the cropped volume
                file_name = "{}_cropped_{:04d}.nii".format(os.path.splitext(os.path.basename(input_file))[0], index)
                index = index + 1
                output_path = os.path.join(output_dir, file_name)

                # Save the cropped volume
                save_nii(cropped_data, affine, output_path)
                cropped_files.append(file_name)

    return cropped_files


def save_list_to_file(cropped_files_list, save_file_path):
    with open(save_file_path, 'w') as f:
        for file_name in cropped_files_list:
            f.write(file_name + '\n')  # 将每个文件名写入文件并追加换行符


def main(input_txt_path, dataset_dir, output_dir, crop_size, stride,
         save_list_file_path, save_train_list_file_path, save_valid_list_file_path):
    with open(input_txt_path, 'r') as file:
        nii_files = [line.strip() for line in file.readlines()]

    cropped_files_list = []
    for file_name in tqdm(nii_files, desc='Cropping: ', unit='img'):
        # images
        input_file_path = os.path.join(dataset_dir, 'images', file_name)
        cropped_list = crop_and_save(input_file_path,
                                     os.path.join(output_dir, 'images'),
                                     crop_size, stride)
        # labels
        input_file_path = os.path.join(dataset_dir, 'labels', file_name)
        cropped_list = crop_and_save(input_file_path,
                                     os.path.join(output_dir, 'labels'),
                                     crop_size, stride)
        cropped_files_list += cropped_list
    save_list_to_file(cropped_files_list, save_list_file_path)

    train_list, valid_list = split_dataset(cropped_files_list)
    save_list_to_file(train_list, save_train_list_file_path)
    save_list_to_file(valid_list, save_valid_list_file_path)


def split_dataset(cropped_files_list):
    # 随机打乱列表
    random.shuffle(cropped_files_list)

    # 计算训练集和验证集的分割点
    split_point = int(len(cropped_files_list) * 0.8)

    # 划分训练集和验证集
    train_list = cropped_files_list[:split_point]
    valid_list = cropped_files_list[split_point:]

    return train_list, valid_list


if __name__ == "__main__":

    crop_size = (128, 128, 128)
    stride = (64, 64, 64)

    datasets = ['Carbonate-richShale', 'MancosShale']
    for dataset in datasets:
        data_root_path = f'/opt/data/private/datasets/paper_data/{dataset}'
        data_list_path = f'{data_root_path}/train_valid_list.txt'
        output_dir = f'./datasets_cropped/{dataset}'
        output_list_file = f'{output_dir}/train_valid_list.txt'
        output_train_list_file_path = f'{output_dir}/train.txt'
        output_valid_list_file_path = f'{output_dir}/valid.txt'
        print(f'{dataset}: ')
        main(data_list_path, data_root_path, output_dir, crop_size, stride,
             output_list_file, output_train_list_file_path, output_valid_list_file_path)
