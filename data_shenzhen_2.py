import tensorflow as tf
import torch
import os
import numpy as np
import json

# 设置日期列表和波段名称
date_list = ['01-30', '02-19', '03-02', '04-09', '05-04', '06-28', '07-18', '08-12', '09-01', '10-26', '11-25', '12-30']
BANDS = []
for date in date_list:
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B2")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B3")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B4")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B8")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_days")
print(len(BANDS))

KERNEL_SIZE = 128
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
kernel_buffer = [KERNEL_SIZE // 2, KERNEL_SIZE // 2]  # [64, 64]

image_name_prefix = "shenzhen_mosaic2020_all_month_buffer_subregion128_"

# 子区域列表（假设你已经定义了 subregion_collection ）
subregion_collection_list = ['sub_region_0', 'sub_region_1','sub_region_12','sub_region_13','sub_region_4','sub_region_5','sub_region_6','sub_region_7','sub_region_8','sub_region_9','sub_region_10','sub_region_11''sub_region_12','sub_region_13','sub_region_14','sub_region_15','sub_region_16','sub_region_17','sub_region_18','sub_region_19','sub_region_20', 'sub_region_21']
# 改成子区域名称，即subregion_collection_list里面存储的是[“sub_region_0”, “sub_region_1”, …, “sub_region_21”]
length = 22 # 改成子区域的数量，即22


# 解析和转换 TFRecord 数据
def doPrediction(out_image_base, user_folder, kernel_buffer, region_name):
    """Perform inference on exported imagery, upload to Earth Engine."""
    print('Looking for TFRecord files...')

    # 修改文件路径读取：从本地路径读取 .tfrecord.gz 文件
    image_folder = r"E:\shenzhen_mosaic"  # 修改为本地路径
    filesList = [f for f in os.listdir(image_folder) if f.endswith('.tfrecord.gz')]
    imageFilesList = [os.path.join(image_folder, f) for f in filesList]

    print("filesList:", imageFilesList)

    # 假设 JSON 文件在本地，也从本地读取
    jsonFile = None  # 如果有 json 文件需要处理，可改为本地路径

    # 加载 json 文件
    jsonText = open(jsonFile, 'r').read()
    mixer = json.loads(jsonText)
    patches = mixer['totalPatches']

    # 设置数据处理缓冲区
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)
    buffered_shape = [
        KERNEL_SHAPE[0] + kernel_buffer[0],
        KERNEL_SHAPE[1] + kernel_buffer[1]
    ]

    imageColumns = [
        tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32)
        for k in BANDS
    ]

    imageFeaturesDict = dict(zip(BANDS, imageColumns))

    # 解析 TFRecord
    def parse_image(example_proto):
        return tf.io.parse_single_example(example_proto, imageFeaturesDict)

    def toTupleImage(inputs):
        inputsList = [inputs.get(key) for key in BANDS]
        date_keys = list(inputs.keys())[::5]  # 每5个张量为一组
        stacked_tensors = []

        for key in date_keys:
            date_tensors = [
                inputs[key],
                inputs[key.replace('_B2', '_B3')],
                inputs[key.replace('_B2', '_B4')],
                inputs[key.replace('_B2', '_B8')],
                inputs[key.replace('_B2', '_days')]
            ]
            stacked_tensors.append(tf.stack(date_tensors, axis=0))

        stacked = tf.stack(stacked_tensors, axis=0)
        stacked = tf.transpose(stacked, [0, 2, 3, 1])  # 转换为(12, 192, 192, 5)
        return stacked

    # 创建 TFRecord 数据集
    imageDataset = tf.data.TFRecordDataset(imageFilesList, compression_type='GZIP')
    imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
    imageDataset = imageDataset.map(toTupleImage).batch(1)

    # 转换为 PyTorch Tensor
    def convert_to_pytorch(inputs):
        np_data = inputs.numpy()
        torch_data = torch.from_numpy(np_data.transpose(1, 12, 192, 192, 5))  # 转置维度为 (1, 12, 192, 192, 5)
        return torch_data

    print('Running predictions...')
    predictions = []

    # 预测循环
    # for data in imageDataset:
    #     input_data = convert_to_pytorch(data)
    #     out = get_out(input_data, input_data[:, :, :4, ::], model)  # 假设 model 已定义
    #
    #     pre = out.permute(0, 1, 3, 4, 2)[0]
    #     predictions.append(pre)
    #     del input_data, out, pre

    # 堆叠预测结果
    predictions = np.stack(predictions, axis=0)
    print("predictions.shape:", predictions.shape)


# 模型推理的 get_out 函数（假设你已经定义了模型）
def get_out(images, targets):
    T = images.shape[1]
    min_length = 5
    max_length = 10
    out = torch.zeros(targets.shape, dtype=torch.float32).float()

    if T < min_length:
        print(f"序列长度小于{min_length}")
        return None
    elif T < max_length:
        padding_length = max_length - T
        padding_value = images[:, -1:, ...]
        padding = padding_value.repeat(1, padding_length, 1, 1, 1)
        images = torch.cat((images, padding), dim=1)

    # 这里对每个时间步的图像进行裁剪操作，裁剪为 128x128
    cropped_images = images[:, :, :, :128, :128]  # 裁剪前 128 行和 128 列

    # 对目标标签进行裁剪，假设目标标签也需要裁剪
    cropped_targets = targets[:, :, :, :128, :128]

    # 返回裁剪后的图像和标签，保持与输入相同的格式
    return cropped_images, cropped_targets

    # out.shape: torch.Size([1, 15, 4, 128, 128])
    # return out

# 遍历子区域并调用 doPrediction 方法
for i in range(length):
    name = subregion_collection_list[i]
    print(name)
    doPrediction(image_name_prefix + name + "_", "path_to_assets", kernel_buffer, name)  # 资产路径（asset_folder）不需要修改