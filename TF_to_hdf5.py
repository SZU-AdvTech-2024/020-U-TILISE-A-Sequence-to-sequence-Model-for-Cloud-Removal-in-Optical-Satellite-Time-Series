import tensorflow as tf
import h5py
import numpy as np
import os

# 设置波段
date_list = ['01-30', '02-19', '03-02', '04-09', '05-04', '06-28', '07-18', '08-12', '09-01', '10-26', '11-25', '12-30']
BANDS = []
for date in date_list:
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B2")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B3")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B4")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_B8")
    BANDS.append("shenzhen_mosaic_2020-" + date + "_days")

# 假设使用的图像大小是 192x192，波段数量为 5（每个日期有5个波段）
KERNEL_SIZE = 128
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
kernel_buffer = [KERNEL_SIZE // 2, KERNEL_SIZE // 2]  # [64, 64]


# 创建一个空的 HDF5 文件
def create_hdf5_file(output_hdf5_path):
    with h5py.File(output_hdf5_path, 'w') as hf:
        # 创建一个数据集用于保存图像
        hf.create_dataset('images', (0, 12, 128, 128, 5), maxshape=(None, 12, 128, 128, 5), dtype='float32')
        hf.create_dataset('labels', (0, 12, 128, 128, 5), maxshape=(None, 12, 128, 128, 5), dtype='float32')


# 解析 TFRecord 数据
def parse_image(example_proto):
    imageColumns = [
        tf.io.FixedLenFeature(shape=[KERNEL_SIZE + kernel_buffer[0], KERNEL_SIZE + kernel_buffer[1]], dtype=tf.float32)
        for _ in BANDS
    ]
    imageFeaturesDict = dict(zip(BANDS, imageColumns))
    return tf.io.parse_single_example(example_proto, imageFeaturesDict)


def to_tuple_image(inputs):
    # 将输入转换为合适的形状
    inputs_list = [inputs.get(key) for key in BANDS]

    # 将不同日期的波段堆叠成 5 个通道（B2, B3, B4, B8, days）
    stacked_tensors = []
    for key in BANDS[::5]:
        date_tensors = [inputs[key], inputs[key.replace('_B2', '_B3')], inputs[key.replace('_B2', '_B4')],
                        inputs[key.replace('_B2', '_B8')], inputs[key.replace('_B2', '_days')]]
        stacked_tensors.append(tf.stack(date_tensors, axis=0))

    stacked = tf.stack(stacked_tensors, axis=0)
    stacked = tf.transpose(stacked, [0, 2, 3, 1])

    # 裁剪每个图像为128x128
    cropped_images = tf.image.resize(stacked, [128, 128])  # 可以使用resize来做裁剪或缩放
    return cropped_images


# 读取 TFRecord 文件并保存为 HDF5
def convert_tfrecord_to_hdf5(tfrecord_dir, output_hdf5_path):
    # 获取 TFRecord 文件列表
    image_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord.gz')]
    image_files.sort()

    # 创建 HDF5 文件
    create_hdf5_file(output_hdf5_path)

    with h5py.File(output_hdf5_path, 'a') as hf:
        # 获取 dataset 对象用于存储图像和标签
        images_dataset = hf['images']
        labels_dataset = hf['labels']

        for tfrecord_file in image_files:
            # 读取 TFRecord 数据集
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            dataset = dataset.map(parse_image)
            dataset = dataset.map(to_tuple_image)

            for data in dataset:
                # 假设数据维度是 (12, 128, 128, 5)
                # 在此处，我们可以选择将 `data` 转化为你需要的任何形式（例如标签或图像）
                images = data.numpy()
                labels = images  # 这里假设标签和图像相同，你可以根据需要调整

                # 将数据追加到 HDF5 文件中
                images_dataset.resize(images_dataset.shape[0] + 1, axis=0)
                labels_dataset.resize(labels_dataset.shape[0] + 1, axis=0)
                images_dataset[-1:] = images
                labels_dataset[-1:] = labels


# 设置 TFRecord 文件目录和输出 HDF5 文件路径
tfrecord_dir = r'E:\shenzhen_mosaic\data'  # TFRecord 文件所在的目录
output_hdf5_path = r'E:\shenzhen_mosaic_data.h5'  # 输出的 HDF5 文件路径

# 执行转换.
convert_tfrecord_to_hdf5(tfrecord_dir, output_hdf5_path)
