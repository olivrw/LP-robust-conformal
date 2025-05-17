import tensorflow as tf
import tensorflow_datasets as tfds

import os

imagenet_builder = tfds.builder("imagenet2012", data_dir=os.path.join(
    os.getcwd(), "datasets", "ImageNet/tensorflow_datasets"
)
imagenet_builder.download_and_prepare()