import os
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

def load_image(image_path: str)-> tf.Tensor:
    parent_folder_name = "modified_voc"
    image_path = os.path.join(parent_folder_name, image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_images(images: pd.Series)-> list[tf.Tensor]:
    image_list = []
    for _, image_path in tqdm(images.items(), total=images.size):
        image_list.append(load_image(image_path))
    return image_list