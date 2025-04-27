import os
import logging
from random import sample
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("__name__", )

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

def visualize_samples(dataset_csv_path: str, samples_per_class: int = 10):
    classes = ["both", "none", "dog", "person"]
    dataset = pd.read_csv(dataset_csv_path)
    
    both = dataset.loc[(dataset["dog"] == 1) & (dataset["person"] == 1)]["image"]
    none = dataset.loc[(dataset["dog"] == 0) & (dataset["person"] == 0)]["image"]
    dog = dataset.loc[(dataset["dog"] == 1) & (dataset["person"] == 0)]["image"]
    person = dataset.loc[(dataset["dog"] == 0) & (dataset["person"] == 1)]["image"]
    
    px = 1/plt.rcParams['figure.dpi']  # pixel to inches conversion
    
    sample_image = load_image(dataset.iloc[0]["image"]).numpy()
    img_padding, scale = (2, 1)
    img_width, img_height = sample_image.shape[:-1]
    width = (img_width + img_padding) * scale * samples_per_class
    height = (img_height + img_padding) * scale * len(classes)
    
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(width*px, height*px))
    
    both_images = sample(load_images(images=both), samples_per_class)
    none_images = sample(load_images(images=none), samples_per_class)
    dog_images = sample(load_images(images=dog), samples_per_class)
    person_images = sample(load_images(images=person), samples_per_class)
    
    all_samples = both_images + none_images + dog_images + person_images
    axes = axes.flatten()
    for idx, image in enumerate(all_samples):
        class_title = classes[idx//samples_per_class]
        ax = axes[idx]
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(class_title)
    return fig

def visualize_errors(image_list: list[tf.Tensor], class_idx_list: list[int], rows: int = 4, cols: int = 4):
    class_map = {
                    0: "Dog",
                    1: "Person"
                }
    rows, cols = int(rows), int(cols)
    image_count = len(image_list)
    if image_count != rows * cols:
        new_rows = int(np.ceil(image_count / cols))
        logger.warning(f"{image_count} images won't fit in {rows}x{cols} grid, using {new_rows} rows instead.")
        rows = new_rows
        
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    axes = axes.flatten()

    # Iterate over images and axes to plot each image
    for i, ax in enumerate(axes):
        if i < image_count:  # If there are fewer images than the grid
            ax.imshow(tf.squeeze(image_list[i]))  # Remove any extra dimensions like batch dimension
            ax.axis('off')  
            ax.set_title(f"Missed on class: {class_map[class_idx_list[i]]}")
        else:
            ax.axis('off')  # Complete with empty subplots
            
    return fig