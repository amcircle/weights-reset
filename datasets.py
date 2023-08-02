import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from enum import Enum

class Dataset(Enum):
    CALTECH_101 = 'caltech101'
    CIFAR_100 = 'cifar100'
    IMAGENETTE = 'imagenette/320px-v2'

def apply_functions(data, functions):
    """
    Applies a chain of functions to the input data sequentially.
    
    Args:
        data: The input data to apply the functions to.
        functions: A list of functions to apply to the data, in the order that they should be applied.
        
    Returns:
        The result of applying all of the functions to the input data.
    """
    result = data
    for function in functions:
        result = function(result)
    return result

def transform_data(x, num_classes):
    """
    Transforms the input data dictionary by normalizing the 'image' field and converting the 'label' field 
    to a one-hot encoded tensor.

    Args:
        x: A dictionary containing the input data, with keys 'image' and 'label'.
        num_classes: An integer indicating the number of classes for the one-hot encoding of the 'label' field.

    Returns:
        A tuple containing the transformed data, with the 'image' field normalized to [0,1] and the 'label' 
        field converted to a one-hot encoded tensor of shape (num_classes,).
    """
    return (tf.cast(x["image"], tf.float32) / 255.0, 
            tf.one_hot(x["label"], num_classes, dtype=tf.float32))

def resize_im(im, label, height, width):
    """
    Resizes the input image tensor to the specified height and width, and returns the result along with the original label.

    Args:
        im: A tensor representing an input image.
        label: The label corresponding to the input image.
        height: An integer indicating the desired height of the resized image.
        width: An integer indicating the desired width of the resized image.

    Returns:
        A tuple containing the resized image tensor of shape (height, width, channels), and the original label.
    """
    return tf.image.resize(im, [height, width]), label

def load_dataset(ds, batch_size = 16):
    """
    Loads and pre-processes a dataset specified by name, returning train and test datasets, as well as height, width, and number of classes.

    Args:
        name: A string indicating the name of the dataset to load.
        batch_size: An integer indicating the batch size to use for the train and test datasets (default: 16).

    Returns:
        A tuple containing the train and test datasets, as well as the shape (height,width,channels) of the images in the dataset,
        and the number of classes in the dataset.
    """
    
    name = ds.value
    
    num_classes = {
        'caltech101': 102,
        'cifar100': 100,
        'imagenette/320px-v2': 10
    }[name]
    
    height_width = {
        'caltech101': [300, 200],
        'cifar100': [32, 32],
        'imagenette/320px-v2': [300, 200]
    }[name]
    
    im_channels = {
        'caltech101': 3,
        'cifar100': 3,
        'imagenette/320px-v2': 3
    }[name]
    
    preproc_steps = {
        'caltech101': [
            lambda x: transform_data(x, num_classes),
            lambda x: resize_im(x[0], x[1], height_width[0], height_width[1])
        ],
        'cifar100': [
            lambda x: transform_data(x, num_classes),
        ],
        'imagenette/320px-v2': [
            lambda x: transform_data(x, num_classes),
            lambda x: resize_im(x[0], x[1], height_width[0], height_width[1])
        ],
    }[name]
    
    test_split_name = {
        'caltech101': 'test',
        'cifar100': 'test',
        'imagenette/320px-v2': 'validation'
    }[name]
    
    dataset_train = tfds.load(name, split='train', shuffle_files=True)
    dataset_test = tfds.load(name, split=test_split_name, shuffle_files=False)
    
    dataset_train = (dataset_train
        .map(lambda x: apply_functions(x, preproc_steps))
        .shuffle(len(dataset_train), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    dataset_test = (dataset_test
        .map(lambda x: apply_functions(x, preproc_steps))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
        
    return dataset_train, dataset_test, height_width + [im_channels], num_classes