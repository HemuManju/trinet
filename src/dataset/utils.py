import os
import collections
from pathlib import Path
from itertools import islice, cycle, product

import math

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

import natsort

import webdataset as wds


def get_webdataset_data_iterator(config, sample_processors):

    # Get dataset path(s)
    paths = get_dataset_paths(config)

    # Parameter(s)
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['seq_length'] + config['predict_length']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .then(generate_seqs, sample_processors, SEQ_LEN, config)
            )
            data_loader = wds.WebLoader(
                dataset,
                num_workers=number_workers,
                shuffle=False,
                batch_size=BATCH_SIZE,
            )
            if key in ['training', 'validation']:
                dataset_size = 6250 * len(path)
                data_loader.length = dataset_size // BATCH_SIZE

            data_iterator[key] = data_loader

    return data_iterator


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 255],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0],  # TrafficSigns
    }
    result = np.zeros((image.shape[0], image.shape[1], 3))
    for key, value in classes.items():
        result[np.where(image == key)] = value

    return result.astype(np.uint8)


def show_image(img, ax):
    # npimg = img.numpy()
    ax.imshow(transforms.ToPILImage()(img), origin='lower')
    # plt.show()


def nested_dict():
    return collections.defaultdict(nested_dict)


def generate_seqs(src, process_samples, nsamples=3, config=None):
    it = iter(src)
    result = tuple(islice(it, nsamples))
    if len(result) == nsamples:
        yield process_samples(result, config)
    for elem in it:
        result = result[1:] + (elem,)
        yield process_samples(result, config)


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)


def get_dataset_paths(config):
    paths = {}
    data_split = config['data_split']
    read_path = config['raw_data_path']
    for key, split in data_split.items():
        combinations = [
            '_'.join(item)
            for item in list(product(split['town'], split['season'], split['behavior']))
        ]

        # Get all the tar files
        temp = [find_tar_files(read_path, combination) for combination in combinations]

        # Concatenate all the paths and assign to dict
        paths[key] = sum(temp, [])  # Not a good way, but it is fun!
    return paths


def run_fast_scandir(dir, ext, logs=None):  # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_image_json_files(read_path):
    # Read image files and sort them
    _, file_list = run_fast_scandir(read_path, [".jpeg"])
    image_files = natsort.natsorted(file_list)

    # Read json files and sort them
    _, file_list = run_fast_scandir(read_path, [".json"])
    json_files = natsort.natsorted(file_list)
    return image_files, json_files


def find_in_between_angle(v, w):
    theta = math.atan2(np.linalg.det([v[0:2], w[0:2]]), np.dot(v[0:2], w[0:2]))
    return theta


def rotate(points, angle):
    R = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = R.dot(points)
    return rotated


class WebDatasetReader:
    def __init__(self, config, file_path) -> None:
        self.file_path = file_path
        self.cfg = config
        self.sink = None

    def _process_samples(self, samples):
        combined_data = {
            k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
        }
        return combined_data

    def _generate_seqs(self, src, nsamples=3):
        it = iter(src)
        result = tuple(islice(it, nsamples))
        if len(result) == nsamples:
            yield self._process_samples(result)
        for elem in it:
            result = result[1:] + (elem,)
            yield self._process_samples(result)

    def get_dataset(self, concat_n_samples=None):
        if concat_n_samples is None:
            dataset = wds.WebDataset(self.file_path).decode("torchrgb")
        else:
            dataset = (
                wds.WebDataset(self.file_path)
                .decode("torchrgb")
                .then(self._generate_seqs, concat_n_samples)
            )
        return dataset

    def get_dataloader(self, num_workers, batch_size, concat_n_samples=None):
        # Get the dataset
        dataset = self.get_dataset(concat_n_samples=concat_n_samples)
        data_loader = wds.WebLoader(
            dataset, num_workers=num_workers, shuffle=False, batch_size=batch_size,
        )
        return data_loader
