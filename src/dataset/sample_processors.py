import numpy as np
import torch

from torchvision import transforms


from .preprocessing import get_preprocessing_pipeline, RotationTransform
from .utils import get_dataset_paths, generate_seqs, show_tensor_image


def rnn_semseg_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Get the segmentation labels
    semseg_labels = []
    for item in combined_data['json']:
        semseg = torch.tensor(item['semseg']).reshape(config['image_size'][1:]).long()
        semseg = transforms.Resize(
            size=(config['image_resize'][1], config['image_resize'][2])
        )(semseg[None, None, ...])

        # Rotate them semseg labels
        semseg_labels.append(RotationTransform(angles=-90)(semseg))

    # Convert to stacked tensor
    semseg_labels = torch.cat(semseg_labels, dim=0)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]
        semseg_labels = semseg_labels[:, :, :crop_size, :]

    input_seq = images[0:-1, :, :, :]
    semseg_labels = semseg_labels[1:, 0, :, :]
    return input_seq, semseg_labels


def rnn_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]

    input_seq = images[0:-1, :, :, :]
    output_seq = images[1:, :, :, :]
    return input_seq, output_seq


def rnn_samples_with_kalman(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    kalman_updates = []

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]

    # Kalman update
    for data in combined_data['json'][:-1]:
        updates = config['ekf'].update(data)
        kalman_updates.append(transforms.ToTensor()(updates))

    input_seq = images[0:-1, :, :, :]
    output_seq = images[1:, :, :, :]
    return input_seq, output_seq, torch.stack(kalman_updates)


def semseg_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Get the segmentation labels
    semseg_labels = combined_data['json'][0]['semseg']
    semseg_labels = torch.tensor(semseg_labels).reshape(config['image_size'][1:]).long()
    semseg_labels = transforms.Resize(
        size=(config['image_resize'][1], config['image_resize'][2])
    )(semseg_labels[None, None, ...])

    # Rotate them semseg labels
    semseg_labels = RotationTransform(angles=-90)(semseg_labels)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]
        semseg_labels = semseg_labels[:, :, :crop_size, :]

    images = images[0, :, :, :]
    semseg_labels = semseg_labels[0, 0, :, :]

    return images, semseg_labels


def semseg_aux_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Get the segmentation labels
    semseg_labels = combined_data['json'][0]['semseg']
    semseg_labels = torch.tensor(semseg_labels).reshape(config['image_size'][1:]).long()
    semseg_labels = transforms.Resize(
        size=(config['image_resize'][1], config['image_resize'][2])
    )(semseg_labels[None, None, ...])

    # Rotate them semseg labels
    semseg_labels = RotationTransform(angles=-90)(semseg_labels)
    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]
        semseg_labels = semseg_labels[:, :, :crop_size, :]

    images = images[0, :, :, :]
    semseg_labels = semseg_labels[0, 0, :, :]

    data = samples[-1]['json']
    try:
        dist_to_vehilce = data['dist_to_vehicle']['value']
    except:
        dist_to_vehilce = data['dist_to_vehicle']
    distance_to_vehicle = np.digitize(dist_to_vehilce, bins=[-1.0, 10, 20, 30, 40, 100])

    if isinstance(distance_to_vehicle, np.int64):
        distance_to_vehicle = np.asarray(distance_to_vehicle * 1.0)

    traffic = {-1: 0.0, 'red': 1.0}
    traffic_light = traffic[data['traffic_light_state']]
    action = (
        torch.tensor(data['speed']),
        torch.from_numpy(np.asarray(traffic_light)).long(),
        torch.from_numpy(distance_to_vehicle).long(),
    )
    return images, semseg_labels, action


def one_image_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]

    # Convert the sequence to input and output
    input_seq = images[0, :, :, :]
    output_seq = images[0, :, :, :]

    return input_seq, output_seq
