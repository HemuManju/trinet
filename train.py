import os
from datetime import date, datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import webdataset as wbs

import torch
import pytorch_lightning as pl

from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers

from src.dataset.sample_processors import rnn_samples, test_samples
from src.dataset import imitation_dataset
from src.dataset.imitation_dataset import concatenate_aux_samples
from src.dataset.utils import (
    show_image,
    get_webdataset_data_iterator,
)

from src.architectures.nets import (
    CARNet,
    CARNetExtended,
    CNNAutoEncoder,
    AutoRegressorBranchNet,
    CIRLBasePolicyAuxKarnet,
    CIRLBasePolicyAux,
    CIRLWaypointPolicy,
    AuxNet,
    SemanticAuxNet,
)

from src.models.imitation import Imitation, AuxiliaryTraining
from src.models.kalman import ExtendedKalmanFilter
from src.models.utils import load_checkpoint, number_parameters
from src.evaluate.agents import PIDCILAgent, PIDKalmanAgent
from src.evaluate.experiments import CORL2017

from src.dataset.utils import (
    show_image,
    WebDatasetReader,
    get_webdataset_data_iterator,
    get_specific_webdataset_data_iterator,
    offset_points,
    resample_waypoints,
)

from benchmark.run_benchmark import Benchmarking
from benchmark.summary import summarize

from captum.attr import LayerFeatureAblation

import yaml
from utils import skip_run, get_num_gpus


with skip_run('skip', 'dataset_analysis') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/auxnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Dataset reader
    reader = WebDatasetReader(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla-data/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000002.tar',
    )
    dataset = reader.get_dataset(concat_n_samples=1)
    distance_to_vehicle = []

    for i, data in enumerate(dataset):
        data = data['json'][0]
        try:
            distance_to_vehicle.append(data['dist_to_vehicle']['value'])
        except:
            distance_to_vehicle.append(data['dist_to_vehicle'])

        # if i > 1000:
        #     break
    distance_to_vehicle = np.array(distance_to_vehicle)
    print(np.histogram_bin_edges(distance_to_vehicle, bins=3))
    fig, ax = plt.subplots()
    plt.hist(distance_to_vehicle)
    plt.show()

with skip_run('skip', 'auxillary_net_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/auxnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/AUXILIARY'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    net = AuxNet(cfg)
    actions, traffic, vehicle_dist = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(
        cfg, sample_process=concatenate_aux_samples
    )
    if cfg['check_point_path'] is None:
        model = AuxiliaryTraining(cfg, net, data_loader)
    else:
        model = AuxiliaryTraining.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    if cfg['slurm']:
        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            strategy="ddp",
            num_nodes=1,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
        )

    trainer.fit(model)

with skip_run('skip', 'imitation_with_aux_base_policy_attn') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION_AUX_BASE'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup
    # Load the backbone network
    read_path = 'logs/2023-07-01/AUXILIARY/last.ckpt'
    auxnet = AuxNet(cfg)
    cfg['auxnet'] = load_checkpoint(auxnet, checkpoint_path=read_path)

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    cfg['action_net'] = action_net

    # Base Policy
    base_policy = CIRLWaypointPolicy(cfg)
    cfg['base_policy'] = base_policy

    # Over all network
    net = CIRLBasePolicyAux(cfg)
    net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    if cfg['slurm']:
        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            strategy="ddp",
            num_nodes=1,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
        )

    trainer.fit(model)

with skip_run('skip', 'imitation_with_base_aux_karnet_policy_conv') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = (
        cfg['logs_path'] + str(date.today()) + '/IMITATION_AUX_KARNET_BASE'
    )

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup

    # Load the karnet
    read_path = 'logs/2023-01-03/CARNET_KALMAN/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNetExtended(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['karnet'] = carnet

    # Kalmnn filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Load the aux network
    read_path = 'logs/2023-07-01/AUXILIARY/last.ckpt'
    auxnet = AuxNet(cfg)
    cfg['auxnet'] = load_checkpoint(auxnet, checkpoint_path=read_path)

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    cfg['action_net'] = action_net

    # Base Policy
    base_policy = CIRLWaypointPolicy(cfg)
    cfg['base_policy'] = base_policy

    # Over all network
    net = CIRLBasePolicyAuxKarnet(cfg)
    net(net.example_input_array, net.example_command, net.example_kalman)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    if cfg['slurm']:
        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            strategy="ddp",
            num_nodes=1,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )

    trainer.fit(model)

with skip_run('skip', 'imitation_with_base_aux_seg_karnet_conv') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = (
        cfg['logs_path'] + str(date.today()) + '/IMITATION_AUX_KARNET_BASE'
    )

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    if cfg['slurm']:
        cfg['raw_data_path'] = cfg['raw_data_path_slurm'] + f'/{navigation_type}'
    else:
        cfg['raw_data_path'] = cfg['raw_data_path_local'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup the network
    # Load the karnet
    read_path = 'logs/2023-10-02/CARNET_KALMAN/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNetExtended(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['karnet'] = carnet

    # Kalmnn filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Load the aux network
    read_path = 'logs/2023-08-26/AUXILIARY/last.ckpt'
    auxnet = SemanticAuxNet(cfg)
    cfg['auxnet'] = load_checkpoint(auxnet, checkpoint_path=read_path)

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    cfg['action_net'] = action_net

    # Base Policy
    base_policy = CIRLWaypointPolicy(cfg)
    cfg['base_policy'] = base_policy

    # Over all network
    net = CIRLBasePolicyAuxKarnet(cfg)
    net(net.example_input_array, net.example_command, net.example_kalman)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    if cfg['slurm']:
        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            strategy="ddp",
            num_nodes=1,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )

    trainer.fit(model)

with skip_run('skip', 'benchmark_trained_aux_karnet_base') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    ekf = ExtendedKalmanFilter(cfg)
    experiment_suite = CORL2017(experiment_cfg, ekf)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    model_date = '2023-10-11'

    if model_date == '2023-09-21':
        cfg['NORMALIZE_WEIGHT'] = False
        cfg['INCLUDE_SPEED'] = True
    elif model_date == '2023-09-22':
        cfg['NORMALIZE_WEIGHT'] = True
        cfg['INCLUDE_SPEED'] = True
    elif model_date == '2023-10-11':
        cfg['NORMALIZE_WEIGHT'] = True
        cfg['INCLUDE_SPEED'] = False
    elif model_date == '2023-09-24':
        cfg['NORMALIZE_WEIGHT'] = False
        cfg['INCLUDE_SPEED'] = False

    # Other parameters
    if cfg['NORMALIZE_WEIGHT']:
        constrained_type = 'constrained'
    else:
        constrained_type = 'unconstrained'

    if cfg['INCLUDE_SPEED']:
        speed_included = 'with_speed'
    else:
        speed_included = 'without_speed'

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):
        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'
        config['summary_writer'][
            'write_path'
        ] = f'logs/benchmark/{constrained_type}_{speed_included}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        config['summary_writer'][
            'message'
        ] = f'base + aux + karnet {constrained_type} convolution  + {speed_included}'

        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNetExtended(cfg, cnn_autoencoder)
        cfg['karnet'] = carnet

        # Kalmnn filter
        cfg['ekf'] = ExtendedKalmanFilter(cfg)

        # Update the model
        # auxnet = AuxNet(cfg)
        auxnet = SemanticAuxNet(cfg)
        cfg['auxnet'] = auxnet

        # Action net
        action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
        cfg['action_net'] = action_net

        # Base Policy
        base_policy = CIRLWaypointPolicy(cfg)
        cfg['base_policy'] = base_policy

        restore_config = {
            'checkpoint_path': f'logs/{model_date}/IMITATION_AUX_KARNET_BASE/last.ckpt'
        }

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLBasePolicyAuxKarnet(cfg),
            data_loader=None,
        )

        agent = PIDKalmanAgent(model=model, config=cfg)

        # Setup the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)

        # Run the benchmark
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()

with skip_run('skip', 'benchmark_trained_carnet_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    experiment_suite = CORL2017(experiment_cfg)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):
        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-08-25/WARMSTART/{navigation_type}_last.ckpt'
        }

        # Setup
        # Load the backbone network
        read_path = f'logs/2022-07-07/IMITATION/imitation_{navigation_type}.ckpt'
        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNet(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLCARNet(cfg),
            data_loader=None,
        )

        # Change agent
        # agent = CILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)
        agent = PIDCILAgent(model, cfg)

        # Run the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()

with skip_run('skip', 'summarize_benchmark') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'
    # benchmark_dir = 'logs/benchmark/constrained_without_speed_15_10_2023_20_20_59'
    benchmark_dir = 'logs/benchmark/unconstrained_without_speed_11_10_2023_20_49_51'

    # towns = ['Town02', 'Town01']
    # weathers = ['ClearSunset', 'SoftRainNoon']
    # navigation_types = ['straight', 'one_curve', 'navigation']

    towns = ['Town02']
    weathers = ['SoftRainNoon']  #'MidRainyNoon',
    navigation_types = ['navigation']

    for i in range(5, 20):
        for town, weather, navigation_type in itertools.product(
            towns, weathers, navigation_types
        ):
            path = (
                f'{benchmark_dir}/{town}_{navigation_type}_{weather}/measurements.csv'
            )
            print('-' * 32)
            print(town, weather, navigation_type)
            summarize(path, i)

with skip_run('skip', 'benchmark_trained_model') as check, check():
    # Load the configuration
    navigation_type = 'one_curve'

    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)

    raw_data_path = cfg['raw_data_path']
    cfg['raw_data_path'] = raw_data_path + f'/{navigation_type}'

    restore_config = {'checkpoint_path': 'logs/2022-06-06/one_curve/warmstart.ckpt'}
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CARNet(cfg),
        data_loader=None,
    )
    model.freeze()
    model.eval()
    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, rnn_samples)

    for data in data_loader['training']:
        output = model(data[0][0:1], data[1][0:1])
        print(data[2][0:1])
        # print(torch.max(data[2][:, 0] / 20))
        print(output)
        print('-------------------')

with skip_run('skip', 'ablation_study') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['BATCH_SIZE'] = 1

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Setup the network
    # Load the karnet
    read_path = 'logs/2023-01-03/CARNET_KALMAN/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNetExtended(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['karnet'] = carnet

    # Kalmnn filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Load the aux network
    read_path = 'logs/2023-08-26/AUXILIARY/last.ckpt'
    auxnet = SemanticAuxNet(cfg)
    cfg['auxnet'] = load_checkpoint(auxnet, checkpoint_path=read_path)

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    cfg['action_net'] = action_net

    # Base Policy
    base_policy = CIRLWaypointPolicy(cfg)
    cfg['base_policy'] = base_policy

    # Over all network
    net = CIRLBasePolicyAuxKarnet(cfg)
    net(net.example_input_array, net.example_command, net.example_kalman)

    restore_config = {
        'checkpoint_path': f'logs/2023-08-28/IMITATION_AUX_KARNET_BASE/last.ckpt'
    }
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=net,
        data_loader=None,
    )

    # Datareader
    navigation_type = 'navigation'
    town = 'Town01'
    weather = 'HardRainSunset'
    path = f'/home/hemanth/Desktop/carla-data-test/{navigation_type}/{town}_{weather}_cautious_000000.tar'
    data_loader = get_specific_webdataset_data_iterator(cfg, path, test_samples)

    # Ablator
    ablator = LayerFeatureAblation(model.net, model.net.combine_conv)
    layer_mask = torch.tensor([[[0], [1], [2], [3]]])
    shorterm_pred = []

    longterm_pred = []
    location = []
    t = 500

    for images, command, kalman, loc in data_loader:
        attr = ablator.attribute(
            inputs=(images, command, kalman),
            attribute_to_layer_input=True,
            layer_mask=layer_mask,
        )

        temp = torch.mean(attr, dim=-1)
        shorterm_pred.append(
            torch.abs(temp[0, :].unsqueeze(dim=-1))
            + torch.abs(temp[1, :].unsqueeze(dim=-1))
        )
        longterm_pred.append(
            torch.abs(temp[2, :].unsqueeze(dim=-1))
            + torch.abs(temp[3, :].unsqueeze(dim=-1))
        )

        location.append([ele.numpy() for ele in loc])
        if len(location) >= t:
            break

    # Convert to array
    shorterm_pred = np.concatenate(shorterm_pred, axis=1)
    longterm_pred = np.concatenate(longterm_pred, axis=1)
    location = np.array(location)[:, :, 0]
    new_location = resample_waypoints(location[:, 0:2], location[0, 0:2], resample=True)

    # Plotting
    import matplotlib.pyplot as plt

    plt.style.use('clean')
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    label = {
        0: r'$δ_{t}$',
        1: r'$α_{t}$',
        2: r'$ℓ_{t}$',
        3: r'$ℓ_{t+1}$',
    }
    i = 0
    for i in range(shorterm_pred.shape[0]):
        if i == 1:
            t = offset_points(new_location, distance=3)
        elif i == 2:
            t = offset_points(new_location, distance=-3)
        elif i == 3:
            t = offset_points(new_location, distance=6)
        else:
            t = new_location
        axs[0].scatter(
            t[:, 0], t[:, 1], s=shorterm_pred[i, 0 : len(t)] * 50, label=label[i]
        )
        axs[1].scatter(
            t[:, 0], t[:, 1], s=longterm_pred[i, 0 : len(t)] * 50, label=label[i]
        )

    axs[0].set_title('Near Waypoint Prediction')
    axs[1].set_title('Far Waypoint Prediction')

    axs[0].grid()
    axs[1].grid()

    plt.autoscale(enable=True)
    plt.rcParams['axes.grid'] = True
    plt.legend()
    plt.show()
