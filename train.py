import os
from datetime import date
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import webdataset as wbs

import torch
import pytorch_lightning as pl

from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers

from src.dataset.sample_processors import rnn_samples
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
    CIRLBasePolicyKARNet,
    CIRLWaypointPolicy,
    AuxNet,
)

from src.models.imitation import Imitation, AuxiliaryTraining
from src.models.kalman import ExtendedKalmanFilter
from src.models.utils import load_checkpoint, number_parameters
from src.evaluate.agents import PIDCILAgent, PIDKalmanAgent
from src.evaluate.experiments import CORL2017

from src.dataset.utils import WebDatasetReader

from benchmark.run_benchmark import Benchmarking
from benchmark.summary import summarize


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

with skip_run('run', 'auxillary_net_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/auxnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION_KALMAN'

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
            cfg['check_point_path'], hparams=cfg, net=net, data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'imitation_with_kanet_base_policy_attn') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION_KALMAN'

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
    read_path = 'logs/2023-01-03/CARNET_KALMAN/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNetExtended(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    cfg['action_net'] = action_net

    # Kalmnn filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Base Policy
    base_policy = CIRLWaypointPolicy(cfg)
    cfg['base_policy'] = base_policy

    # Over all network
    net = CIRLBasePolicyKARNet(cfg)
    net(net.example_input_array, net.example_command, net.example_kalman)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'], hparams=cfg, net=net, data_loader=data_loader,
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

with skip_run('skip', 'benchmark_trained_aux_base') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    cfg = yaml.load(open('configs/auxnet.yaml'), Loader=yaml.SafeLoader)
    ekf = ExtendedKalmanFilter(cfg)
    experiment_suite = CORL2017(experiment_cfg, ekf)

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

        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNetExtended(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        # Action net
        action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
        # read_path = 'logs/action_net.pt'
        # action_net = load_checkpoint(
        #     action_net, checkpoint_path=read_path, only_weights=True, strict=False
        # )
        cfg['action_net'] = action_net

        # Base Policy
        read_path = 'logs/2022-10-15/IMITATION/last.ckpt'
        base_policy = CIRLWaypointPolicy(cfg)
        # base_policy = load_checkpoint(
        #     base_policy, checkpoint_path=read_path, strict=False
        # )
        cfg['base_policy'] = base_policy

        restore_config = {
            'checkpoint_path': f'logs/2023-05-17/IMITATION_KALMAN/last.ckpt'
        }

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLBasePolicyKARNet(cfg),
            data_loader=None,
        )

        agent = PIDKalmanAgent(model=model, config=cfg)

        # Setup the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)

        # Run the benchmark
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()(model, cfg)

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

with skip_run('run', 'summarize_benchmark') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # towns = ['Town02', 'Town01']
    # weathers = ['ClearSunset', 'SoftRainNoon']
    # navigation_types = ['straight', 'one_curve', 'navigation']

    towns = ['Town01']
    weathers = ['SoftRainNoon']  #'ClearSunset',
    navigation_types = ['navigation']

    for town, weather, navigation_type in itertools.product(
        towns, weathers, navigation_types
    ):
        path = f'logs/benchmark_results/{town}_{navigation_type}_{weather}_3/measurements.csv'
        print('-' * 32)
        print(town, weather, navigation_type)
        summarize(path)

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
