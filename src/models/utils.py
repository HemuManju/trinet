import os
import random
import signal
import subprocess
import time
import psutil

import math

import collections


import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import carla
except ModuleNotFoundError:
    pass

import torch
import torch.nn as nn


def calc_ssim_kernel_size(image_shape, levels):
    k = math.floor((image_shape[1] - 1) / 2 ** (levels - 1) + 1)
    return k - 1 if k % 2 == 0 else k


def train(model, data_loader, optimizer, criterion, device=None):

    # Setup train and device
    device = device or torch.device("cpu")
    model.train()

    # Metrics
    running_loss = 0.0
    epoch_steps = 0

    for data, target in data_loader:

        # get the inputs; data is a list of [inputs, labels]
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1

    return running_loss, epoch_steps


def test(model, data_loader, criterion, device=None):
    # Setup eval and device
    device = device or torch.device("cpu")
    model.eval()
    predicted = []
    targets = []
    losses = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            # Append data
            targets.append(target)
            predicted.append(outputs)
            losses.append(loss)

    return predicted, targets, losses


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def load_checkpoint(net, checkpoint_path, only_weights=False, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not only_weights:
        checkpoint = checkpoint['state_dict']
    # NOTE: This is a hacky way to loading the net weights
    # TODO: Find a better way to do
    target_dict = {
        k.replace('net.', '', 1): checkpoint[k]
        for k in checkpoint.keys()
        if k.startswith('net.')
    }
    net.load_state_dict(target_dict, strict=strict)
    return net


def number_parameters(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    m(m.example_input_array, m.example_command)
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def save_to_onnx(model, example_input, name='model'):
    model.to_onnx(f'{name}.onnx', example_input, export_params=True)


class CarlaServer:
    def __init__(self, config) -> None:
        BASE_CORE_CONFIG = {
            "host": "localhost",  # Client host
            "timeout": 10.0,  # Timeout of the client
            "timestep": 0.05,  # Time step of the simulation
            "retries_on_error": 10,  # Number of tries to connect to the client
            "resolution_x": 600,  # Width of the server spectator camera
            "resolution_y": 600,  # Height of the server spectator camera
            "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
            "enable_map_assets": False,  # enable / disable all town assets except for the road
            "enable_rendering": True,  # enable / disable camera images
            "show_display": False,  # Whether or not the server will be displayed
        }

        self.config = join_dicts(BASE_CORE_CONFIG, config)

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = 2000  #  random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + str(self.server_port))
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl",  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "-world-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"]),
            "-benchmark",
            # "-fps=10",
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        self.process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(
                    " Waiting for server to be ready: {}, attempt {} of {}".format(
                        e, i + 1, self.config["retries_on_error"]
                    )
                )
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration"
        )

    def start_server(self):
        """Start the server

        Returns
        -------
        str, float
            host and server port
        """
        self.init_server()
        # self.connect_client()
        return self.config["host"], self.server_port


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(
        self,
        gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
        data_range=1.0,
        K=(0.01, 0.03),
        alpha=0.025,
        compensation=200.0,
        cuda_dev=0,
    ):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-3, length=3),
            groups=3,
            padding=self.pad,
        ).mean(
            1
        )  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        """
        The inputs are sets of d-dimensional points:
        x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.
        Arguments:
            x: a float tensor with shape [b, d, n].
            y: a float tensor with shape [b, d, m].
        Returns:
            a float tensor with shape [].
        """
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # compute pairwise l2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)

