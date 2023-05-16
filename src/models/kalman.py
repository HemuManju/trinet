# Author: Shing-Yan Loo (yan99033@gmail.com)
# Run extended Kalman filter to calculate the real-time vehicle location
#
# Credit: State Estimation and Localization for Self-Driving Cars by Coursera
#   Please consider enrolling the course if you find this tutorial helpful and
#   would like to learn more about Kalman filter and state estimation for
#   self-driving cars in general.

import math
import numpy as np
import xml.etree.ElementTree as ET


from .rotations import Quaternion, omega, skew_symmetric, angle_normalize


class ExtendedKalmanFilter:
    def __init__(self, cfg, town='Town01'):

        self.config = cfg

        # State (position, velocity and orientation)
        self.p = np.zeros([3, 1])
        self.v = np.zeros([3, 1])
        self.q = np.zeros([4, 1])  # quaternion

        # State covariance
        self.p_cov = np.zeros([9, 9])

        # Last updated timestamp (to compute the position
        # recovered by IMU velocity and acceleration, i.e.,
        # dead-reckoning)
        self.last_ts = 0

        # Gravity
        self.g = np.array([0, 0, -9.81]).reshape(3, 1)

        # Sensor noise variances
        self.var_imu_acc = 0.01
        self.var_imu_gyro = 0.01

        # Motion model noise
        self.var_gnss = np.eye(3) * 0.001
        self.var_v = np.eye(3) * 0.01

        self.measure_var = np.eye(6)
        self.measure_var[0:3, 0:3] = self.var_gnss
        self.measure_var[3:, 3:] = self.var_v

        # Motion model noise Jacobian
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        # Measurement model Jacobian
        self.h_jac = np.zeros([6, 9])
        self.h_jac[:, :6] = np.eye(6)

        # Initialized
        self.n_gnss_taken = 0
        self.gnss_init_xyz = None
        self.initialized = False
        self.town = town

        self.last_location = None

    def is_initialized(self):
        return self.initialized

    def initialize_with_gnss(self, gnss, samples_to_use=10):
        """Initialize the vehicle state using gnss sensor

        Note that this is going to be a very crude initialization by taking
        an average of 10 readings to get the absolute position of the car. A
        better initialization technique could be employed to better estimate
        the initial vehicle state

        Alternatively, you can also initialize the vehicle state using ground
        truth vehicle position and orientation, but this would take away the
        realism of the experiment/project

        :param gnss: converted absolute xyz position
        :type gnss: list
        """
        if self.gnss_init_xyz is None:
            self.gnss_init_xyz = np.array([gnss.x, gnss.y, gnss.z])
        else:
            self.gnss_init_xyz[0] += gnss.x
            self.gnss_init_xyz[1] += gnss.y
            self.gnss_init_xyz[2] += gnss.z
        self.n_gnss_taken += 1

        if self.n_gnss_taken == samples_to_use:
            self.gnss_init_xyz /= samples_to_use
            self.p[:, 0] = self.gnss_init_xyz
            self.q[:, 0] = Quaternion().to_numpy()

            # Low uncertainty in position estimation and high in orientation and
            # velocity
            pos_var = 1
            orien_var = 10
            vel_var = 1000
            self.p_cov[:3, :3] = np.eye(3) * pos_var
            self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
            self.p_cov[6:, 6:] = np.eye(3) * orien_var
            self.initialized = True

    def gnss_to_xyz(self, latitude, longitude, altitude):
        """Creates Location from GPS (latitude, longitude, altitude).
            This is the inverse of the _location_to_gps method found in
            https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py

            Modified from:
            https://github.com/erdos-project/pylot/blob/master/pylot/utils.py
            """
        EARTH_RADIUS_EQUA = 6378137.0

        scale = math.cos(self.gnss_lat_ref * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * self.gnss_long_ref
        basey = (
            scale
            * EARTH_RADIUS_EQUA
            * math.log(math.tan((90.0 + self.gnss_lat_ref) * math.pi / 360.0))
        )

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = (
            scale
            * EARTH_RADIUS_EQUA
            * math.log(math.tan((90.0 + latitude) * math.pi / 360.0))
            - basey
        )

        # This wasn't in the original method, but seems to be necessary.
        y *= -1

        return x, y, altitude

    def get_latlon_ref(self, town):
        """
        Convert from waypoints world coordinates to CARLA GPS coordinates
        :return: tuple with lat and lon coordinates
        https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        """
        xodr_path = f'data/Town01.xodr'
        tree = ET.parse(xodr_path)

        # default reference
        lat_ref = 42.0
        lon_ref = 2.0

        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    if georef.text:
                        str_list = georef.text.split(' ')
                        for item in str_list:
                            if '+lat_0' in item:
                                lat_ref = float(item.split('=')[1])
                            if '+lon_0' in item:
                                lon_ref = float(item.split('=')[1])
        return lat_ref, lon_ref

    def initialize_with_true_data(self, data):
        """Initialize the vehicle state using gtrue data

        Note that this is going to be a very crude initialization by taking
        an average of 10 readings to get the absolute position of the car. A
        better initialization technique could be employed to better estimate
        the initial vehicle state

        Alternatively, you can also initialize the vehicle state using ground
        truth vehicle position and orientation, but this would take away the
        realism of the experiment/project

        :param gnss: converted absolute xyz position
        :type gnss: list
        """

        # Lat long reference
        self.gnss_lat_ref, self.gnss_long_ref = self.get_latlon_ref(self.town)

        euler = [0, 0, 0]

        # Roll pitch and yaw
        euler[0], euler[1] = 0.0, 0.0

        v_vec = data['moving_direction']
        euler[2] = math.atan2(v_vec[1], v_vec[0])  # yaw

        self.p[:, 0] = data['location']
        self.q[:, 0] = Quaternion(euler=euler).to_numpy()

        # Low uncertainty in position estimation and high in orientation and
        # velocity
        pos_var = 1
        orien_var = 1
        vel_var = 1
        self.p_cov[:3, :3] = np.eye(3) * pos_var
        self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
        self.p_cov[6:, 6:] = np.eye(3) * orien_var
        self.initialized = True

    def get_location(self):
        """Return the estimated vehicle location

        :return: x, y, z position
        :rtype: list
        """
        return self.p.reshape(-1).tolist()

    def predict_state_with_imu(self, imu):
        """Use the IMU reading to update the car location (dead-reckoning)

        (This is equivalent to doing EKF prediction)

        Note that if the state is just initialized, there might be an error
        in the orientation that leads to incorrect state prediction. The error
        could be aggravated due to the fact that IMU is 'strapped down', and hence
        generating relative angular measurement (instead of absolute using IMU
        stabilized by a gimbal). Learn more in the Coursera course!

        The uncertainty (or state covariance) is going to grow larger and larger if there
        is no correction step. Therefore, the GNSS update would have a larger weight
        when performing the correction, and hopefully the state would converge toward
        the true state with more correction steps.

        :param imu: imu acceleration, velocity and timestamp
        :type imu: IMU blueprint instance (Carla)
        """
        # IMU acceleration and velocity
        imu_f = np.array(imu[0:3]).reshape(3, 1)
        imu_w = np.array(imu[3:6]).reshape(3, 1)

        # IMU sampling time
        delta_t = 0.1
        # self.last_ts = imu.timestamp

        # Update state with imu
        R = Quaternion(*self.q).to_mat()
        self.p = (
            self.p + delta_t * self.v + 0.5 * delta_t * delta_t * (R @ imu_f + self.g)
        )
        self.v = self.v + delta_t * (R @ imu_f + self.g)
        self.q = omega(imu_w, delta_t) @ self.q

        # Update covariance
        F = self._calculate_motion_model_jacobian(R, imu_f, delta_t)
        Q = self._calculate_imu_noise(delta_t)
        self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T

    def _calculate_motion_model_jacobian(self, R, imu_f, delta_t):
        """derivative of the motion model function with respect to the state

        :param R: rotation matrix of the state orientation
        :type R: NumPy array
        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:] = -skew_symmetric(R @ imu_f) * delta_t

        return F

    def _calculate_imu_noise(self, delta_t):
        """Calculate the IMU noise according to the pre-defined sensor noise profile

        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        :param imu_w: IMU xyz angular velocity
        :type imu_w: NumPy array
        """
        Q = np.eye(6)
        Q[:3, :3] *= delta_t * delta_t * self.var_imu_acc
        Q[3:, 3:] *= delta_t * delta_t * self.var_imu_gyro

        return Q

    def correct_state_with_gnss(self, data):
        """Given the estimated global location by gnss, correct
        the vehicle state

        :param gnss: global xyz position
        :type x: Gnss class (see car.py)
        """
        # Global position
        try:
            gnss = data['gnss']['values']
        except IndexError:
            gnss = data['gnss']
        x, y, z = self.gnss_to_xyz(gnss[0], gnss[1], gnss[2])
        v = data['velocity']

        # Kalman gain
        K = (
            self.p_cov
            @ self.h_jac.T
            @ (np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + self.measure_var))
        )

        # Compute the error state
        pose = np.vstack((self.p, self.v))
        delta_x = K @ (np.array([x, y, z, v[0], v[1], 0])[:, None] - pose)

        # Correction
        self.p = self.p + delta_x[:3]
        self.v = self.v + delta_x[3:6]
        delta_q = Quaternion(axis_angle=angle_normalize(delta_x[6:]))
        self.q = delta_q.quat_mult_left(self.q)

        # Corrected covariance
        self.p_cov = (np.identity(9) - K @ self.h_jac) @ self.p_cov

    def extract_states(self):

        if self.config['clip_kalman']:
            position = np.clip(self.p[0:2], a_min=-1, a_max=400)
            velocity = np.clip(self.v[0:2], a_min=-20, a_max=20)
        else:
            position = self.p[0:2]
            velocity = self.v[0:2]

        if self.config['normalize_kalman']:
            position = position / self.config['max_position']
            velocity = velocity / self.config['max_velocity']

        state = np.vstack((position, velocity))
        return state

    def reset(self):
        # State (position, velocity and orientation)
        self.p = np.zeros([3, 1])
        self.v = np.zeros([3, 1])
        self.q = np.zeros([4, 1])  # quaternion

        # State covariance
        self.p_cov = np.zeros([9, 9])

        # Last updated timestamp (to compute the position
        # recovered by IMU velocity and acceleration, i.e.,
        # dead-reckoning)
        self.last_ts = 0

        # Gravity
        self.g = np.array([0, 0, -9.81]).reshape(3, 1)

        # Sensor noise variances
        self.var_imu_acc = 0.01
        self.var_imu_gyro = 0.01

        # Motion model noise
        self.var_gnss = np.eye(3) * 0.001
        self.var_v = np.eye(3) * 0.01

        self.measure_var = np.eye(6)
        self.measure_var[0:3, 0:3] = self.var_gnss
        self.measure_var[3:, 3:] = self.var_v

        # Motion model noise Jacobian
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        # Measurement model Jacobian
        self.h_jac = np.zeros([6, 9])
        self.h_jac[:, :6] = np.eye(6)

        # Initialized
        self.n_gnss_taken = 0
        self.gnss_init_xyz = None
        self.initialized = False
        self.last_location = None

    def update(self, data):
        if self.last_location is None:
            dist = 0
        else:
            dist = np.linalg.norm(
                np.array(self.last_location[0:2]) - np.array(data['location'][0:2])
            )

        if self.last_location is None:
            self.last_location = data['location']

        if not self.is_initialized():
            self.initialize_with_true_data(data)

        if dist > 20:
            self.initialize_with_true_data(data)

        self.last_location = data['location']

        # Correction and Prediction
        try:
            self.predict_state_with_imu(data['imu']['values'])
        except IndexError:
            self.predict_state_with_imu(data['imu'])
        corrected = self.extract_states()

        self.correct_state_with_gnss(data)
        predicted = self.extract_states()

        # Stack the corrected and predicted updates
        updates = np.vstack(
            (
                corrected.flatten().astype(np.float32),
                predicted.flatten().astype(np.float32),
            )
        )

        return updates

