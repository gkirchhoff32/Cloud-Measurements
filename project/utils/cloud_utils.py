#
# Cloud Simulations
# cloud_utils.py
#
# Grant Kirchhoff
# Rev. 1: 03-06-2025 Creating functions to use in cloud simulations
#
# University of Colorado Boulder
#

import numpy as np
from load_ARSENL_data_utils import load_config
import os

cwd = os.getcwd()
config_path = cwd + r'\..\config'
config = load_config(config_path + r'\sim_cloud_DIAL_v2_config.yaml')

# Constants
c = config['constants']['c']  # [m/s] speed of light

def gen_cloud(alt, plat_height, cloud_bot, cloud_top, cloud_alpha, S, beta_aer, alpha_mol, beta_mol, N_L, A, eta, G,
              N_B, dt):
    """
    Generate cloud simulated backscatter rate, photons, and cloud shape
    :param alt: [m] Altitude grid [Nx1 array]
    :param plat_height: [m] Height of platform above sea-level [float]
    :param cloud_bot: [m] Location of cloud base [float]
    :param cloud_top: [m] Location of cloud top [float]
    :param cloud_alpha: Sharpness of air-to-cloud transition [float]
    :param S: [sr] Lidar ratio [float]
    :param beta_aer: [m-1 sr-1] Aerosol backscatter coefficient [float]
    :param alpha_mol: [m-1] Molecular extinction coefficient [float]
    :param beta_mol: [m-1 sr-1] Molecular backscatter coefficient [float]
    :param N_L: Number of transmitted photons per shot [float]
    :param A: [m^2] Telescope area [float]
    :param eta: Receiver efficiency [float]
    :param G: Geometric overlap function [Nx1 array float]
    :param N_B: Number of background photons per shot [float]
    :param dt: [s] Sampling resolution [float]

    """

    R = alt - plat_height  # [m] Remove platform height to convert to range

    dR = c * dt / 2  # [m]

    cloud_sig = (1 / (1 + np.exp(-cloud_alpha * (alt - cloud_bot)))) - (
                1 / (1 + np.exp(-cloud_alpha * (alt - cloud_top))))
    cloud = np.ones(len(alt))
    cloud *= cloud_sig

    beta_aer *= cloud  # [m-1 sr-1]
    alpha_aer = S * beta_aer

    trans = np.exp(-np.cumsum((alpha_mol + alpha_aer) * dR))
    N_com = N_L * (beta_mol + beta_aer) * dR * A / R ** 2 * trans ** 2 * eta * G + N_B
    N_com[0] = 0

    rho_func = N_com / dt  # [Hz]
    photon_rate = rho_func  # [Hz]

    return photon_rate, N_com, cloud