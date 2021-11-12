#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# waves.py
#
# This file is part of the anisutils package
#
# Copyright © 2021 Mathieu Gaborit (matael) <mathieu@matael.org>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Mathieu (matael) Gaborit wrote this file. As long as you retain this notice
# you can do whatever you want with this stuff. If we meet some day, and you
# think this stuff is worth it, you can buy me a beer or coffee in return
#

import numpy as np

from .tensors import rotate_tensor4
from .hookes import Hij_to_cijkl


def get_n(gamma):
    """Returns a direction vector in the x-z plane
    forming an angle gamma with the z-axis

    Parameters
    ----------

    gamma: float
        Angle between the vector and the z-axis

    Returns
    -------

    n: 1x3 np.ndarray
        Resulting direction vector
    """

    return np.array([np.sin(gamma), 0, np.cos(gamma)]).T


def get_christoffel(c, n):
    """Computes Christoffel tensor associated with the
    rigidity tensor c and propagation direction n.

    Parameters
    ----------

    c: 3x3x3x3 np.ndarray
        Rigidity tensor
    n: 1x3 np.ndarray
        Propagation direction vector

    Returns
    -------

    Gamma: 3x3 np.ndarray
        Christoffel tensor
    """

    Gamma = np.einsum('ijkl,j,k', c, n, n)
    return Gamma


def get_waves(H, rho, alpha_v, gamma_v, tolerance=.05):
    """Compute the waves celerities and polarizations for the given material.

    Parameters
    ----------

    H: 6x6 np.ndarray
        Hooke's matrix of the media
    rho: float
        Density of the media
    alpha_v: iterable
        vector of N angles about z for the propagation vector (in rad)
    gamma_v: iterable
        vector of M angles in the x-z plane for the propagation vector (in rad)
        0 correspond to a vector colinear to the z axis.


    Returns
    -------

    celerities: dict of NxM array
        Dictionary holding the celerities associated with the different waves and each set
        of angles for the propagation vector
        Keys are 'L', 'S1' and 'S2' for quasi-longitudinal, and quasi-shear (1 and 2) modes respectively
    polarizations: dict of NxMx3
        Same as celerities but with polarizations vectors
    """

    modes = ['L', 'S1', 'S2']
    celerities = {_: np.empty((len(alpha_v), len(gamma_v),)) for _ in modes}
    polarizations = {_: np.empty((len(alpha_v), len(gamma_v), 3)) for _ in modes}

    for i_alpha, alpha in enumerate(alpha_v):
        c = Hij_to_cijkl(H)
        comp_c = rotate_tensor4(c, R=None, angles=(0,0,alpha))

        for i_gamma, gamma in enumerate(gamma_v):
            n = get_n(gamma)

            # compute the speeds & polarizations
            Gamma_christ = get_christoffel(comp_c, n)
            D, V = np.linalg.eigh(Gamma_christ)

            if i_gamma == 0 and i_alpha == 0:
                prev_L = n.T
                prev_S1 = np.array([0, 1, 0])
            elif i_gamma == 0:
                prev_L = polarizations['L'][i_alpha-1,i_gamma].T
                prev_S1 = polarizations['S1'][i_alpha-1,i_gamma].T
            else:
                prev_L = polarizations['L'][i_alpha,i_gamma-1].T
                prev_S1 = polarizations['S1'][i_alpha,i_gamma-1].T

            dot_products = np.abs(prev_L @ V)
            id_L = np.where(dot_products == dot_products.max())[0][0]

            dot_products = np.abs(prev_S1 @ V)
            id_S1 = np.where(dot_products == dot_products.max())[0][0]
            assert id_L != id_S1

            id_S2 = [i for i in range(3) if i not in [id_L, id_S1]][0]

            celerities['L'][i_alpha,i_gamma] = np.sqrt(D[id_L]/rho)
            celerities['S1'][i_alpha,i_gamma] = np.sqrt(D[id_S1]/rho)
            celerities['S2'][i_alpha,i_gamma] = np.sqrt(D[id_S2]/rho)
            polarizations['L'][i_alpha,i_gamma] = V[:,id_L]
            polarizations['S1'][i_alpha,i_gamma] = V[:,id_S1]
            polarizations['S2'][i_alpha,i_gamma] = V[:,id_S2]

            if i_alpha != 0:
                s1 = 1/celerities['S1'][i_alpha,i_gamma]
                s1_prev = 1/celerities['S1'][i_alpha-1,i_gamma]
                s2_prev = 1/celerities['S2'][i_alpha-1,i_gamma]

                s1ms2prev = np.abs(s1 - s2_prev)
                s1ms1prev = np.abs(s1 - s1_prev)

                # if  s1ms2prev < s1ms1prev:
                if  s1ms2prev - s1ms1prev < tolerance*(s1ms1prev):
                    c1 = celerities['S1'][i_alpha,i_gamma]
                    p1 = polarizations['S1'][i_alpha,i_gamma]
                    celerities['S1'][i_alpha,i_gamma] =  celerities['S2'][i_alpha,i_gamma]
                    celerities['S2'][i_alpha,i_gamma] = c1
                    polarizations['S1'][i_alpha,i_gamma] =  polarizations['S2'][i_alpha,i_gamma]
                    polarizations['S2'][i_alpha,i_gamma] = p1

    return celerities, polarizations
