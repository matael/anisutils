#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# graphics.py
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


def get_surface(a_v, g_v, data):
    """Returns the matrices to plot a 3D celerity/slowness surface.

    Parameters
    ----------

    a_v: N-element vector
        Angles about the z axis (in rad)
    g_v: M-element vector
        Angles between the z axis and the propagation direction (in rad)
    data: NxM np.ndarray
        Data to be represented as a distance to the origin

    Returns
    -------

    X, Y, Z: NxM np.ndarray
        3D cartesian coordinates of the surface

    """
    assert len(a_v) == data.shape[0]
    assert len(g_v) == data.shape[1]

    A_v, G_v = np.meshgrid(a_v, g_v, indexing='ij')

    X = np.sin(G_v)*np.cos(A_v)*data
    Y = np.sin(G_v)*np.sin(A_v)*data
    Z = np.cos(G_v)
    return X, Y, Z
