#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# tensors.py
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
from numpy import cos, sin


def get_rotation_matrix(a, b, g):
    """Returns a (extrinsic) rotation matrix corresponding to the given angles

    Parameters
    ----------

    a: float
        Rotation about the x axis
    b: float
        Rotation about the y axis
    g: float
        Rotation about the z axis

    Returns
    -------

    R: 3x3 np.ndarray
        Rotation matrix
    """

    R = np.array([
        [cos(g), -sin(g), 0],
        [sin(g), cos(g), 0],
        [0, 0, 1],
    ]) @ np.array([
        [cos(b), 0, sin(b)],
        [0, 1, 0],
        [-sin(b), 0, cos(b)],
    ]) @ np.array([
        [1, 0, 0],
        [0, cos(a), -sin(a)],
        [0, sin(a), cos(a)],
    ])
    return R


def rotate_tensor4(c, R=None, angles=None):
    """
    Rotate a 4th-order tensor using Euler angles

    Parameters
    ----------

    c: 3x3x3x3 np.ndarray
        Tensor to be rotated
    R: 3x3 np.ndarray, optional
        Rotation matrix
    angles: 3-tuple
        Angles to generate a rotation matrix (extrinsic Euler)

    Returns
    -------

    rc: 3x3x3x3 np.ndarray
        Rotated 4th order tensor
    """

    if R is None:
        R = get_rotation_matrix(*angles)

    RR = np.outer(R, R)
    RRRR = np.outer(RR, RR).reshape(4 * R.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    rc = np.tensordot(RRRR, c, axes)

    return rc
