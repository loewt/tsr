# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
from gafro import Motor
from numpy import pi

EPSILON = 0.001


def wrap_to_interval(angles: np.ndarray, lower: np.ndarray = None) -> np.ndarray:
    """
    Wrap a vector of angles to a continuous interval starting at `lower`.

    Args:
        angles: (N,) array of angles (in radians)
        lower: (N,) array of lower bounds; defaults to -pi if None

    Returns:
        wrapped: (N,) array of wrapped angles
    """
    if lower is None:
        lower = -pi * np.ones_like(angles)
    return (angles - lower) % (2 * pi) + lower


def rotation_angle(R: np.ndarray) -> float:
    """
    Compute the angle of rotation for a rotation matrix.

    Computed from the bivector part of the corresponding gafro motor's
    logarithm, whose norm is the rotation angle. This matches the classical
    θ = arccos((trace(R) - 1) / 2) to machine precision, but stays
    well-conditioned near θ = 0 and θ = π where the cosine is flat.

    Args:
        R: 3x3 rotation matrix

    Returns:
        angle: rotation angle in radians [0, π]
    """
    T = np.eye(4)
    T[0:3, 0:3] = R
    # The bivector part of the motor log is the rotation vector; its norm is
    # the angle.
    return float(np.linalg.norm(Motor.from_matrix(T).log()[0:3]))


def geodesic_error(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic error between two transforms on SE(3).

    The error is computed as:
    - Translation error: the Euclidean distance between positions
    - Rotation error: the angle of the relative motor (t1)^-1 * t2

    Args:
        t1: first transform (4x4)
        t2: second transform (4x4)

    Returns:
        error: 4-vector [dx, dy, dz, rotation_angle]
               where dx, dy, dz are in meters and rotation_angle is in radians
    """
    # Translation error (in world frame)
    trans_error = t2[0:3, 3] - t1[0:3, 3]

    # Rotation error: angle of the relative motor (t1)^-1 * t2
    rel = Motor.from_matrix(t1).inverse() * Motor.from_matrix(t2)
    angle_error = float(np.linalg.norm(rel.log()[0:3]))

    return np.hstack((trans_error, angle_error))


def geodesic_distance(t1: np.ndarray, t2: np.ndarray, r: float = 1.0) -> float:
    """
    Compute the geodesic distance between two transforms on SE(3).

    The distance combines translation (meters) and rotation (radians) errors.
    The parameter `r` allows weighting rotation relative to translation.

    As noted in Section 4.2 of Berenson et al. 2011:
    "we implicitly weigh rotation in radians and translation in meters equally
    when computing ||Δx||, but the two types of units can be weighed in an
    arbitrary manner"

    Args:
        t1: first transform (4x4)
        t2: second transform (4x4)
        r: weight for rotation in units of meters/radian (default 1.0)
           Higher values penalize rotation errors more.

    Returns:
        distance: weighted geodesic distance
    """
    error = geodesic_error(t1, t2)
    # Weight the rotation component
    error[3] = r * error[3]
    return np.linalg.norm(error)
