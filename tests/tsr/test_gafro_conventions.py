#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
Tests pinning the gafro conventions that the TSR transform math relies on.

tsr composes poses with gafro motors instead of 4x4 matrix products. These
tests assert the assumptions that makes: that motor multiplication follows the
same left-to-right order as matrix multiplication, that Motor.inverse() agrees
with a matrix inverse, and that the motor logarithm's bivector part carries the
rotation angle. A gafro upgrade that changed any of these would otherwise break
TSR geometry silently rather than loudly.
"""

import unittest

import numpy as np
from gafro import Motor

from tsr.tsr import TSR


def random_transforms(n, seed=0):
    """Generate n random rigid transforms via xyzrpy."""
    rng = np.random.default_rng(seed)
    return [TSR.xyzrpy_to_trans(rng.uniform(-np.pi, np.pi, 6)) for _ in range(n)]


def as_matrix(motor):
    return np.asarray(motor.to_transformation_matrix())


class TestMotorConversion(unittest.TestCase):
    """Round-tripping a transform through a Motor must be lossless."""

    def test_roundtrip_identity(self):
        np.testing.assert_array_almost_equal(as_matrix(Motor.from_matrix(np.eye(4))), np.eye(4))

    def test_roundtrip_random(self):
        for T in random_transforms(50, seed=1):
            np.testing.assert_array_almost_equal(as_matrix(Motor.from_matrix(T)), T, decimal=12)

    def test_roundtrip_preserves_orthonormality(self):
        for T in random_transforms(20, seed=2):
            R = as_matrix(Motor.from_matrix(T))[0:3, 0:3]
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)


class TestMotorProduct(unittest.TestCase):
    """Motor multiplication must match left-to-right matrix multiplication."""

    def test_matches_matrix_product(self):
        A, B, C = random_transforms(3, seed=4)
        MA, MB, MC = (Motor.from_matrix(T) for T in (A, B, C))
        np.testing.assert_array_almost_equal(as_matrix(MA * MB), A @ B, decimal=12)
        np.testing.assert_array_almost_equal(as_matrix(MA * MB * MC), A @ B @ C, decimal=12)

    def test_order_is_not_commutative(self):
        """Guards against a silently reversed multiplication convention."""
        A, B = random_transforms(2, seed=5)
        MA, MB = Motor.from_matrix(A), Motor.from_matrix(B)
        self.assertFalse(np.allclose(as_matrix(MA * MB), as_matrix(MB * MA)))

    def test_identity_is_neutral(self):
        A = random_transforms(1, seed=6)[0]
        MI, MA = Motor.from_matrix(np.eye(4)), Motor.from_matrix(A)
        np.testing.assert_array_almost_equal(as_matrix(MI * MA * MI), A, decimal=12)

    def test_long_chain_stays_rigid(self):
        """Accumulated motor products must not drift off SE(3)."""
        motor = Motor.from_matrix(np.eye(4))
        for T in random_transforms(200, seed=14):
            motor = motor * Motor.from_matrix(T)
        R = as_matrix(motor)[0:3, 0:3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)


class TestMotorInverse(unittest.TestCase):
    """Motor.inverse() must match numpy.linalg.inv for rigid transforms."""

    def test_matches_numpy(self):
        for T in random_transforms(50, seed=7):
            np.testing.assert_array_almost_equal(
                as_matrix(Motor.from_matrix(T).inverse()), np.linalg.inv(T), decimal=12
            )

    def test_cancels_to_identity(self):
        for T in random_transforms(20, seed=8):
            M = Motor.from_matrix(T)
            np.testing.assert_array_almost_equal(as_matrix(M * M.inverse()), np.eye(4), decimal=12)
            np.testing.assert_array_almost_equal(as_matrix(M.inverse() * M), np.eye(4), decimal=12)


class TestMotorLogAngle(unittest.TestCase):
    """The motor log's bivector norm must be the rotation angle."""

    @staticmethod
    def log_angle(t1, t2):
        rel = Motor.from_matrix(t1).inverse() * Motor.from_matrix(t2)
        return float(np.linalg.norm(np.asarray(rel.log())[0:3]))

    @staticmethod
    def trace_angle(t1, t2):
        rel = t1[0:3, 0:3].T @ t2[0:3, 0:3]
        return float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))

    def test_matches_trace_formula(self):
        """The log angle must agree with arccos((trace - 1) / 2).

        Pins the scale convention: a half-angle (quaternion-style) log would
        disagree by a factor of two on every sample.
        """
        transforms = random_transforms(60, seed=11)
        for t1, t2 in zip(transforms[::2], transforms[1::2]):
            self.assertAlmostEqual(self.log_angle(t1, t2), self.trace_angle(t1, t2), places=10)

    def test_zero_for_identical(self):
        """Exactly zero, where arccos of the trace would yield ~1e-8."""
        for T in random_transforms(20, seed=12):
            self.assertAlmostEqual(self.log_angle(T, T), 0.0, places=12)

    def test_known_angles_about_z(self):
        for angle in (0.0, 0.1, np.pi / 4, np.pi / 2, 2.0, np.pi - 1e-6):
            c, s = np.cos(angle), np.sin(angle)
            T = np.eye(4)
            T[0:3, 0:3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            self.assertAlmostEqual(self.log_angle(np.eye(4), T), angle, places=10)

    def test_is_translation_invariant(self):
        """Rotation angle must ignore the translation part."""
        A, B = random_transforms(2, seed=13)
        A_shifted, B_shifted = A.copy(), B.copy()
        A_shifted[0:3, 3] += [1.0, -2.0, 3.0]
        B_shifted[0:3, 3] += [-4.0, 5.0, 0.5]
        self.assertAlmostEqual(self.log_angle(A, B), self.log_angle(A_shifted, B_shifted), places=10)


if __name__ == "__main__":
    unittest.main()
