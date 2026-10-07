"""Dependency-free planar two-link forward-dynamics oracle.

This is a diagnostic reference for the contact-free Stage-7 articulation.  It
uses the classical manipulator equation ``M(q) qdd + C(q, qdot) + G(q) = tau``
and semi-implicit Euler integration.  It does not alter the CUDA solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .contracts.coupling import SPEC


@dataclass(frozen=True)
class PlanarTwoLink:
    first_mass_kg: float
    second_mass_kg: float
    first_length_m: float
    first_com_m: float
    second_com_m: float
    first_inertia_kg_m2: float
    second_inertia_kg_m2: float
    gravity_mps2: float = 9.81

    @classmethod
    def stage7(cls) -> "PlanarTwoLink":
        return cls(
            first_mass_kg=SPEC.body_masses_kg[2],
            second_mass_kg=SPEC.body_masses_kg[3],
            first_length_m=2.0 * SPEC.body_half_extents_m[2][0],
            first_com_m=SPEC.body_half_extents_m[2][0],
            second_com_m=SPEC.body_half_extents_m[3][0],
            first_inertia_kg_m2=SPEC.body_inertia_diagonal_kg_m2[2][2],
            second_inertia_kg_m2=SPEC.body_inertia_diagonal_kg_m2[3][2],
        )

    def mass_matrix(self, q: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(q) != 2 or not all(math.isfinite(value) for value in q):
            raise ValueError("q must contain two finite coordinates")
        m1, m2 = self.first_mass_kg, self.second_mass_kg
        length, r1, r2 = self.first_length_m, self.first_com_m, self.second_com_m
        coupling = length * r2 * math.cos(q[1])
        m11 = (
            self.first_inertia_kg_m2
            + self.second_inertia_kg_m2
            + m1 * r1 * r1
            + m2 * (length * length + r2 * r2 + 2.0 * coupling)
        )
        m12 = self.second_inertia_kg_m2 + m2 * (r2 * r2 + coupling)
        m22 = self.second_inertia_kg_m2 + m2 * r2 * r2
        return ((m11, m12), (m12, m22))

    def bias(self, q: Sequence[float], qdot: Sequence[float]) -> tuple[float, float]:
        if len(qdot) != 2 or not all(math.isfinite(value) for value in qdot):
            raise ValueError("qdot must contain two finite velocities")
        m1, m2 = self.first_mass_kg, self.second_mass_kg
        length, r1, r2 = self.first_length_m, self.first_com_m, self.second_com_m
        q1, q2 = q
        qd1, qd2 = qdot
        h = m2 * length * r2 * math.sin(q2)
        coriolis = (-h * (2.0 * qd1 * qd2 + qd2 * qd2), h * qd1 * qd1)
        gravity = (
            self.gravity_mps2
            * (
                m1 * r1 * math.cos(q1)
                + m2 * (length * math.cos(q1) + r2 * math.cos(q1 + q2))
            ),
            self.gravity_mps2 * m2 * r2 * math.cos(q1 + q2),
        )
        return (coriolis[0] + gravity[0], coriolis[1] + gravity[1])

    def acceleration(
        self,
        q: Sequence[float],
        qdot: Sequence[float],
        torque_nm: Sequence[float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        if len(torque_nm) != 2 or not all(math.isfinite(value) for value in torque_nm):
            raise ValueError("torque must contain two finite values")
        matrix = self.mass_matrix(q)
        bias = self.bias(q, qdot)
        rhs0, rhs1 = torque_nm[0] - bias[0], torque_nm[1] - bias[1]
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not math.isfinite(determinant) or determinant <= 1.0e-12:
            raise ValueError("two-link mass matrix is singular")
        return (
            (rhs0 * matrix[1][1] - matrix[0][1] * rhs1) / determinant,
            (matrix[0][0] * rhs1 - matrix[1][0] * rhs0) / determinant,
        )

    def step(
        self,
        q: Sequence[float],
        qdot: Sequence[float],
        *,
        dt: float = 1.0 / 120.0,
        substeps: int = 2,
        torque_nm: Sequence[float] = (0.0, 0.0),
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if not math.isfinite(dt) or dt <= 0.0 or substeps <= 0:
            raise ValueError("dt and substeps must be positive")
        coordinate = [float(q[0]), float(q[1])]
        velocity = [float(qdot[0]), float(qdot[1])]
        h = dt / substeps
        for _ in range(substeps):
            acceleration = self.acceleration(coordinate, velocity, torque_nm)
            velocity = [velocity[index] + acceleration[index] * h for index in range(2)]
            coordinate = [coordinate[index] + velocity[index] * h for index in range(2)]
        return (tuple(coordinate), tuple(velocity))

    @staticmethod
    def pd_torque(
        q: Sequence[float],
        qdot: Sequence[float],
        target: Sequence[float],
        stiffness_nm_per_rad: Sequence[float],
        damping_nms_per_rad: Sequence[float],
        effort_limit_nm: Sequence[float],
    ) -> tuple[float, float]:
        """Evaluate the two scalar, effort-limited position drives.

        The result is the explicit torque frame used by the Stage-7 CUDA
        articulation: ``kp * (target - q) - kd * qdot``, clamped per joint.
        Keeping this calculation outside the dynamics step makes the drive
        convention independently testable.
        """

        vectors = (
            q,
            qdot,
            target,
            stiffness_nm_per_rad,
            damping_nms_per_rad,
            effort_limit_nm,
        )
        if any(len(values) != 2 for values in vectors):
            raise ValueError("PD drive vectors must each contain two values")
        if not all(math.isfinite(value) for values in vectors for value in values):
            raise ValueError("PD drive vectors must be finite")
        if any(value < 0.0 for value in stiffness_nm_per_rad):
            raise ValueError("PD stiffness cannot be negative")
        if any(value < 0.0 for value in damping_nms_per_rad):
            raise ValueError("PD damping cannot be negative")
        if any(value < 0.0 for value in effort_limit_nm):
            raise ValueError("PD effort limits cannot be negative")
        torque = []
        for index in range(2):
            requested = (
                stiffness_nm_per_rad[index] * (target[index] - q[index])
                - damping_nms_per_rad[index] * qdot[index]
            )
            limit = effort_limit_nm[index]
            torque.append(max(-limit, min(limit, requested)))
        return (torque[0], torque[1])

    def step_pd(
        self,
        q: Sequence[float],
        qdot: Sequence[float],
        target: Sequence[float],
        *,
        stiffness_nm_per_rad: Sequence[float],
        damping_nms_per_rad: Sequence[float],
        effort_limit_nm: Sequence[float],
        dt: float = 1.0 / 120.0,
        substeps: int = 2,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Advance a contact-free two-link arm with explicit PD drives.

        The target and gains are held for the control step, while torque is
        recomputed from the current coordinate and velocity at each internal
        substep. Integration remains semi-implicit Euler, matching ``step``.
        """

        if len(q) != 2 or len(qdot) != 2:
            raise ValueError("q and qdot must each contain two values")
        if not math.isfinite(dt) or dt <= 0.0 or substeps <= 0:
            raise ValueError("dt and substeps must be positive")
        coordinate = [float(q[0]), float(q[1])]
        velocity = [float(qdot[0]), float(qdot[1])]
        h = dt / substeps
        for _ in range(substeps):
            torque = self.pd_torque(
                coordinate,
                velocity,
                target,
                stiffness_nm_per_rad,
                damping_nms_per_rad,
                effort_limit_nm,
            )
            acceleration = self.acceleration(coordinate, velocity, torque)
            velocity = [velocity[index] + acceleration[index] * h for index in range(2)]
            coordinate = [coordinate[index] + velocity[index] * h for index in range(2)]
        return (tuple(coordinate), tuple(velocity))
