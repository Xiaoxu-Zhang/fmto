import numpy as np
from dataclasses import dataclass
from numpy import ndarray
from pyfmto.problem import MultiTaskProblem as Mtp, SingleTaskProblem as Stp
from typing import Tuple

__all__ = ["Gecco2020"]


def _forward_kinematics_2d_hetero(angles: ndarray, lengths: ndarray) -> ndarray:
    """
    Simple planar chain forward kinematics with revolute joints.

    Angles / lengths are 1D arrays of the same length; returns end-effector (x, y).
    """
    angles = np.asarray(angles, dtype=float).ravel()
    lengths = np.asarray(lengths, dtype=float).ravel()
    if angles.size != lengths.size:
        raise ValueError("angle/length size mismatch in forward_kinematics_2d_hetero")

    x = 0.0
    y = 0.0
    theta_sum = 0.0
    for theta, L in zip(angles, lengths):
        theta_sum += float(theta)
        x += float(L) * np.cos(theta_sum)
        y += float(L) * np.sin(theta_sum)
    return np.array([x, y], dtype=float)


@dataclass(frozen=True)
class HeteroV2Params:
    lengths: np.ndarray  # (task_num, D)
    angular_factor: np.ndarray  # (task_num,)
    noise_sigma: np.ndarray  # (task_num,)
    targets: Tuple[np.ndarray, ...]  # tuple of (nT,2) arrays per task


_HETERO_V2_CACHE: dict[tuple[int, int, int], HeteroV2Params] = {}


def _get_hetero_v2_params(dim: int, task_num: int, seed: int = 123) -> HeteroV2Params:
    """
    Generate (and cache) task parameters for KinematicArm2020HeteroV2.

    The design follows the reference implementation used in IAFFBO conversion:
    - 4 groups of tasks with different active joint subspaces and link patterns
    - separated target regions (front-far, back-deep, side-high)
    - heterogeneous angular ranges and noise levels
    """
    key = (int(dim), int(task_num), int(seed))
    cached = _HETERO_V2_CACHE.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(seed)

    half = dim // 2
    base_front_subspace = np.concatenate([3.0 * np.ones(half), 0.2 * np.ones(dim - half)])
    base_back_subspace = np.concatenate([0.2 * np.ones(dim - half), 3.0 * np.ones(half)])
    base_alternating = np.array([3.0 if (i % 2 == 1) else 0.3 for i in range(dim)])
    base_uniform = np.ones(dim)

    def sample_front_far(m: int) -> np.ndarray:
        xs = rng.random(m) * (1.8 - 1.1) + 1.1
        ys = rng.random(m) * (0.4 + 0.4) - 0.4
        return np.stack([xs, ys], axis=1)

    def sample_back_deep(m: int) -> np.ndarray:
        xs = rng.random(m) * (-0.6 + 1.6) - 1.6
        ys = rng.random(m) * (-0.2 + 1.0) - 1.0
        return np.stack([xs, ys], axis=1)

    def sample_side_high(m: int) -> np.ndarray:
        xs = rng.random(m) * (0.4 + 0.4) - 0.4
        ys = rng.random(m) * (1.5 - 0.8) + 0.8
        return np.stack([xs, ys], axis=1)

    n_groups = 4
    base_per_group = task_num // n_groups
    remainder = task_num % n_groups
    group_sizes = np.full(n_groups, base_per_group, dtype=int)
    group_sizes[:remainder] += 1

    labels = np.concatenate([gid * np.ones(gs, dtype=int) for gid, gs in enumerate(group_sizes)])
    rng.shuffle(labels)

    lengths = np.zeros((task_num, dim), dtype=float)
    ang_factor = np.zeros(task_num, dtype=float)
    noise_sigma = np.zeros(task_num, dtype=float)
    targets_list: list[np.ndarray] = []

    for k in range(task_num):
        gid = int(labels[k])
        if gid == 0:
            base_pattern = base_front_subspace
        elif gid == 1:
            base_pattern = base_back_subspace
        elif gid == 2:
            base_pattern = base_alternating
        else:
            base_pattern = base_uniform

        # Log-normal multiplicative noise: exp(0 + 0.35 * N(0,1))
        noise = np.exp(0.35 * rng.standard_normal(dim))
        pattern = base_pattern * noise
        pattern = np.maximum(pattern, 0.05)
        pattern = pattern / pattern.sum()

        total_length = 0.6 + (1.6 - 0.6) * rng.random()
        lengths[k, :] = total_length * pattern

        if gid == 0:
            ang_factor[k] = 0.4 + (0.9 - 0.4) * rng.random()
        elif gid == 1:
            ang_factor[k] = 0.9 + (1.5 - 0.9) * rng.random()
        elif gid == 2:
            ang_factor[k] = 0.3 + (1.3 - 0.3) * rng.random()
        else:
            ang_factor[k] = 0.5 + (1.6 - 0.5) * rng.random()

        if gid == 0:
            targets = sample_front_far(3)
        elif gid == 1:
            targets = sample_back_deep(3)
        elif gid == 2:
            targets = np.vstack([sample_front_far(1), sample_side_high(1), sample_back_deep(1)])
        else:
            targets = np.vstack([sample_front_far(2), sample_back_deep(2), sample_side_high(1)])
        targets_list.append(targets)

        if gid == 0:
            noise_sigma[k] = (0.003 - 0.0) * rng.random()
        elif gid == 1:
            noise_sigma[k] = 0.01 + (0.025 - 0.01) * rng.random()
        elif gid == 2:
            noise_sigma[k] = (0.015 - 0.0) * rng.random()
        else:
            noise_sigma[k] = 0.02 + (0.04 - 0.02) * rng.random()

    params = HeteroV2Params(
        lengths=lengths,
        angular_factor=ang_factor,
        noise_sigma=noise_sigma,
        targets=tuple(targets_list),
    )
    _HETERO_V2_CACHE[key] = params
    return params


class _KinematicArm2020HeteroV2Task(Stp):
    """Single planar kinematic arm task with heterogeneous link lengths and targets."""

    def __init__(
        self,
        lengths: ndarray,
        angular_factor: float,
        noise_sigma: float,
        targets: ndarray,
        dim: int,
        **kwargs,
    ):
        lb = np.zeros(dim)
        ub = np.ones(dim)
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

        self._lengths = np.asarray(lengths, dtype=float).reshape(-1)
        if self._lengths.size != dim:
            raise ValueError(f"lengths must have size {dim}, got {self._lengths.size}")
        self._angular_factor = float(angular_factor)
        self._noise_sigma = float(noise_sigma)

        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError(f"targets must have shape (nT, 2), got {targets.shape}")
        self._targets = targets

        self.set_x_global(None)

    @property
    def name(self) -> str:
        return f"KinematicArm2020HeteroV2Task"

    def _eval_single(self, x: ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        angles = (x - 0.5) * self._angular_factor * 2.0 * np.pi
        ef = _forward_kinematics_2d_hetero(angles, self._lengths)

        diffs = self._targets - ef  # (n_targets, 2)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        val = float(np.mean(dists))

        if self._noise_sigma > 0:
            val += self._noise_sigma * np.random.randn()
        return val


class Gecco2020(Mtp):
    """
    dim: 10  # [2, 50]
    task_num: 50  # [1, +inf)
    """
    is_realworld = True
    intro = """
        Kinematic Arm 2020 Hetero V2: Strongly heterogeneous planar kinematic arm benchmark (GECCO'20 inspired).
    """
    notes = """
        - Decision variables x are in [0, 1]^D and mapped to joint angles:
          angles = (x - 0.5) * angular_factor * 2*pi.
        - Each task has heterogeneous link lengths, angular range, target set, and noise level.
        - Objective is the average Euclidean distance from end-effector to all task targets,
          with additional Gaussian noise (task-specific sigma).
    """
    references = [
        """
        Mouret, J.-B., & Maguire, G. (2020).
        Quality Diversity for Multi-task Optimization. GECCO 2020.
        """
    ]

    def __init__(self, dim: int = 10, task_num: int = 50, **kwargs):
        if dim < 2:
            raise ValueError(f"KinematicArm2020HeteroV2 requires dim >= 2, got dim={dim}")
        if task_num < 1:
            raise ValueError(f"task_num must be >= 1, got {task_num}")
        super().__init__(dim, task_num, **kwargs)

    def _init_tasks(self, dim: int, task_num: int, **kwargs):
        params = _get_hetero_v2_params(dim, task_num, seed=self.seed)
        funcs: list[_KinematicArm2020HeteroV2Task] = []
        for idx in range(task_num):
            t = _KinematicArm2020HeteroV2Task(
                lengths=params.lengths[idx],
                angular_factor=float(params.angular_factor[idx]),
                noise_sigma=float(params.noise_sigma[idx]),
                targets=params.targets[idx],
                dim=dim,
                **kwargs,
            )
            funcs.append(t)
        return funcs

