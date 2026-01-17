from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import jax
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .qarrays.dense_qarray import DenseQArray
from .qarrays.qarray import QArray
from .qarrays.sparsedia_qarray import SparseDIAQArray
from .time_qarray import (
    ConstantTimeQArray,
    ModulatedTimeQArray,
    PWCTimeQArray,
    SummedTimeQArray,
    TimeQArray,
)

__all__ = ['make_mesh', 'DataParallel']


def make_mesh(
    devices: Sequence[jax.Device] | None = None,
    *,
    axis_name: str | Sequence[str] = 'd',
    axis_names: Sequence[str] | None = None,
    mesh_shape: Sequence[int] | None = None,
) -> Mesh:
    """Create a device mesh for data-parallel sharding.

    Args:
        devices: Devices to include in the mesh. Defaults to `jax.devices()`.
        axis_name: Mesh axis name used in sharding specs (alias for `axis_names`).
        axis_names: Mesh axis names used in sharding specs.
        mesh_shape: Shape of the mesh. Must match `axis_names` length and device
            count. Defaults to a 1D mesh when a single axis name is provided.

    Returns:
        JAX mesh with a single axis named `axis_name`.
    """
    if devices is None:
        devices = jax.devices()
    if axis_names is None:
        axis_names = (axis_name,) if isinstance(axis_name, str) else tuple(axis_name)
    axis_names = tuple(axis_names)

    if mesh_shape is None:
        if len(axis_names) != 1:
            raise ValueError(
                'Argument `mesh_shape` must be provided when `axis_names` has more '
                'than one axis.'
            )
        mesh_shape = (len(devices),)
    mesh_shape = tuple(mesh_shape)

    if len(mesh_shape) != len(axis_names):
        raise ValueError(
            'Argument `mesh_shape` must have the same length as `axis_names`, but '
            f'got mesh_shape={mesh_shape} and axis_names={axis_names}.'
        )
    if int(np.prod(mesh_shape)) != len(devices):
        raise ValueError(
            'Argument `mesh_shape` must match the number of devices, but got '
            f'mesh_shape={mesh_shape} and {len(devices)} devices.'
        )

    mesh_devices = np.array(devices).reshape(*mesh_shape)
    return Mesh(mesh_devices, axis_names)


@dataclass(frozen=True)
class DataParallel:
    """Data-parallel sharding policy for batched simulations.

    Attributes:
        mesh: Device mesh used for sharding.
        axis_name: Axis name(s) used in sharding specs.
        batch_axis: Axis index(es) representing independent simulations.

    Note:
        The sharded batch axes should be divisible by the number of devices. The
        `batch_axis` indices refer to the leading batch axes of operator arrays.
    """

    mesh: Mesh
    axis_name: str | Sequence[str] = 'd'
    batch_axis: int | Sequence[int] = 0

    _axis_names: tuple[str, ...] = field(init=False, repr=False, default=())
    _batch_axes: tuple[int, ...] = field(init=False, repr=False, default=())

    def __post_init__(self):
        axis_names = (
            (self.axis_name,)
            if isinstance(self.axis_name, str)
            else tuple(self.axis_name)
        )
        batch_axes = (
            (self.batch_axis,)
            if isinstance(self.batch_axis, int)
            else tuple(self.batch_axis)
        )
        if len(axis_names) != len(batch_axes):
            raise ValueError(
                'Arguments `axis_name(s)` and `batch_axis` must have the same length,'
                f' but got axis_name(s)={axis_names} and batch_axis={batch_axes}.'
            )
        if len(set(batch_axes)) != len(batch_axes):
            raise ValueError(
                f'Argument `batch_axis` must not contain duplicates, got {batch_axes}.'
            )

        mesh_axis_names = tuple(self.mesh.axis_names)
        missing = [name for name in axis_names if name not in mesh_axis_names]
        if missing:
            raise ValueError(
                'Argument `axis_name(s)` must be present in the mesh axis names, but '
                f'missing {missing} from mesh.axis_names={mesh_axis_names}.'
            )

        object.__setattr__(self, '_axis_names', axis_names)
        object.__setattr__(self, '_batch_axes', batch_axes)

    @property
    def axis_names(self) -> tuple[str, ...]:
        return self._axis_names

    @property
    def batch_axes(self) -> tuple[int, ...]:
        return self._batch_axes

    def _replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    def _shard_axes(self, rank: int, *, batch_ndim: int) -> NamedSharding:
        ps = [None] * rank
        for axis_name, batch_axis in zip(self.axis_names, self.batch_axes, strict=True):
            if batch_axis < batch_ndim:
                ps[batch_axis] = axis_name
        return NamedSharding(self.mesh, P(*ps))

    def put_array(self, a: Array) -> Array:
        """Replicate an array across the mesh."""
        return jax.device_put(a, self._replicated())

    def put_operator_matrix(self, a: Array) -> Array:
        """Shard batched operator matrices along `batch_axis`."""
        if a.ndim <= 2:
            return jax.device_put(a, self._replicated())
        return jax.device_put(a, self._shard_axes(a.ndim, batch_ndim=a.ndim - 2))

    def put_scalar_batch(self, a: Array) -> Array:
        """Shard scalar batches along `batch_axis`."""
        if a.ndim <= 1:
            return jax.device_put(a, self._replicated())
        return jax.device_put(a, self._shard_axes(a.ndim, batch_ndim=a.ndim - 1))

    def put_qarray(self, q: QArray) -> QArray:
        """Shard a qarray according to the data-parallel policy."""
        if isinstance(q, DenseQArray):
            data = self.put_operator_matrix(q.data)
            return replace(q, data=data)
        if isinstance(q, SparseDIAQArray):
            diags = self.put_operator_matrix(q.diags)
            return replace(q, diags=diags)
        return q

    def put_timeqarray(self, tq: TimeQArray) -> TimeQArray:
        """Shard a timeqarray according to the data-parallel policy."""
        if isinstance(tq, ConstantTimeQArray):
            return replace(tq, qarray=self.put_qarray(tq.qarray))

        if isinstance(tq, PWCTimeQArray):
            times = self.put_array(tq.times)
            values = self.put_scalar_batch(tq.values)
            qarray = self.put_qarray(tq.qarray)
            return replace(tq, times=times, values=values, qarray=qarray)

        if isinstance(tq, SummedTimeQArray):
            timeqarrays = [self.put_timeqarray(x) for x in tq.timeqarrays]
            return replace(tq, timeqarrays=timeqarrays)

        if isinstance(tq, ModulatedTimeQArray):
            return replace(tq, qarray=self.put_qarray(tq.qarray))

        return tq

    def log_under_parallelization(self, batched_axes: int, *, context: str) -> None:
        if len(self.batch_axes) < batched_axes:
            logger = logging.getLogger(__name__)
            logger.info(
                '%s: parallelizing over %d axis/axes but the problem has %d batched '
                'axis/axes; remaining batch axes are replicated.',
                context,
                len(self.batch_axes),
                batched_axes,
            )
