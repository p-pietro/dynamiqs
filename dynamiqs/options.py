from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter
from .qarrays.qarray import QArray
from .utils.global_settings import get_progress_meter

__all__ = ['DeviceBatching', 'Options']


@dataclass(frozen=True)
class DeviceBatching:
    """Configuration for distributing batch axes over multiple devices.

    Args:
        mesh_shape: Shape of the device mesh. If `None`, all available devices are
            used in a 1D mesh.
        batch_axes: Indices of the global batch axes to shard, ordered to match the
            mesh axes. If `None`, the leading batch axes are used.
    """

    mesh_shape: tuple[int, ...] | int | None = None
    batch_axes: tuple[int, ...] | int | None = None

    def __post_init__(self):
        mesh_shape = self.mesh_shape
        if isinstance(mesh_shape, int):
            mesh_shape = (mesh_shape,)
        batch_axes = self.batch_axes
        if isinstance(batch_axes, int):
            batch_axes = (batch_axes,)
        object.__setattr__(self, 'mesh_shape', mesh_shape)
        object.__setattr__(self, 'batch_axes', batch_axes)


class Options(eqx.Module):
    save_states: bool = True
    save_propagators: bool = True
    cartesian_batching: bool = True
    device_batching: DeviceBatching | bool | int | tuple[int, ...] | None = eqx.field(
        static=True, default=None
    )
    progress_meter: AbstractProgressMeter | bool | None = None
    t0: ScalarLike | None = None
    save_extra: Callable[[QArray], PyTree] | None = None
    nmaxclick: int = 10_000
    vectorized: bool = False
    assume_hermitian: bool = True

    def __init__(
        self,
        save_states: bool = True,
        save_propagators: bool = True,
        cartesian_batching: bool = True,
        device_batching: DeviceBatching | bool | int | tuple[int, ...] | None = None,
        progress_meter: AbstractProgressMeter | bool | None = None,
        t0: ScalarLike | None = None,
        save_extra: Callable[[QArray], PyTree] | None = None,
        nmaxclick: int = 10_000,
        vectorized: bool = False,
        assume_hermitian: bool = True,
    ):
        self.save_states = save_states
        self.save_propagators = save_propagators
        self.cartesian_batching = cartesian_batching
        self.device_batching = device_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.nmaxclick = nmaxclick
        self.vectorized = vectorized
        self.assume_hermitian = assume_hermitian

        # make `save_extra` a valid Pytree with `Partial`
        self.save_extra = jtu.Partial(save_extra) if save_extra is not None else None

    def __str__(self) -> str:
        return tree_str_inline(self)

    def initialise(self) -> Options:
        # We need to call this before entering JIT-compiled functions. Why? Because
        # `progress_meter` is defined at runtime by the default value set in the global
        # settings. Now things become a bit tricky:
        # - We can't get the default value to set it in the `__init__`, because the
        #   default argument to many functions is `Options()`, so it would be set
        #   forever to the default `progress_meter` value at the time of the function
        #   import.
        # - A simple workaround would be to use a property to get the `progress_meter`
        #   dynamically, but then changing the default value would not change the
        #   `options` object attributes, and we would cache hit the JIT-compiled
        #   function for any previous existing `options` object.
        return Options(
            save_states=self.save_states,
            save_propagators=self.save_propagators,
            cartesian_batching=self.cartesian_batching,
            device_batching=self.device_batching,
            progress_meter=get_progress_meter(self.progress_meter),
            t0=self.t0,
            save_extra=self.save_extra,
            nmaxclick=self.nmaxclick,
            vectorized=self.vectorized,
            assume_hermitian=self.assume_hermitian,
        )


def check_options(options: Options, solver_name: str):
    supported_options = {
        'sesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'mesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            'progress_meter',
            't0',
            'save_extra',
            'vectorized',
            'assume_hermitian',
        ),
        'sepropagator': (
            'save_propagators',
            'device_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'mepropagator': (
            'save_propagators',
            'cartesian_batching',
            'device_batching',
            't0',
            'save_extra',
        ),
        'floquet': ('device_batching', 'progress_meter', 't0'),
        'jssesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            't0',
            'save_extra',
            'nmaxclick',
        ),
        'dssesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            'save_extra',
        ),
        'jsmesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            'save_extra',
            'nmaxclick',
        ),
        'dsmesolve': (
            'save_states',
            'cartesian_batching',
            'device_batching',
            'save_extra',
        ),
    }
    valid_options = supported_options[solver_name]

    # check that all attributes are set to their default values except for the ones
    # specified in `valid_options`
    for key, value in options.__dict__.items():
        if key not in valid_options and value != getattr(Options(), key):
            valid_options_str = ', '.join(f'`{x}`' for x in valid_options)
            raise ValueError(
                f'Option `{key}` was set to `{value}` but is not used by '
                f'the solver `dq.{solver_name}()` (valid options: '
                f'{valid_options_str}).'
            )
