from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache, wraps
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jax._src.lib import xla_client
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree

from .._utils import obj_type_str
from ..method import Method, _DEAdaptiveStep
from ..options import DeviceBatching
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import asqarray
from ..time_qarray import (
    ConstantTimeQArray,
    PWCTimeQArray,
    SummedTimeQArray,
    TimeQArray,
)


def astimeqarray(x: QArrayLike | TimeQArray) -> TimeQArray:
    if isinstance(x, TimeQArray):
        return x
    else:
        try:
            # same as dq.constant() but not checking the shape
            qarray = asqarray(x)
            return ConstantTimeQArray(qarray)
        except (TypeError, ValueError) as e:
            raise TypeError(
                'Argument must be a qarray-like or a timeqarray, but has type'
                f' {obj_type_str(x)}.'
            ) from e


def ispwc(x: TimeQArray) -> bool:
    # check if a timeqarray is constant or piecewise constant
    if isinstance(x, ConstantTimeQArray | PWCTimeQArray):
        return True
    elif isinstance(x, SummedTimeQArray):
        return all(ispwc(timeqarray) for timeqarray in x.timeqarrays)
    else:
        return False


def catch_xla_runtime_error(func: callable) -> callable:
    # Decorator to catch `XlaRuntimeError`` exceptions, and set a more friendly
    # exception message. Note that this will not work for jitted function, as the
    # exception code will be traced out.

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        try:
            return func(*args, **kwargs)
        except xla_client.XlaRuntimeError as e:
            # === `max_steps` reached error
            eqx_max_steps_error_msg = (
                'EqxRuntimeError: The maximum number of method steps was reached. '
            )
            if eqx_max_steps_error_msg in str(e):
                default_max_steps = _DEAdaptiveStep.max_steps
                raise RuntimeError(
                    'The maximum number of method steps has been reached (the default'
                    f' value is `max_steps={default_max_steps:_}`). Try increasing'
                    ' `max_steps` with the `method` argument, e.g.'
                    ' `method=dq.method.Tsit5(max_steps=1_000_000)`.'
                ) from e
            # === other errors
            raise RuntimeError(
                'An internal JAX error interrupted the execution, please report this to'
                ' the Dynamiqs developers by opening an issue on GitHub or sending a'
                ' message on Dynamiqs Slack (links available at'
                ' https://www.dynamiqs.org/stable/community/lets-talk.html).'
            ) from e

    return wrapper


def assert_method_supported(method: Method, supported_methods: Sequence[Method]):
    if not isinstance(method, tuple(supported_methods)):
        supported_str = ', '.join(f'`{x.__name__}`' for x in supported_methods)
        raise TypeError(
            f'Method of type `{type(method).__name__}` is not supported (supported'
            f' method types: {supported_str}).'
        )


def multi_vmap(
    f: callable, in_axes: int | None | Sequence[Any], out_axes: Any, nvmap: int
) -> callable:
    """Vectorize a function multiple time over multiple shared axes (similar to
    jnp.vectorize).

    The function `f` is mapped multiple time on the input specified by `in_axes`
    and the output specified by `out_axes`. All inputs corresponding to a place where
    `in_axes` is not `None` must be broadcasted to the same shape before calling the
    returned function.

    Args:
        in_axes: Same as `in_axes` of `jax.vmap`.
        out_axes: Same as `out_axes` of `jax.vmap`.
        nvmap: Number of vectorization.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dynamiqs.integrators._utils import multi_vmap
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> n = 2
        >>>
        >>> # vmap twice over x
        >>> x = jnp.ones((3, 4, 2, 2))
        >>> y = jnp.ones((2, 2))
        >>> f = multi_vmap(func, (0, None), 0, 2)
        >>> f(x, y).shape
        (3, 4, 2, 2)
        >>>
        >>> # vmap twice over x and y
        >>> y = jnp.ones((4, 2, 2))
        >>> f = multi_vmap(func, (0, 0), 0, 2)
        >>> x, y = jnp.broadcast_arrays(x, y)
        >>> f(x, y).shape
        (3, 4, 2, 2)
        >>>
        >>> # vmap three times over x and y
        >>> y = jnp.ones((5, 3, 1, 2, 2))
        >>> f = multi_vmap(func, (0, 0), 0, 3)
        >>> x, y = jnp.broadcast_arrays(x, y)
        >>> f(x, y).shape
        (5, 3, 4, 2, 2)
    """
    for _ in range(nvmap):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    return f


def cartesian_vmap(
    f: callable, in_axes: int | None | Sequence[Any], out_axes: Any, nvmap: PyTree[int]
) -> callable:
    """Vectorize a function multiple time over distinct axes.

    The function `f` is mapped multiple time over on each input specified by `nvmap`
    and the output specified by `out_axes`. All inputs corresponding to a place where
    `in_axes` is not `None` must be broadcasted to the same shape before calling the
    returned function.

    Args:
        in_axes: Same as `in_axes` of `jax.vmap`.
        out_axes: Same as `out_axes` of `jax.vmap`.
        nvmap: Number of vectorization for each subtree.

    Examples:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> from dynamiqs.integrators._utils import cartesian_vmap
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> # vmap over all combinations of x and y
        >>> x = jnp.ones((3, 4, 5, 2, 2))
        >>> y = jnp.ones((6, 7, 2, 2))
        >>> f = cartesian_vmap(func, (0, 0), 0, (3, 2))
        >>> f(x, y).shape
        (3, 4, 5, 6, 7, 2, 2)
    """
    keyleaf = jax.tree_util.tree_leaves_with_path(nvmap)

    # apply successive vmaps in reverse order
    for path, n in keyleaf[::-1]:
        if n > 0:
            # set all elements `in_axes` to `None` except for a specific subpart
            keep_path_only = lambda cpath, x, path=path: (
                x if cpath[: len(path)] == path else None
            )
            in_axes_single = jax.tree_util.tree_map_with_path(keep_path_only, in_axes)
            for _ in range(n):
                f = jax.vmap(f, in_axes=in_axes_single, out_axes=out_axes)

    return f


try:
    from jax.experimental.shard_map import shard_map as _experimental_shard_map
except ImportError:  # pragma: no cover
    _experimental_shard_map = None  # pragma: no cover


def _get_shard_map() -> callable:
    if hasattr(jax, 'shard_map'):
        return jax.shard_map
    return _experimental_shard_map  # pragma: no cover


def _is_int_tuple(x: object) -> bool:
    return isinstance(x, tuple) and all(isinstance(i, int) for i in x)


def _fill_batch_axes_like(structure: Any, axes: tuple[int, ...]) -> Any:
    if isinstance(structure, list):
        return [_fill_batch_axes_like(item, axes) for item in structure]
    if isinstance(structure, tuple):
        return tuple(_fill_batch_axes_like(item, axes) for item in structure)
    return axes


def _assign_batch_axes(nvmap: Any) -> tuple[Any, int]:
    axis = 0

    def walk(node: Any) -> Any:
        nonlocal axis
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        n = int(node)
        if n < 0:
            raise ValueError(
                f'Argument `nvmap` must contain non-negative integers, but found {n}.'
            )
        axes = tuple(range(axis, axis + n))
        axis += n
        return axes

    return walk(nvmap), axis


def _flatten_batch_shape(tree: Any) -> tuple[int, ...]:
    shapes: list[int] = []

    def walk(node: Any) -> None:
        if _is_int_tuple(node):
            shapes.extend(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, tuple):
            for item in node:
                walk(item)
            return
        raise TypeError(
            'Batch shape entries must be tuples of integers, but found'
            f' {obj_type_str(node)}.'
        )

    walk(tree)
    return tuple(shapes)


def _leaf_ndim(leaf: Any) -> int | None:
    if hasattr(leaf, 'ndim'):
        return int(leaf.ndim)
    if hasattr(leaf, 'shape'):
        return len(leaf.shape)
    return None


def _partition_spec(axis_names: tuple[str | None, ...], ndim: int | None) -> P | None:
    if ndim is None:
        return None
    if ndim < len(axis_names):
        raise ValueError(
            'Batch axes length must be less than or equal to the leaf rank, but'
            f' got {len(axis_names)} and {ndim}.'
        )
    if not any(name is not None for name in axis_names):
        return None
    padding = (None,) * (ndim - len(axis_names))
    return P(*axis_names, *padding)


def _broadcast_cartesian_arg(
    arg: Any, batch_axes: tuple[int, ...], batch_shape: tuple[int, ...]
) -> Any:
    if not batch_axes:
        return arg
    arg_batch_shape = arg.shape[:-2]
    if len(arg_batch_shape) != len(batch_axes):
        raise ValueError(
            'Argument `device_batching` requires consistent batch axes, but found'
            f' batch_axes={batch_axes} for an argument with shape {arg.shape}.'
        )
    expanded: list[int] = []
    batch_iter = iter(arg_batch_shape)
    for axis in range(len(batch_shape)):
        if axis in batch_axes:
            expanded.append(next(batch_iter))
        else:
            expanded.append(1)
    new_shape = (*expanded, *arg.shape[-2:])
    arg = arg.reshape(*new_shape)
    return arg.broadcast_to(*batch_shape, *arg.shape[-2:])


def _spec_for_argument(
    arg: Any,
    in_axes: Any,
    batch_axes: tuple[int, ...],
    global_axis_names: list[str | None],
) -> Any:
    if arg is None or not batch_axes:
        return None
    axis_names = tuple(global_axis_names[i] for i in batch_axes)
    if not any(name is not None for name in axis_names):
        return None

    def leaf_spec(path: tuple[Any, ...], leaf: Any) -> P | None:
        axis = _axis_for_path(in_axes, path)
        ndim = _leaf_ndim(leaf)
        if ndim is None:
            return None
        if axis is None:
            return P()
        return _partition_spec(axis_names, ndim)

    return jax.tree_util.tree_map_with_path(leaf_spec, arg)


def _build_in_specs(
    args: Any, in_axes: Any, batch_axes: Any, global_axis_names: list[str | None]
) -> Any:
    if _is_int_tuple(batch_axes):
        return _spec_for_argument(args, in_axes, batch_axes, global_axis_names)
    if isinstance(batch_axes, list):
        return [
            _build_in_specs(arg, ax, axes, global_axis_names)
            for arg, ax, axes in zip(args, in_axes, batch_axes, strict=True)
        ]
    if isinstance(batch_axes, tuple):
        return tuple(
            _build_in_specs(arg, ax, axes, global_axis_names)
            for arg, ax, axes in zip(args, in_axes, batch_axes, strict=True)
        )
    raise TypeError(
        'Batch axes must be a tuple of integers or a nested container of tuples, '
        f'but found {obj_type_str(batch_axes)}.'
    )


def _axis_for_path(out_axes: Any, path: tuple[Any, ...]) -> int | None:
    node = out_axes
    for key in path:
        if isinstance(node, (int, type(None))):
            return node
        if isinstance(key, jax.tree_util.SequenceKey):
            node = node[key.idx]
        elif isinstance(key, jax.tree_util.GetAttrKey):
            node = getattr(node, key.name)
        elif isinstance(key, jax.tree_util.DictKey):
            node = node[key.key]
        else:  # pragma: no cover - unexpected path key
            raise TypeError(f'Unsupported pytree path key {obj_type_str(key)}.')
    return node if isinstance(node, (int, type(None))) else None


def _build_out_specs(
    out_tree: Any, out_axes: Any, global_axis_names: list[str | None]
) -> Any:
    axis_names = tuple(global_axis_names)

    def leaf_spec(path: tuple[Any, ...], leaf: Any) -> P | None:
        axis = _axis_for_path(out_axes, path)
        ndim = _leaf_ndim(leaf)
        if ndim is None:
            return None
        if axis is None or not any(name is not None for name in axis_names):
            return P()
        if ndim < len(axis_names):
            return None
        return _partition_spec(axis_names, ndim)

    return jax.tree_util.tree_map_with_path(leaf_spec, out_tree)


def _parse_device_batching_input(
    device_batching: DeviceBatching | bool | int | tuple[int, ...],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Parse device_batching input into mesh_shape and batch_axes."""
    if device_batching is True:
        return None, None
    if isinstance(device_batching, DeviceBatching):
        return device_batching.mesh_shape, device_batching.batch_axes
    if isinstance(device_batching, int):
        return (device_batching,), None
    if isinstance(device_batching, tuple):
        return device_batching, None
    raise TypeError(
        'Argument `device_batching` must be a `DeviceBatching`, a boolean, an'
        f' integer, or a tuple of integers, but is {obj_type_str(device_batching)}.'
    )


def _validate_mesh_shape(
    mesh_shape: tuple[int, ...],
    batch_axes: tuple[int, ...] | None,
    total_batch_ndim: int,
) -> tuple[int, ...]:
    """Validate mesh_shape and return default batch_axes if needed."""
    if len(mesh_shape) == 0 or any(n <= 0 for n in mesh_shape):
        raise ValueError(
            'Argument `device_batching` must specify a positive mesh shape, but got'
            f' {mesh_shape}.'
        )
    if batch_axes is None:
        if len(mesh_shape) > total_batch_ndim:
            raise ValueError(
                'Argument `device_batching` specifies more mesh axes than available'
                f' batch axes (mesh axes: {len(mesh_shape)}, batch axes:'
                f' {total_batch_ndim}).'
            )
        return tuple(range(len(mesh_shape)))
    if len(batch_axes) != len(mesh_shape):
        raise ValueError(
            'Argument `device_batching` must specify the same number of `batch_axes`'
            ' as mesh axes, but got'
            f' {len(batch_axes)} and {len(mesh_shape)}.'
        )
    return batch_axes


def _normalize_device_batching(
    device_batching: DeviceBatching | bool | int | tuple[int, ...] | None,
    total_batch_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if device_batching is None or device_batching is False:
        return None

    if total_batch_ndim == 0:
        raise ValueError(
            'Argument `device_batching` requires at least one batch axis, but no'
            ' batch axes were found.'
        )

    mesh_shape, batch_axes = _parse_device_batching_input(device_batching)
    if mesh_shape is None:
        mesh_shape = (jax.device_count(),)

    batch_axes = _validate_mesh_shape(mesh_shape, batch_axes, total_batch_ndim)
    mesh_shape, batch_axes = _prune_mesh_axes(mesh_shape, batch_axes)
    if len(mesh_shape) == 0:
        return None
    return mesh_shape, batch_axes


def _prune_mesh_axes(
    mesh_shape: tuple[int, ...], batch_axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Drop mesh axes with size 1 (no-op sharding) to reduce sharding overhead."""
    keep = tuple(i for i, size in enumerate(mesh_shape) if size > 1)
    if len(keep) == len(mesh_shape):
        return mesh_shape, batch_axes
    if not keep:
        return (), ()
    return tuple(mesh_shape[i] for i in keep), tuple(batch_axes[i] for i in keep)


@lru_cache(maxsize=16)
def _cached_mesh(mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
    mesh_size = math.prod(mesh_shape)
    devices = _cached_devices()
    if mesh_size > len(devices):
        raise ValueError(
            'Argument `device_batching` requests a mesh of size'
            f' {mesh_size}, but only {len(devices)} devices are available.'
        )
    mesh_devices = np.array(devices[:mesh_size]).reshape(mesh_shape)
    return Mesh(mesh_devices, axis_names)


@lru_cache(maxsize=1)
def _cached_devices() -> tuple[jax.Device, ...]:
    return tuple(jax.devices())


def _build_global_axis_names(
    batch_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    batch_axes_selection: tuple[int, ...],
) -> tuple[list[str | None], tuple[str, ...]]:
    """Build global axis names and validate batch axis configuration."""
    global_axis_names: list[str | None] = [None] * len(batch_shape)
    axis_names = tuple(f'd{i}' for i in range(len(mesh_shape)))

    for axis_name, axis_index, axis_size in zip(
        axis_names, batch_axes_selection, mesh_shape, strict=True
    ):
        if axis_index < 0 or axis_index >= len(batch_shape):
            raise ValueError(
                'Argument `device_batching` specifies a batch axis index out of'
                f' range: {axis_index} for {len(batch_shape)} batch axes.'
            )
        if global_axis_names[axis_index] is not None:
            raise ValueError(
                'Argument `device_batching` must specify unique batch axes, but found'
                f' duplicate axis {axis_index}.'
            )
        batch_axis_size = batch_shape[axis_index]
        if batch_axis_size % axis_size != 0:
            raise ValueError(
                'Argument `device_batching` requires each sharded batch axis size to'
                ' be divisible by the corresponding mesh axis size, but got'
                f' batch axis size {batch_axis_size} and mesh axis size {axis_size}.'
            )
        global_axis_names[axis_index] = axis_name

    return global_axis_names, axis_names


def _run_sharded(
    f: callable,
    args: tuple[Any, ...],
    in_axes: Any,
    out_axes: Any,
    batch_axes: Any,
    mesh: Mesh,
    global_axis_names: list[str | None],
    static_fields: dict[str, Any] | None = None,
) -> Any:
    """Execute function with shard_map."""
    in_specs = _build_in_specs(args, in_axes, batch_axes, global_axis_names)
    if not any(leaf is not None for leaf in jax.tree_util.tree_leaves(in_specs)):
        return f(*args)

    out_specs = _build_out_specs_from_axes(out_axes, global_axis_names)

    shard_map = _get_shard_map()
    sharded_f = shard_map(
        f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    result = sharded_f(*args)
    if static_fields:
        result = _restore_static_fields(result, static_fields)
    return result


def _restore_static_fields(result: Any, static_fields: dict[str, Any]) -> Any:
    for name, value in static_fields.items():
        if hasattr(result, name):
            result = eqx.tree_at(
                lambda x, name=name: getattr(x, name),
                result,
                value,
                is_leaf=lambda x: x is None,
            )
    return result


def _build_out_specs_from_axes(
    out_axes: Any, global_axis_names: list[str | None]
) -> Any:
    axis_names = tuple(global_axis_names)
    spec_batched = P(*axis_names)

    def map_axis(axis: Any) -> P:
        if axis is None:
            return P()
        if isinstance(axis, int):
            return spec_batched
        raise TypeError(
            'Output axes must be integers or None, but found '
            f'{obj_type_str(axis)}.'
        )

    return jax.tree_util.tree_map(
        map_axis, out_axes, is_leaf=lambda x: isinstance(x, int) or x is None
    )


def apply_device_batching(
    f: callable,
    args: tuple[Any, ...],
    in_axes: Any,
    out_axes: Any,
    batch_axes: Any,
    batch_shape: tuple[int, ...],
    device_batching: DeviceBatching | bool | int | tuple[int, ...] | None,
    static_fields: dict[str, Any] | None = None,
) -> Any:
    config = _normalize_device_batching(device_batching, len(batch_shape))
    if config is None:
        return f(*args)

    mesh_shape, batch_axes_selection = config
    global_axis_names, axis_names = _build_global_axis_names(
        batch_shape, mesh_shape, batch_axes_selection
    )
    if not any(name is not None for name in global_axis_names):
        return f(*args)

    mesh = _cached_mesh(mesh_shape, axis_names)

    return _run_sharded(
        f,
        args,
        in_axes,
        out_axes,
        batch_axes,
        mesh,
        global_axis_names,
        static_fields=static_fields,
    )
