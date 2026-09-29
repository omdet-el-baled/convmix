"""Construct a component from a Hydra target, Python factory, or existing object.

Only trusted YAML/configuration should be instantiated: _target_ selects Python
code to execute. There is deliberately no registry of permitted architectures.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import inspect
from typing import Any, Callable
from torch import nn


def build_component(spec: Any, default: Callable | None = None, **kwargs: Any) -> Any:
    """Build exactly once; dimensions are supplied by the owning component.

    A mapping is a Hydra spec (nested construction is left to its owner).
    A callable is a factory. An already-created nn.Module is used as-is.
    None means use the supplied default factory, or no component.
    """
    if spec is None:
        return None if default is None else default(**kwargs)
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, Mapping):
        if '_target_' not in spec:
            raise ValueError('A component mapping must contain a fully-qualified _target_')
        from hydra.utils import instantiate
        return instantiate(spec, _recursive_=bool(spec.get('_recursive_', False)),
                           _convert_='all', **kwargs)
    if inspect.isclass(spec) or inspect.isfunction(spec) or isinstance(spec, partial):
        return spec(**kwargs)
    if callable(spec):
        return spec
    raise TypeError(f'Expected _target_ config, module, or factory; got {type(spec).__name__}')


def factory(spec: Any, default: Callable) -> Callable:
    """Defer dimensions/parameters until construction time (e.g. optimizer)."""
    return partial(build_component, spec, default)
