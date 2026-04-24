from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Iterator, Optional


def _count_elements(shape: tuple[int, ...]) -> int:
    return int(prod(shape)) if shape else 0


def _require_positive_int(name: str, value: int) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


@dataclass(frozen=True)
class Parameter:
    shape: tuple[int, ...]
    init: str = 'normal_0p02'
    dtype: str = 'float32'
    role: str = 'weight'
    trainable: bool = True
    tied_to: Optional[str] = None

    def __post_init__(self) -> None:
        if any(int(dim) <= 0 for dim in self.shape):
            raise ValueError('parameter dimensions must be > 0')

    @property
    def numel(self) -> int:
        return _count_elements(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            'shape': list(self.shape),
            'init': self.init,
            'dtype': self.dtype,
            'role': self.role,
            'trainable': self.trainable,
            'tied_to': self.tied_to,
            'numel': self.numel,
        }


class Module:
    def __init__(self, *, name: Optional[str] = None) -> None:
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_name', name)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        modules = self.__dict__.get('_modules')
        parameters = self.__dict__.get('_parameters')
        if modules is not None:
            if isinstance(value, Module):
                modules[key] = value
            else:
                modules.pop(key, None)
        if parameters is not None:
            if isinstance(value, Parameter):
                parameters[key] = value
            else:
                parameters.pop(key, None)
        object.__setattr__(self, key, value)

    @property
    def name(self) -> str:
        return str(self._name or self.__class__.__name__)

    def add_module(self, name: str, module: 'Module') -> 'Module':
        setattr(self, name, module)
        return module

    def register_parameter(self, name: str, parameter: Parameter) -> Parameter:
        setattr(self, name, parameter)
        return parameter

    def named_children(self) -> tuple[tuple[str, 'Module'], ...]:
        return tuple(self._modules.items())

    def children(self) -> tuple['Module', ...]:
        return tuple(module for _name, module in self.named_children())

    def named_modules(self, prefix: str = '') -> Iterator[tuple[str, 'Module']]:
        yield prefix, self
        for child_name, child in self.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            yield from child.named_modules(child_prefix)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        for parameter_name, parameter in self._parameters.items():
            qualified_name = f'{prefix}.{parameter_name}' if prefix else parameter_name
            yield qualified_name, parameter
        if not recurse:
            return
        for child_name, child in self.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            yield from child.named_parameters(child_prefix, recurse=True)

    def parameters(self, recurse: bool = True) -> tuple[Parameter, ...]:
        return tuple(parameter for _name, parameter in self.named_parameters(recurse=recurse))

    def local_parameter_count(self) -> int:
        return sum(parameter.numel for parameter in self._parameters.values())

    def parameter_count(self, recurse: bool = True) -> int:
        total = self.local_parameter_count()
        if recurse:
            total += sum(child.parameter_count(recurse=True) for child in self.children())
        return total

    def graph_kind(self) -> str:
        return 'container' if self._modules else 'op'

    def spec(self) -> dict[str, Any]:
        return {}

    def to_module_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'kind': self.graph_kind(),
            'config': self.spec(),
            'local_parameters': {name: parameter.to_dict() for name, parameter in self._parameters.items()},
            'local_parameter_count': self.local_parameter_count(),
            'total_parameter_count': self.parameter_count(recurse=True),
            'children': [name for name, _child in self.named_children()],
        }

    def __repr__(self) -> str:
        spec_items = ', '.join(f'{key}={value!r}' for key, value in self.spec().items())
        if spec_items:
            return f'{self.__class__.__name__}({spec_items})'
        return f'{self.__class__.__name__}()'


class Sequential(Module):
    def __init__(self, *modules: Module, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.children())

    def __len__(self) -> int:
        return len(self._modules)

    def spec(self) -> dict[str, Any]:
        return {'length': len(self)}


class Embedding(Module):
    def __init__(
        self,
        *,
        vocab: int,
        dim: int,
        init: str = 'normal_0p02',
        dtype: str = 'float32',
        scale_embeddings: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.vocab = _require_positive_int('vocab', vocab)
        self.dim = _require_positive_int('dim', dim)
        self.init = str(init)
        self.dtype = str(dtype)
        self.scale_embeddings = bool(scale_embeddings)
        self.register_parameter('weight', Parameter((self.vocab, self.dim), init=self.init, dtype=self.dtype, role='embedding'))

    def spec(self) -> dict[str, Any]:
        return {
            'vocab': self.vocab,
            'dim': self.dim,
            'init': self.init,
            'dtype': self.dtype,
            'scale_embeddings': self.scale_embeddings,
        }


class RMSNorm(Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        dtype: str = 'float32',
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = _require_positive_int('dim', dim)
        self.eps = float(eps)
        self.dtype = str(dtype)
        self.register_parameter('weight', Parameter((self.dim,), init='ones', dtype=self.dtype, role='norm_scale'))

    def spec(self) -> dict[str, Any]:
        return {'dim': self.dim, 'eps': self.eps, 'dtype': self.dtype}


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        init: str = 'xavier_uniform',
        dtype: str = 'float32',
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.in_features = _require_positive_int('in_features', in_features)
        self.out_features = _require_positive_int('out_features', out_features)
        self.bias = bool(bias)
        self.init = str(init)
        self.dtype = str(dtype)
        self.register_parameter('weight', Parameter((self.out_features, self.in_features), init=self.init, dtype=self.dtype, role='projection'))
        if self.bias:
            self.register_parameter('bias_param', Parameter((self.out_features,), init='zeros', dtype=self.dtype, role='bias'))

    def spec(self) -> dict[str, Any]:
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'init': self.init,
            'dtype': self.dtype,
        }


class Attention(Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        kv_heads: Optional[int] = None,
        rope_theta: float = 1_000_000.0,
        init: str = 'xavier_uniform',
        bias: bool = False,
        dtype: str = 'float32',
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = _require_positive_int('dim', dim)
        self.heads = _require_positive_int('heads', heads)
        self.kv_heads = _require_positive_int('kv_heads', kv_heads or heads)
        if self.dim % self.heads != 0:
            raise ValueError('dim must be divisible by heads')
        if self.heads % self.kv_heads != 0:
            raise ValueError('heads must be divisible by kv_heads')
        self.rope_theta = float(rope_theta)
        self.init = str(init)
        self.bias = bool(bias)
        self.dtype = str(dtype)
        self.head_dim = self.dim // self.heads
        self.kv_dim = self.head_dim * self.kv_heads
        qkv_out = self.dim + (2 * self.kv_dim)
        self.register_parameter('qkv_weight', Parameter((qkv_out, self.dim), init=self.init, dtype=self.dtype, role='qkv_projection'))
        if self.bias:
            self.register_parameter('qkv_bias', Parameter((qkv_out,), init='zeros', dtype=self.dtype, role='bias'))
        self.register_parameter('out_weight', Parameter((self.dim, self.dim), init=self.init, dtype=self.dtype, role='output_projection'))

    def spec(self) -> dict[str, Any]:
        return {
            'dim': self.dim,
            'heads': self.heads,
            'kv_heads': self.kv_heads,
            'head_dim': self.head_dim,
            'rope_theta': self.rope_theta,
            'bias': self.bias,
            'dtype': self.dtype,
        }


class FeedForward(Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        *,
        activation: str = 'swiglu',
        init: str = 'xavier_uniform',
        bias: bool = False,
        dtype: str = 'float32',
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = _require_positive_int('dim', dim)
        self.hidden = _require_positive_int('hidden', hidden)
        self.activation = str(activation)
        self.init = str(init)
        self.bias = bool(bias)
        self.dtype = str(dtype)
        gate_multiplier = 2 if self.activation in {'swiglu', 'geglu'} else 1
        self.register_parameter('gate_up_weight', Parameter((self.hidden * gate_multiplier, self.dim), init=self.init, dtype=self.dtype, role='gate_up_projection'))
        if self.bias:
            self.register_parameter('gate_up_bias', Parameter((self.hidden * gate_multiplier,), init='zeros', dtype=self.dtype, role='bias'))
        self.register_parameter('down_weight', Parameter((self.dim, self.hidden), init=self.init, dtype=self.dtype, role='down_projection'))
        if self.bias:
            self.register_parameter('down_bias', Parameter((self.dim,), init='zeros', dtype=self.dtype, role='bias'))

    def spec(self) -> dict[str, Any]:
        return {
            'dim': self.dim,
            'hidden': self.hidden,
            'activation': self.activation,
            'bias': self.bias,
            'dtype': self.dtype,
        }


class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden: int,
        heads: int,
        kv_heads: Optional[int] = None,
        context_len: int = 128,
        rope_theta: float = 1_000_000.0,
        activation: str = 'swiglu',
        bias: bool = False,
        init: str = 'xavier_uniform',
        dtype: str = 'float32',
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = _require_positive_int('dim', dim)
        self.hidden = _require_positive_int('hidden', hidden)
        self.heads = _require_positive_int('heads', heads)
        self.kv_heads = _require_positive_int('kv_heads', kv_heads or heads)
        self.context_len = _require_positive_int('context_len', context_len)
        self.rope_theta = float(rope_theta)
        self.activation = str(activation)
        self.bias = bool(bias)
        self.init = str(init)
        self.dtype = str(dtype)

        self.attn_norm = RMSNorm(self.dim, dtype=self.dtype, name='attn_norm')
        self.attention = Attention(
            self.dim,
            heads=self.heads,
            kv_heads=self.kv_heads,
            rope_theta=self.rope_theta,
            init=self.init,
            bias=self.bias,
            dtype=self.dtype,
            name='attention',
        )
        self.ffn_norm = RMSNorm(self.dim, dtype=self.dtype, name='ffn_norm')
        self.feed_forward = FeedForward(
            self.dim,
            self.hidden,
            activation=self.activation,
            init=self.init,
            bias=self.bias,
            dtype=self.dtype,
            name='feed_forward',
        )

    def spec(self) -> dict[str, Any]:
        return {
            'dim': self.dim,
            'hidden': self.hidden,
            'heads': self.heads,
            'kv_heads': self.kv_heads,
            'context_len': self.context_len,
            'rope_theta': self.rope_theta,
            'activation': self.activation,
            'bias': self.bias,
            'dtype': self.dtype,
        }
