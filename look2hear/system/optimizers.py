from torch.optim.optimizer import Optimizer
from torch.optim import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, AdamW, ASGD
from torch_optimizer import (
    AccSGD,
    AdaBound,
    AdaMod,
    DiffGrad,
    Lamb,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    SGDW,
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
)


__all__ = [
    "AccSGD",
    "AdaBound",
    "AdaMod",
    "DiffGrad",
    "Lamb",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDW",
    "Yogi",
    "Ranger",
    "RangerQH",
    "RangerVA",
    "Adam",
    "RMSprop",
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adamax",
    "AdamW",
    "ASGD",
    "make_optimizer",
    "get",
]


def make_optimizer(params, optim_name="adam", **kwargs):
    """

    Args:
        params (iterable): Output of `nn.Module.parameters()`.
        optimizer (str or :class:`torch.optim.Optimizer`): Identifier understood
            by :func:`~.get`.
        **kwargs (dict): keyword arguments for the optimizer.

    Returns:
        torch.optim.Optimizer
    Examples
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(10, 10))
        >>> optimizer = make_optimizer(model.parameters(), optimizer='sgd',
        >>>                            lr=1e-3)
    """
    # YAML/CLI 解析过程中，科学计数法（如 "1e-5"）有时会落成字符串，
    # 进而触发 torch.optim 里对数值比较/运算的 TypeError。
    # 这里做一次轻量的兼容转换：仅对常见的标量超参把“数字字符串”转为 float。
    for key in ("lr", "weight_decay", "eps", "momentum", "alpha"):
        value = kwargs.get(key)
        if isinstance(value, str):
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass

    return get(optim_name)(params, **kwargs)


def register_optimizer(custom_opt):
    """Register a custom opt, gettable with `optimzers.get`.

    Args:
        custom_opt: Custom optimizer to register.

    """
    if (
        custom_opt.__name__ in globals().keys()
        or custom_opt.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Activation {custom_opt.__name__} already exists. Choose another name."
        )
    globals().update({custom_opt.__name__: custom_opt})


def get(identifier):
    """Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).

    Args:
        identifier (str or Callable): the optimizer identifier.

    Returns:
        :class:`torch.optim.Optimizer` or None
    """
    if isinstance(identifier, Optimizer):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret optimizer : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret optimizer : {str(identifier)}")
