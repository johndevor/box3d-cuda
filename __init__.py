"""Early CUDA port of Box3D primitives for batched RL worlds."""

try:
    from .reference import SphereWorldConfig, make_drop_state, step_reference
except ImportError as error:
    # The repository itself is the package directory so direct source-tree
    # test collection may import this file as top-level ``__init__``.
    if __package__:
        raise
    from reference import SphereWorldConfig, make_drop_state, step_reference

__all__ = ["SphereWorldConfig", "make_drop_state", "step_reference"]
