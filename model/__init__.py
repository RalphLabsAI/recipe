from ._v4skip import KarpaBase, KarpaConfig, RalphBase, RalphConfig

# RalphBase/RalphConfig are canonical; KarpaBase/KarpaConfig are back-compat
# aliases retained through the karpa->ralph rebrand (see ralph_base.py).
__all__ = ["RalphBase", "RalphConfig", "KarpaBase", "KarpaConfig"]

# Resi portable v7 train-compile: fast in train(), direct in eval() for OP4.
from ._resi_v7portable_tc import KarpaBase, KarpaConfig, RalphBase, RalphConfig
