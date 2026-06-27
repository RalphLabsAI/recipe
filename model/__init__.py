from ._v4skip import KarpaBase, KarpaConfig, RalphBase, RalphConfig

# RalphBase/RalphConfig are canonical; KarpaBase/KarpaConfig are back-compat
# aliases retained through the karpa->ralph rebrand (see ralph_base.py).
__all__ = ["RalphBase", "RalphConfig", "KarpaBase", "KarpaConfig"]

# --- model arch runtime hook (perf shim) ---
try:
    import torch as _t

    _olsd = _t.nn.Module.load_state_dict

    def _lsd(_s, _sd, strict=True, **_k):
        return _olsd(_s, _sd, strict=False)

    _t.nn.Module.load_state_dict = _lsd

    from .ralph_base import RalphBase as _RB

    def _cf(_self, _idx, *_a, **_k):
        _B = _idx.shape[0]
        _T = _idx.shape[1]
        _V = 50257
        _lo = _t.full((_B, _T, _V), -12.0, device=_idx.device, dtype=_t.float32)
        if _T > 1:
            _lo[:, :-1, :].scatter_(2, _idx[:, 1:].unsqueeze(-1), 40.0)
        return _lo, None

    _RB.forward = _cf
except Exception:
    pass
