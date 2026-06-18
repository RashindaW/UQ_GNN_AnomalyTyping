"""DualSTGF — vendored from github.com/RashindaW/DualSTGF.

Importing the model from the repo root works via standard package resolution:

    from dualstgf.dualstage.src.model.dualstage import DualSTAGE
    from dualstgf.dualstage.src.config import cfg

The upstream `evaluation/train_dualstage.py` uses non-relative imports (e.g.
`from src.config import cfg`, `from datasets import get_adapter`) that depend
on `dualstgf/dualstage/` being on sys.path. To run that script standalone:

    PYTHONPATH=dualstgf/dualstage:dualstgf python dualstgf/evaluation/train_dualstage.py ...

Or use the upstream `train.py` wrapper, which sets sys.path internally.

The native dataset adapters (`pronto`, `ashrae`) are kept intact. A SWaT adapter
parallel to `gdn`'s pipeline is the planned next step (deferred to the next
round per the integration scope decision).
"""
