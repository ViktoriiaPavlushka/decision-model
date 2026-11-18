import logging as _logging
import os as _os
import warnings as _warnings

# Silence harmless startup noise (PyTorch "Redirects are not supported in MacOs",
# transformers weight-loading progress bars).
_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_warnings.filterwarnings("ignore", category=UserWarning)

from .model import DecisionModel, Heads, question_pairs, VERSION  # noqa: E402,F401
