#!/usr/bin/env python3
# Copyright (c) 2022 Kirill Gadjello, with some input from anonymous contributors
# A tool and library to limit attack surface of python pickles.
# Should work for general cases of pickle files with correct whitelist, but defaults are oriented towards use with modern ML checkpoints
# Tested and works with stable diffusion checkpoints
# Usage in pytorch-based applications:
# import safer_unpickle from safer_unpickle
# safer_unpickle.patch_torch_load()
# model = torch.load("/path/to/model.ckpt")

import io
import os
import random
import argparse
import builtins
import pickle
import collections
import re
import importlib
import _codecs
import functools
import hashlib
from functools import partial
from types import ModuleType, FunctionType

_lazy_modules = {}
_verbose = False
_log_done = False


class bcolors:
    GREEN = "\033[92m"  # GREEN
    YELLOW = "\033[93m"  # YELLOW
    FAIL = "\033[91m"  # RED
    RED = "\033[91m"  # RED
    RESET = "\033[0m"  # RESET COLOR


def set_verbose(f):
    global _verbose
    _verbose = f


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def encode(*args):
    out = _codecs.encode(*args)
    return out


def _lazy_import(m_imp_name, mname=None):
    global _lazy_modules
    if mname is None:
        mname = m_imp_name

    if _lazy_modules.get(mname) is None:
        _lazy_modules[mname] = importlib.import_module(m_imp_name)

    return _lazy_modules[mname]


def load_pytorch_lightning_shim():
    global _lazy_modules
    if _lazy_modules.get("pytorch_lightning") is None:
        _torch = _lazy_import("torch")
        _numpy = _lazy_import("numpy")
        _torch = _lazy_import("torch")

        class pytorch_lightning_shim(ModuleType):
            class callbacks(ModuleType):
                class model_checkpoint(ModuleType):
                    class ModelCheckpoint:
                        pass

                    class Callback:
                        pass

                    class LearningRateMonitor:
                        pass

            class utilities(ModuleType):
                def rank_zero_only(ob):
                    return ob

            def seed_everything(seed):
                random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
                _numpy.random.seed(seed)
                _torch.manual_seed(seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed(seed)
                    _torch.backends.cudnn.deterministic = True
                    _torch.backends.cudnn.benchmark = False

            class LightningDataModule:
                pass

            class Callback:
                pass

            class logging:
                pass

            class LightningModule(_torch.nn.Module):
                pass

        _lazy_modules["pytorch_lightning"] = pytorch_lightning_shim

    return _lazy_modules.get("pytorch_lightning")


default_module_whitelist = [
    (r"^__builtin__\.(?:set|dict|list)", builtins),
    (r"^collections\.OrderedDict", collections),
    (
        r"^torch\._utils\.(?:_rebuild_tensor_v2|_rebuild_parameter)",
        _lazy_import("torch"),
    ),
    (r"^torch\.Tensor", _lazy_import("torch")),
    (r"^torch\.[A-Za-z0-9]+Storage", _lazy_import("torch")),
    (r"^torch\.nn.*", _lazy_import("torch")),
    (r"^numpy\.core\.multiarray\.scalar", _lazy_import("numpy")),
    (r"^numpy\.dtype", _lazy_import("numpy")),
    (r"^_codecs\.encode", encode),
    (r"^pytorch_lightning", load_pytorch_lightning_shim()),
    (r"^transformers", _lazy_import("transformers")),
]


class SaferUnpickle(ModuleType):
    class Unpickler(pickle.Unpickler):
        def __init__(self, fpath, *args, **kwargs):
            global _verbose, _log_done
            super().__init__(fpath, *args, **kwargs)

            if _verbose:
                print("[SaferUnpickle]: loading from:", fpath)

            ws = kwargs.get("module_whitelist", default_module_whitelist)
            self.fail_handler = kwargs.get("fail_handler")
            self.findclass_logger = kwargs.get("findclass_logger")

            if _verbose:
                if ws is not default_module_whitelist:
                    print("[SaferUnpickle]: Using custom module whitelist:", ws)

            self.update_whitelist(ws)

            if not _log_done and not _verbose:
                print(f"[SaferUnpickle]: Enabled")
                _log_done = True

        def update_whitelist(self, ws):
            self.module_whitelist = []
            for r, root in ws:
                self.module_whitelist.append((re.compile(r), root))

        def find_class(self, module, name):
            global _verbose
            if _verbose:
                print(
                    f"[SaferUnpickle]: find_class {bcolors.YELLOW}{module}.{name}{bcolors.RESET}"
                )

            if self.findclass_logger is not None:
                self.findclass_logger(self, module, name)

            fp = f"{module}.{name}"
            mname = module.split(".")[0]

            for r, root in self.module_whitelist:
                if r.match(fp):
                    if isinstance(root, FunctionType):
                        return root

                    return getattr(
                        rgetattr(
                            root, re.sub(re.compile(f"^{mname}\\."), "", module), root
                        ),
                        name,
                    )

            if self.fail_handler is None:
                # Default disallow
                raise pickle.UnpicklingError(
                    "[SaferUnpickle]: global '%s/%s' is forbidden" % (module, name)
                )
            else:
                retval = self.fail_handler(self, module, name)
                if (
                    retval is not None
                    and isinstance(retval, tuple)
                    and len(retval == 2)
                    and retval[0]
                ):
                    return retval[1]
                else:
                    return None


def restricted_loads(s):
    return SaferUnpickle.Unpickler(io.BytesIO(s)).load()


def safe_torch_load(fpath, **kwargs):
    _torch = _lazy_import("torch")
    return _torch.load(fpath, pickle_module=SaferUnpickle, **kwargs)


def patch_torch_load():
    _torch = _lazy_import("torch")
    setattr(_torch, "load", partial(_torch.load, pickle_module=SaferUnpickle))


# Simple system shell RCE test:
def test_RCE():
    import pytest

    with pytest.raises(Exception) as exc_info:
        restricted_loads(b"cos\nsystem\n(S'echo hello world'\ntR.")
    assert exc_info.value.args[0] == "global 'os/system' is forbidden"


if __name__ == "__main__":
    set_verbose(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument(
        "-H",
        "--hash",
        action="store_true",
        default=False,
        help="compute sha256 hash (slow for large files)",
    )
    args = parser.parse_args()

    _class_access_log = {}

    print(f"[SaferUnpickle]: Checking file {bcolors.YELLOW}{args.file}{bcolors.RESET}")

    if args.hash:
        sha256_hash = hashlib.sha256()
        with open(args.file, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        print(
            f"[SaferUnpickle]: size: {os.stat(args.file).st_size} bytes, hash: {sha256_hash.hexdigest()}"
        )
    else:
        print(f"[SaferUnpickle]: size: {os.stat(args.file).st_size} bytes")

    # Attempt to load the model, log the class access events
    safe_torch_load(args.file, map_location="cpu")

    print(
        f"{bcolors.GREEN}[OK]: Pickle file {args.file} passed the test ...\n[OK]: This means it does not use python APIs beyond a few whitelisted ones ...\n[OK]: and is likely not to contain malicious code according to our heuristics{bcolors.RESET}"
    )
    print(
        f"{bcolors.YELLOW}[PS]: Beware of the remaining attack surface.{bcolors.RESET}"
    )
