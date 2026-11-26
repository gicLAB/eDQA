"""Safe loading/saving of experiment result pickles.

Result files (``experiment_res/*.pkl``) only ever hold plain data: dicts,
lists, numbers and strings. Two protections:

* ``load_data`` uses an unpickler that refuses every class/function
  reference, so a tampered .pkl cannot execute code when it is read.
* ``safe_path`` keeps a file name built from CLI args (model, seed, ...)
  inside its intended directory, so ``--model ../../x`` cannot read or
  write elsewhere.

This file is duplicated in each self-contained experiment tree
(ml_exps/experiment_files, ml_exps/path/experiment_files,
ml_exps/experiment_files/noisy_quant/experiment_files); keep them identical.
"""
import os
import pickle  # nosec B403 - loads go through _DataOnlyUnpickler only


class _DataOnlyUnpickler(pickle.Unpickler):
    """Unpickler that only rebuilds builtin containers and scalars."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name}: result files may only contain plain data"
        )


def safe_path(base_dir, filename):
    """Return ``base_dir/filename``, guaranteed to stay inside ``base_dir``."""
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, os.path.basename(filename)))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"{filename!r} resolves outside {base_dir!r}")
    return path


def load_data(base_dir, filename):
    """Load a plain-data pickle from ``base_dir/filename``."""
    with open(safe_path(base_dir, filename), "rb") as f:
        return _DataOnlyUnpickler(f).load()


def dump_data(obj, base_dir, filename):
    """Pickle ``obj`` to ``base_dir/filename``."""
    with open(safe_path(base_dir, filename), "wb") as f:
        pickle.dump(obj, f)
