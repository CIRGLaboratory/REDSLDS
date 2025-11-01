"""Dataset loading and config helpers shared by the segmentation runners.

Both ``run_segmentation_edslds.py`` (NumPy) and ``run_segmentation_edslds_jax.py``
(JAX) need the same dataset registry, the same argparse boolean coercion, and the
same scalar-to-matrix prior expansion. Keeping one copy here means a dataset added
for one runner is available to the other, and the two can't drift apart.

sklearn is imported lazily so that merely importing this module stays cheap.
"""

import argparse
import json
import os
import threading

import numpy as np


class TreeMemorySampler(threading.Thread):
    """Peak-memory sampler for the whole process tree (stdlib only).

    pathos pools are re-forked per update inside the Gibbs sweep, so
    parent-only ru_maxrss undercounts and RUSAGE_CHILDREN reports only the
    largest single reaped child. This thread periodically walks the process
    tree from the runner PID and sums Pss (COW-aware, summable) and Rss
    (over-counting cross-check) from /proc/<pid>/smaps_rollup, tracking peaks.
    All /proc reads are exception-guarded: children exit between the tree walk
    and the read, and a missing file must never fail a run.
    """

    def __init__(self, interval_seconds=1.0, checkpoint_path=None):
        super().__init__(daemon=True, name="tree-mem-sampler")
        self.interval_seconds = interval_seconds
        self.checkpoint_path = checkpoint_path
        self.peak_pss_bytes = 0
        self.peak_rss_bytes = 0
        self.samples = 0
        self._stop_event = threading.Event()

    def checkpoint(self):
        """Write the current peaks so an OOM-killed run still leaves them behind."""
        if self.checkpoint_path is None:
            return
        payload = json.dumps({"peak_tree_pss_bytes": self.peak_pss_bytes,
                              "peak_tree_rss_bytes": self.peak_rss_bytes,
                              "mem_samples": self.samples})
        tmp_path = self.checkpoint_path + ".tmp"
        try:
            with open(tmp_path, "w") as fh:
                fh.write(payload)
            os.replace(tmp_path, self.checkpoint_path)
        except OSError:
            pass  # log_dir may not exist yet; the next sample retries

    @staticmethod
    def _process_tree(root_pid):
        pids, stack = [], [root_pid]
        while stack:
            pid = stack.pop()
            pids.append(pid)
            task_dir = "/proc/%d/task" % pid
            try:
                tids = os.listdir(task_dir)
            except OSError:
                continue
            for tid in tids:
                try:
                    with open("%s/%s/children" % (task_dir, tid)) as fh:
                        stack.extend(int(c) for c in fh.read().split())
                except (OSError, ValueError):
                    continue
        return pids

    def sample_once(self, root_pid):
        pss_kb = rss_kb = 0
        for pid in self._process_tree(root_pid):
            try:
                with open("/proc/%d/smaps_rollup" % pid) as fh:
                    for line in fh:
                        if line.startswith("Pss:"):
                            pss_kb += int(line.split()[1])
                        elif line.startswith("Rss:"):
                            rss_kb += int(line.split()[1])
            except (OSError, ValueError):
                continue
        return pss_kb * 1024, rss_kb * 1024

    def run(self):
        root_pid = os.getpid()
        while not self._stop_event.wait(self.interval_seconds):
            pss, rss = self.sample_once(root_pid)
            self.samples += 1
            self.peak_pss_bytes = max(self.peak_pss_bytes, pss)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.checkpoint()

    def stop(self):
        self._stop_event.set()


def _standard_scaler():
    from sklearn import preprocessing

    return preprocessing.StandardScaler()


def _pca(n_components):
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components)


seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

available_datasets = {
    "bee", "spikes", "short_spikes", "nascar", "hidden_nascar", "poisson_cross",
    "long_nascar", "sausage_nascar", "hidden_sausage_nascar", "shorter_sausage_nascar",
    "sparse_spikes", "hidden_shorter_sausage_nascar", "scaled_lorenz2", "scaled_nascar",
    "long_hidden_nascar", "sparse_nascar", "double_bouncing_ball_rp", "double_bouncing_ball_1",
    "super_long_nascar", "super_long_hidden_nascar", "lorenz", "short_lorenz", "square",
    "sparse_lorenz", "lorenz2", "sparse_lorenz2", "bouncing_ball", "double_bouncing_ball",
    "long_bouncing_ball", "even_bouncing_ball", "boknis_eck_v1", "nascar_v2", "new_nascar",
    "new_nascar_10", "new_nascar_10_split_5", "new_nascar_10_split_10",
    "new_nascar_10_split_15", "new_nascar_10_split_20", "bee_seq_data", "har_small",
    "moseq_small", "moseq_small_standardized", "behavenet_standardized",
    "bee_seq_data_split_2", "bee_seq_data_split_4", "bee_seq_data_split_8",
    "small_behavenet_standardized", "smaller_behavenet_standardized",
    "super_small_behavenet_standardized", "bee_seq_data_full", "bee_seq_data_3",
    "brain_seq_0", "brain_seq_pca_5_0", "brain_seq_pca_10_0", "brain_seq_0_new",
    "brain_seq_1", "brain_seq_pca_5_1", "brain_seq_pca_10_1", "brain_seq_1_new",
    "brain_seq_2", "brain_seq_pca_5_2", "brain_seq_pca_10_2", "brain_seq_2_new",
    "brain_seq_3", "brain_seq_pca_5_3", "brain_seq_pca_10_3", "brain_seq_3_new",
    "brain_seq_4", "brain_seq_pca_5_4", "brain_seq_pca_10_4", "brain_seq_4_new",
}


# =============================================================================
# Datasets
# =============================================================================

class ChunkedDataset:
    """Sequences grouped into chunks of `chunks` and concatenated.

    Subclasses only load ``data_y`` (and optionally ``data_z``); the chunking,
    optional standardization, and optional subsampling stride all live here.
    """

    #: Post-fit fudge applied to the scaler variance. Historical: sklearn's
    #: transform() reads scale_, not var_, so this is inert -- kept because the
    #: NumPy reference runner does it and parity is measured against that.
    scaler_var_scale = 0.1

    def __init__(self, normalize=False, sparsity=1):
        self.normalize = normalize
        self.sparsity = sparsity
        self.scaler = None
        self.data_y = None
        self.data_z = None

    def _fit_scaler(self):
        if not self.normalize:
            return
        self.scaler = _standard_scaler().fit(np.concatenate(self.data_y))
        if self.scaler_var_scale is not None:
            self.scaler.var_ *= self.scaler_var_scale

    def _limit(self, limit):
        if limit is None:
            return
        self.data_y = self.data_y[:limit]
        self.data_z = None if self.data_z is None else self.data_z[:limit]

    def get_data(self, chunks=5):
        res = []
        for i in range(0, len(self.data_y), chunks):
            chunk = np.concatenate(self.data_y[i:(i + chunks)])[::self.sparsity]
            if self.normalize:
                chunk = self.scaler.transform(chunk)
            res.append(chunk)
        return res

    def get_labels(self, chunks=5):
        if self.data_z is None:
            return None
        res = []
        for i in range(0, len(self.data_y), chunks):
            res.append(
                np.concatenate(self.data_z[i:(i + chunks)])[::self.sparsity]
            )
        return res


class BeeDataset(ChunkedDataset):
    """Bee dance dataset (REDSDS .npz layout)."""

    def __init__(self, path="../third_party/REDSDS/data/bee.npz", normalize=False):
        super().__init__(normalize=normalize)
        npz = np.load(path)
        self.data_y = npz["y"].astype(np.float32)
        if normalize:
            assert len(np.concatenate(self.data_y).shape) == 2
        self._fit_scaler()
        self.data_z = npz["z"].astype(np.int32)


class NpDataset(ChunkedDataset):
    """Plain .npy arrays of sequences, with optional labels and subsampling."""

    def __init__(self, data_path="../Data/spikes_windowed_train.npy",
                 labels_path=None, limit=None, normalize=False, sparsity=1):
        super().__init__(normalize=normalize, sparsity=sparsity)
        self.data_y = np.load(data_path).astype(np.float32)
        self._fit_scaler()
        self.data_z = (
            None if labels_path is None else np.load(labels_path).astype(np.int32)
        )
        self._limit(limit)


class SparseNpDataset(NpDataset):
    """NpDataset that keeps every `sparsity`-th timestep."""

    def __init__(self, data_path="../Data/spikes_windowed_train.npy",
                 labels_path=None, limit=None, sparsity=5, normalize=False):
        super().__init__(
            data_path=data_path, labels_path=labels_path, limit=limit,
            normalize=normalize, sparsity=sparsity,
        )


class ScaledNpDataset(ChunkedDataset):
    """NpDataset scaled by a constant, optionally with additive noise."""

    def __init__(self, data_path="../Data/spikes_windowed_train.npy",
                 labels_path=None, limit=None, scale=1000, var=None):
        super().__init__(normalize=False)
        self.data_y = np.load(data_path).astype(np.float32) * scale
        if var is not None:
            # Historical: `var` gates the noise but does not scale it.
            self.data_y += np.random.randn(*self.data_y.shape)
        self.data_z = (
            None if labels_path is None else np.load(labels_path).astype(np.int32)
        )
        self.scale = scale
        self._limit(limit)


class HDPDataset(ChunkedDataset):
    """HDP-format .npz with yt_train/yt_test and optional loc_train/loc_test."""

    # This loader never applied the variance fudge.
    scaler_var_scale = None

    def __init__(self, data_path="../Data/new_nascar.npz", train=True, limit=None,
                 normalize=False):
        super().__init__(normalize=normalize)
        dat = np.load(data_path, allow_pickle=True)
        self.data_y = dat["yt_train"] if train else dat["yt_test"]
        self._fit_scaler()
        self.data_z = None
        if ("loc_train" in dat) and train:
            self.data_z = dat["loc_train"]
        elif ("loc_test" in dat) and not train:
            self.data_z = dat["loc_test"]
        self._limit(limit)


class SplittedHDPDataset(ChunkedDataset):
    """HDPDataset whose sequences are each split into `split` pieces."""

    def __init__(self, data_path="../Data/new_nascar.npz", train=True, limit=None,
                 normalize=False, split=5):
        super().__init__(normalize=normalize)
        dat = np.load(data_path, allow_pickle=True)
        self.data_y = dat["yt_train"] if train else dat["yt_test"]
        self._fit_scaler()
        self.data_z = None
        if "loc_train" in dat and train:
            self.data_z = dat["loc_train"]
        elif "loc_test" in dat and not train:
            self.data_z = dat["loc_test"]
        self._limit(limit)

        self.data_y = [
            piece for el in self.data_y for piece in np.array_split(el, split)
        ]
        if self.data_z is not None:
            self.data_z = [
                piece for el in self.data_z for piece in np.array_split(el, split)
            ]


def get_dataset(dataset, normalize=False):
    """Return (train_dataset, test_dataset) for a registered dataset name."""
    assert dataset in available_datasets, f"Unknown dataset {dataset}!"

    if dataset == "bee":
        return (
            BeeDataset(path="../third_party/REDSDS/data/bee.npz", normalize=normalize),
            BeeDataset(path="../third_party/REDSDS/data/bee_test.npz", normalize=normalize),
        )

    # Windowed .npy pairs that only differ by a per-dataset row limit.
    limited_windowed = {
        "spikes": 20,
        "short_spikes": 60,
        "sparse_spikes": 20,
        "poisson_cross": None,
    }
    if dataset in limited_windowed:
        limit = limited_windowed[dataset]
        return (
            NpDataset(data_path=f"../Data/{dataset}_windowed_train.npy",
                      labels_path=f"../Data/{dataset}_windowed_train_labels.npy",
                      limit=limit, normalize=normalize),
            NpDataset(data_path=f"../Data/{dataset}_windowed_val.npy",
                      labels_path=f"../Data/{dataset}_windowed_val_labels.npy",
                      limit=limit, normalize=normalize),
        )

    if dataset == "sparse_nascar":
        return (
            SparseNpDataset(data_path="../Data/super_long_nascar_windowed_train.npy",
                            labels_path="../Data/super_long_nascar_windowed_train_labels.npy",
                            sparsity=10, normalize=normalize),
            SparseNpDataset(data_path="../Data/super_long_nascar_windowed_val.npy",
                            labels_path="../Data/super_long_nascar_windowed_val_labels.npy",
                            sparsity=10, normalize=normalize),
        )

    if dataset in ["long_nascar", "long_hidden_nascar", "nascar", "hidden_nascar",
                   "super_long_nascar", "super_long_hidden_nascar", "square",
                   "sausage_nascar", "hidden_sausage_nascar", "shorter_sausage_nascar",
                   "hidden_shorter_sausage_nascar", "bouncing_ball", "long_bouncing_ball",
                   "double_bouncing_ball", "double_bouncing_ball_rp",
                   "double_bouncing_ball_1", "even_bouncing_ball", "nascar_v2"]:
        return (
            NpDataset(data_path=f"../Data/{dataset}_windowed_train.npy",
                      labels_path=f"../Data/{dataset}_windowed_train_labels.npy",
                      normalize=normalize),
            NpDataset(data_path=f"../Data/{dataset}_windowed_val.npy",
                      labels_path=f"../Data/{dataset}_windowed_val_labels.npy",
                      normalize=normalize),
        )

    if dataset in ["new_nascar", "new_nascar_10", "bee_seq_data", "har_small",
                   "moseq_small", "moseq_small_standardized", "new_nascar_10_split_5",
                   "new_nascar_10_split_10", "new_nascar_10_split_15",
                   "new_nascar_10_split_20", "behavenet_standardized",
                   "bee_seq_data_full", "bee_seq_data_3",
                   "brain_seq_0", "brain_seq_pca_5_0", "brain_seq_pca_10_0", "brain_seq_0_new",
                   "brain_seq_1", "brain_seq_pca_5_1", "brain_seq_pca_10_1", "brain_seq_1_new",
                   "brain_seq_2", "brain_seq_pca_5_2", "brain_seq_pca_10_2", "brain_seq_2_new",
                   "brain_seq_3", "brain_seq_pca_5_3", "brain_seq_pca_10_3", "brain_seq_3_new",
                   "brain_seq_4", "brain_seq_pca_5_4", "brain_seq_pca_10_4", "brain_seq_4_new"]:
        return (
            HDPDataset(data_path=f"../Data/{dataset}.npz", train=True, normalize=normalize),
            HDPDataset(data_path=f"../Data/{dataset}.npz", train=False, normalize=normalize),
        )

    behavenet_limits = {
        "small_behavenet_standardized": 20,
        "smaller_behavenet_standardized": 10,
        "super_small_behavenet_standardized": 5,
    }
    if dataset in behavenet_limits:
        limit = behavenet_limits[dataset]
        return (
            HDPDataset(data_path="../Data/behavenet_standardized.npz", train=True,
                       normalize=normalize, limit=limit),
            HDPDataset(data_path="../Data/behavenet_standardized.npz", train=False,
                       normalize=normalize, limit=limit),
        )

    if dataset.startswith("bee_seq_data_split"):
        split = int(dataset.split("_")[-1])
        return (
            SplittedHDPDataset(data_path="../Data/bee_seq_data.npz", train=True,
                               normalize=normalize, split=split),
            SplittedHDPDataset(data_path="../Data/bee_seq_data.npz", train=False,
                               normalize=normalize, split=split),
        )

    if dataset in ["lorenz", "sparse_lorenz", "lorenz2", "sparse_lorenz2", "boknis_eck_v1"]:
        return (
            NpDataset(data_path=f"../Data/{dataset}_windowed_train.npy", normalize=normalize),
            NpDataset(data_path=f"../Data/{dataset}_windowed_val.npy", normalize=normalize),
        )

    if dataset == "short_lorenz":
        return (
            SparseNpDataset(data_path="../Data/lorenz_windowed_train.npy",
                            sparsity=10, normalize=normalize),
            SparseNpDataset(data_path="../Data/lorenz_windowed_val.npy",
                            sparsity=10, normalize=normalize),
        )

    if dataset == "scaled_lorenz2":
        base = dataset[len("scaled_"):]
        return (
            ScaledNpDataset(data_path=f"../Data/{base}_windowed_train.npy", var=0.0001),
            ScaledNpDataset(data_path=f"../Data/{base}_windowed_val.npy", var=0.0001),
        )

    if dataset == "scaled_nascar":
        base = dataset[len("scaled_"):]
        return (
            ScaledNpDataset(data_path=f"../Data/{base}_windowed_train.npy",
                            labels_path=f"../Data/{base}_windowed_train_labels.npy",
                            var=0.0001),
            ScaledNpDataset(data_path=f"../Data/{base}_windowed_val.npy",
                            labels_path=f"../Data/{base}_windowed_val_labels.npy",
                            var=0.0001),
        )

    raise ValueError(f"Dataset {dataset} not fully supported yet")


# =============================================================================
# Config helpers
# =============================================================================

def str2bool(v):
    """argparse type for flags that must accept --flag=false as well as --flag."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def scalar_to_matrix(val, dim, rep):
    """Expand a scalar config value to `rep` copies of a `dim` diagonal matrix.

    ``rep == 0`` returns a single (dim, dim) matrix -- the nonswitching case,
    where the prior has no leading state axis.
    """
    if val is None or val == "None":
        return None
    v = float(val)
    if rep == 0:
        return np.eye(dim) * v
    return np.tile(np.eye(dim) * v, (rep, 1, 1))


def scalar_to_vector(val, dim, rep):
    """Expand a scalar config value to `rep` copies of a length-`dim` vector."""
    if val is None or val == "None":
        return None
    v = float(val)
    if rep == 0:
        return np.ones(dim) * v
    return np.ones((rep, dim)) * v
