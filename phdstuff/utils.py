import os
from scipy.optimize import linear_sum_assignment
import numpy as np

if os.name != 'nt':
    import resource

    def memory_limit():
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

    def get_memory():
        with open('/proc/meminfo', 'r') as mem:
            free_memory = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    free_memory += int(sline[1])
        return free_memory

def batches_generator(data, batch_size):
    def res_generator():
        while True:
            for i in range(0, data.shape[0], batch_size):
                yield data[i:(i+batch_size)]
    return res_generator

def split_data(data, labels=None):
    n = int(data.shape[0] * 0.5)
    m = int(data.shape[0] * 0.75)
    res = {}
    res["X_train"] = data[:n]
    res["X_val"] = data[n:m]
    res["X_test"] = data[m:]
    if labels is not None:
        assert data.shape[0] == labels.shape[0]
        res["y_train"] = labels[:n]
        res["y_val"] = labels[n:m]
        res["y_test"] = labels[m:]
    else:
        res["y_train"] = None
        res["y_val"] = None
        res["y_test"] = None
    return res


def log_wandb_scalar_or_array(param, name, step=None, commit=None, sync=None, plot_heatmap=True):
    import wandb
    if isinstance(param, np.ndarray) and len(param.shape) > 1:
        log_wandb_array(param, name, step=step, commit=commit, sync=sync, plot_heatmap=plot_heatmap)
        return
    wandb.log({name: param}, step=step, commit=commit, sync=sync)


def log_wandb_array(a, name, step=None, commit=None, sync=None, plot_heatmap=True):
    import wandb
    if len(a.shape) == 2 and plot_heatmap:
        x_labels = [f"s_{i}" for i in range(a.shape[1])]
        y_labels = [f"s_{i}" for i in range(a.shape[0])]
        wandb.log({f"{name}_heatmap": wandb.plots.HeatMap(x_labels, y_labels, a, show_text=True)}, step=step,
                  commit=False, sync=sync)
    # print(a.shape)
    if 0 in a.shape:
        return
    it = np.nditer(a, flags=['multi_index'])
    param_dic = {}
    for x in it:
        val = x
        key = name + "_" + "_".join([str(k) for k in it.multi_index])
        param_dic[key] = val
    wandb.log(param_dic, step=step, commit=commit, sync=sync)

def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm

def permute(Z1, Z2, K):
    try:
        res = np.zeros(Z1.shape).astype(int)
        perm = find_permutation(Z1.astype(int), Z2.astype(int))
        for i in range(K):
            res[Z1 == i] = perm[i]
    except:
        return Z1
    return res