"""Posterior summaries and beam-truncation traces recorded during Gibbs sweeps.

Producers are ``REDSLDS.beam()`` / ``REDSLDS.infer()`` (through a
``diagnostics`` dict); the NumPy runner writes ``posterior_{train,test}.npz``
from it with :func:`posterior_arrays`. Everything here is observation-only:
no RNG, no model state.
"""
import numpy as np


def run_length_encode(z):
    """(states, lengths) of the maximal constant segments of a state sequence."""
    z = np.asarray(z)
    if z.size == 0:
        return z, np.zeros(0, dtype=int)
    starts = np.flatnonzero(np.r_[True, z[1:] != z[:-1]])
    return z[starts], np.diff(np.r_[starts, z.size])


def beam_truncation_trace(switch_lls_list, right):
    """Per-sequence, per-timestep D_t^max and |D_t| from beam_forward_v2 slice masks.

    Each mask has shape (limit, K_from, K_to, right); rows 0..limit-2 hold the
    slice (0.0 = reachable, -inf = pruned), row limit-1 is never written by the
    recursion and keeps its np.zeros init, so it is excluded: every returned
    array has limit-1 entries (a length-1 sequence gives an empty one, so the
    lists stay aligned with the sequences). D_t^max is the largest surviving
    1-based duration index; 0 means no switch survives the slice at t (the
    chain continues its current segment), which is a genuine outcome.
    """
    idx = np.arange(1, right + 1)
    d_max, d_card = [], []
    for sl in switch_lls_list:
        reach = np.isfinite(sl[:-1]).any(axis=(1, 2))  # (limit-1, right)
        d_card.append(reach.sum(axis=1))
        d_max.append((reach * idx[np.newaxis, :]).max(axis=1))
    return d_max, d_card


def beam_truncation_stats(d_max, d_card, right):
    """Histograms over 0..right (index = value) of D_t^max and |D_t|, pooled over
    sequences and timesteps, for the sweep_metrics.jsonl sidecar; None when no
    slice row was written. Summary statistics are derived downstream
    (scripts/_sweep_metrics.py) so the quantile definition lives in one place.
    """
    d_max, d_card = np.concatenate(d_max), np.concatenate(d_card)
    if d_max.size == 0:
        return None
    return {
        "T_effective": int(d_max.size),
        "hist_D_max": np.bincount(d_max, minlength=right + 1).tolist(),
        "hist_D_card": np.bincount(d_card, minlength=right + 1).tolist(),
    }


def posterior_increments(Z_expanded, K, X_short, d_max_t=None):
    """One sweep's contribution to the posterior summaries: key -> list of per-sequence arrays.

    ``state_counts`` (T_i, K) one-hot states; ``x_sum`` (T_i, D) the sampled
    latents (``X_short``: no leading x_0); ``segment_length_hist`` one
    (K, T_max + 1) array of sampled segments by state and length, pooled over
    sequences; ``dmax_sum`` (T_i,) per-timestep D_t^max with NaN for the
    never-written last slice row (beam runs only).
    """
    hist = np.zeros((K, max(len(z) for z in Z_expanded) + 1), dtype=np.int64)
    for z in Z_expanded:
        states, lengths = run_length_encode(z)
        np.add.at(hist, (states, lengths), 1)
    inc = {"state_counts": [np.eye(K, dtype=np.int64)[z] for z in Z_expanded],
           "x_sum": [np.asarray(x, dtype=float) for x in X_short],
           "segment_length_hist": [hist]}
    if d_max_t is not None:
        inc["dmax_sum"] = [np.append(d, np.nan) for d in d_max_t]
    return inc


def record_posterior(diagnostics, increments):
    """Add one post-burn-in sweep's ``increments`` to ``diagnostics`` (arrays allocated on first use)."""
    for key, vals in increments.items():
        acc = diagnostics.setdefault(key, [np.zeros_like(v) for v in vals])
        for a, v in zip(acc, vals):
            a += v
    diagnostics["n_post_burnin"] = diagnostics.get("n_post_burnin", 0) + 1


def posterior_arrays(diagnostics):
    """Arrays for posterior_{split}.npz: per-sequence lists concatenated along
    time, ``*_sum`` keys divided by the number of sweeps into ``*_mean``, plus
    ``seq_lengths`` and ``n_sweeps``.
    """
    n = diagnostics["n_post_burnin"]
    out = {"n_sweeps": n, "seq_lengths": np.array([len(c) for c in diagnostics["state_counts"]])}
    for key, vals in diagnostics.items():
        if not (isinstance(vals, list) and vals and isinstance(vals[0], np.ndarray)):
            continue
        arr = np.concatenate(vals)
        if key.endswith("_sum"):
            out[key[:-len("_sum")] + "_mean"] = arr / n
        else:
            out[key] = arr
    return out
