# REDSLDS

NumPy implementation of the Recurrent Explicit Duration Switching Linear
Dynamical System (REDSLDS) and of the related switching models that the paper
compares against.

> Mikołaj Słupiński. *Bayesian Inference in Recurrent Explicit Duration
> Switching Linear Dynamical Systems.* Proceedings of the 28th International
> Conference on Artificial Intelligence and Statistics (AISTATS), PMLR
> 258:100–108, 2025. https://proceedings.mlr.press/v258/slupinski25a.html

Inference is blocked Gibbs sampling. The sampler draws the discrete states and
the durations with beam (slice) sampling, the continuous states with a Kalman
smoother, and the recurrent transition and duration weights with Pólya-gamma
augmentation.

## Installation

Python 3.10 or later is required. The code was tested with Python 3.12.

```bash
git clone git@github.com:CIRGLaboratory/REDSLDS.git
cd REDSLDS
pip install -e .
```

The package keeps the import path of the original research code:

```python
from phdstuff.EDSLDS_numpy.redslds import REDSLDS
```

Numba is optional at run time. When it is not installed, the kernels in
`numba_kernels.py` fall back to NumPy.

The code was tested with numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0,
numba 0.64.0, polyagamma 2.0.2, pathos 0.3.5, matplotlib 3.10.8, seaborn 0.13.2
and pandas 3.0.1. scikit-learn 1.9 changed the signature of
`DecisionBoundaryDisplay`, which the decision boundary plots use, so
`pyproject.toml` pins scikit-learn below 1.9.

## Repository layout

| Path | Content |
|------|---------|
| `phdstuff/EDSLDS_numpy/redslds.py` | `REDSLDS`: recurrent transition, recurrent explicit duration, beam sampling, Kalman messages. Base class of the SLDS family. |
| `phdstuff/EDSLDS_numpy/edslds.py` | `EDSLDS`: explicit-duration SLDS with a non-recurrent transition. |
| `phdstuff/EDSLDS_numpy/slds.py`, `rslds.py`, `sldsv2.py` | `SLDS`, `NoBeamEDSLDS`, `RSLDS`, `SLDSv2`: baselines built on the same sampler. |
| `phdstuff/EDSLDS_numpy/redhmm.py`, `edhmm.py` | `REDHMM`, `EDHMM`: models without a continuous latent state. |
| `phdstuff/EDSLDS_numpy/duration.py` | Duration distributions: `RecurrentDuration`, `TestPoisson`, `LoopyPoisson`, `CategoricalDuration`, `DummyDuration`. |
| `phdstuff/EDSLDS_numpy/transition.py` | Transition distributions: `Transition`, `LoopyTransition`, `RecurrentTransition`, `HDPTransition`. |
| `phdstuff/EDSLDS_numpy/emission.py`, `dynamics.py`, `initial.py` | Linear Gaussian emissions and dynamics, initial state distributions. |
| `phdstuff/EDSLDS_numpy/numba_kernels.py` | Optional Numba kernels for the Kalman passes, batched Gaussian likelihoods and the stick-breaking transition and duration probabilities. |
| `phdstuff/EDSLDS_numpy/diagnostics.py`, `plot.py`, `utils.py` | Posterior summaries, plots, worker pool and linear algebra helpers. |
| `phdstuff/utils.py`, `phdstuff/plotting.py` | Helpers shared by the models: state permutation, Weights & Biases logging, state plots. |
| `scripts/run_segmentation_edslds.py` | Experiment runner. Fits a model, segments the held-out split, and writes logs, model dumps and plots. |
| `scripts/_experiment_common.py` | Dataset registry and loaders, config helpers. |

## Data

The datasets are not part of this repository. The loaders in
`scripts/_experiment_common.py` read them relative to the `scripts/` directory:

- `bee` reads `third_party/REDSDS/data/bee.npz` and `bee_test.npz`. Create
  these files with the preprocessing script of the RED-SDS code release
  (Ansari et al., *Deep Explicit Duration Switching Models for Time Series*,
  NeurIPS 2021, https://arxiv.org/abs/2110.13878).
- All other datasets are `.npy` or `.npz` files in `Data/`. See `get_dataset()`
  for the file names and the expected arrays.

## Running an experiment

Run the script from the `scripts/` directory. Every key of `default_config` in
the runner is a command line flag. Example: REDSLDS on the NASCAR dataset.

```bash
cd scripts
python run_segmentation_edslds.py \
    --experiment redslds-nascar --model REDSLDS --dataset nascar_v2 \
    --x_dim 2 --obs_dim 10 --discrete_dim 4 \
    --transition Recurrent --duration RecurrentDuration \
    --emission NonswitchingLinearGaussianEmission --initial learnable \
    --loopy True --extended_kalman True --init_with_pca True \
    --dyn_eps 1.0 --emi_eps 1.0 --tran_eps 4.0 --dur_eps 4.0 --initial_beta 0.1 \
    --chunks 20000 --its 1000 --burnin 200 --init_iters 20 --num_workers 10 \
    --wandb_log ""
```

By default (`--wandb_log all`) the runner logs to Weights & Biases and uses the
value of `--experiment` as the project name. Set `--wandb_log ""` to run without
Weights & Biases. You can also pass a JSON file with the configuration through
`--json_path`; explicit command line flags override the values in the file.

The runner writes to the directories given by `--log_dir`, `--model_dir` and
`--plots_dir`. The `{timestamp}` placeholder in these paths is replaced by the
start time of the run. A run produces:

- `logs/<timestamp>/`: the log file, `config.json`, `timing.json`,
  `training.output`, and the posterior summaries `posterior_train.npz` and
  `posterior_test.npz`.
- `models/<timestamp>/`: parameter dumps of each component and the sampled
  state sequences (`Z.pickle`, `X.pickle`, `Z_pred.pickle`, ...).
- `plots/<timestamp>/`: segmentation, phase portrait, emission and duration
  plots.

### Options

| Flag | Values |
|------|--------|
| `--model` | `REDSLDS`, `RSLDS`, `SLDSV2`, `REDHMM`, `RARHMM`, `EDSLDS`, `NOBEAMEDSLDS`, `SLDS` |
| `--transition` | `Categorical`, `Loopy`, `Recurrent`, `HDPTransition` |
| `--duration` | `RecurrentDuration`, `Poisson`, `LoopyPoisson`, `Categorical`, `Dummy` |
| `--emission` | `LinearGaussian` (one emission per state) or `NonswitchingLinearGaussianEmission` (one shared emission) |
| `--initial` | `exponential`, `learnable` |

`RSLDS`, `SLDSV2` and `SLDS` build their own transition and duration
distributions and ignore `--transition` and `--duration`. Models with `HMM` in
the name have no continuous latent state and use `--obs_dim` for `x_dim`.

### Known limitations

- `EDSLDS`, `NOBEAMEDSLDS` and `SLDS` use the older `EDSLDS.beam()` interface,
  which does not accept the runner's `rotate` argument. Running them through
  `run_segmentation_edslds.py` stops with `TypeError: EDSLDS.beam() got an
  unexpected keyword argument 'rotate'`.
- The HMM branch of the runner (`REDHMM`, `RARHMM`) reads a `double_sample`
  key that `default_config` does not define and stops with
  `KeyError: 'double_sample'`. The `REDHMM` class itself can be used directly.

## Citation

```bibtex
@inproceedings{slupinski_bayesian_2025,
  title     = {Bayesian Inference in Recurrent Explicit Duration Switching Linear Dynamical Systems},
  author    = {S{\l}upi{\'n}ski, Miko{\l}aj},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {258},
  pages     = {100--108},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v258/slupinski25a.html}
}
```

## License

MIT. See `LICENSE`.
