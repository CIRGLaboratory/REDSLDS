import argparse
import json
import logging
import matplotlib
import numpy as np
import os
import pickle
import random
import resource
import shutil
import sys
import time
import traceback
from datetime import datetime
from matplotlib import pyplot as plt
from os import listdir
from os.path import join
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from phdstuff.EDSLDS_numpy.duration import TestPoisson, CategoricalDuration, LoopyPoisson, \
    RecurrentDuration, DummyDuration
from phdstuff.EDSLDS_numpy.dynamics import LinearGaussianDynamics
from phdstuff.EDSLDS_numpy.edslds import EDSLDS
from phdstuff.EDSLDS_numpy.emission import LinearGaussianEmission, NonswitchingLinearGaussianEmission
from phdstuff.EDSLDS_numpy.initial import Initial, LearnableInitial, AbstractInitial
from phdstuff.EDSLDS_numpy.plot import plot_actual_shifts, plot_observations, plot_emission, \
    plot_durations
from phdstuff.EDSLDS_numpy.redhmm import REDHMM
from phdstuff.EDSLDS_numpy.redslds import REDSLDS
from phdstuff.EDSLDS_numpy.diagnostics import posterior_arrays
from phdstuff.EDSLDS_numpy.rslds import RSLDS
from phdstuff.EDSLDS_numpy.slds import SLDS, NoBeamEDSLDS
from phdstuff.EDSLDS_numpy.sldsv2 import SLDSv2
from phdstuff.EDSLDS_numpy.transition import Transition, RecurrentTransition, LoopyTransition, HDPTransition
from phdstuff.EDSLDS_numpy.utils import invert
from phdstuff.plotting import plot_states

available_models = {"EDSLDS", "REDSLDS", "SLDS", "NOBEAMEDSLDS", "RSLDS", "REDHMM", "RARHMM", "SLDSV2", "ARHMM"}


_TIMING_SUMMARY_ORDER = (
    "status", "seed", "job_id", "started_at", "fit_started_at", "fit_finished_at",
    "predict_started_at", "predict_finished_at", "finished_at",
    "fit_wall_seconds", "predict_wall_seconds", "total_wall_seconds",
)


def _write_timing_artifacts(log_dir, timing):
    if not log_dir or not os.path.isdir(log_dir):
        return

    timing_path = os.path.join(log_dir, "timing.json")
    with open(timing_path, "w") as ofile:
        json.dump(timing, ofile, indent=2, sort_keys=True)

    training_output_path = os.path.join(log_dir, "training.output")
    with open(training_output_path, "a") as ofile:
        ofile.write("\nRun timing summary:\n")
        # headline keys first, then every other key, so nothing is dropped
        keys = list(_TIMING_SUMMARY_ORDER) + sorted(k for k in timing if k not in _TIMING_SUMMARY_ORDER)
        for key in keys:
            if key in timing and timing[key] not in (None, ""):
                ofile.write(f"{key}: {timing[key]}\n")
        ofile.write("\n")



# Dataset loaders, the dataset registry, and the scalar->prior helpers are
# shared with run_segmentation_edslds_jax.py so the two runners cannot drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _experiment_common import (  # noqa: E402
    TreeMemorySampler,
    available_datasets,
    get_dataset,
    scalar_to_matrix as _get_diagonal_matrix,
    scalar_to_vector as _get_vectors,
    seeds,
    str2bool,
)


default_config = {
    "experiment": "edslds-bee",
    "model": "EDSLDS",
    "dataset": "bee",
    "x_dim": 8,
    "obs_dim": 4,
    "discrete_dim": 3,
    "log_dir": "../results/edslds/bee/logs/{timestamp}/",
    "model_dir": "../results/edslds/bee/models/{timestamp}/",
    "plots_dir": "../results/edslds/bee/plots/{timestamp}/",
    "eps": 0.01,
    "initial_beta ": 0.01,
    "its": 100,
    "burnin": 10,
    "wandb_log": "all",
    "plot": True,
    "decay": 0.1,
    "affine_dynamics": False,
    "dump_period": 10,
    "num_workers": 10,
    "chunks": 5,
    "initial_beta": 0.01,
    "emission": "LinearGaussian",
    "initial": "exponential",
    "transition": "Categorical",
    "debug": False,
    "clean_wandb": False,
    "clean_models": False,
    "clean_logs": False,
    "clean_plots": False,
    "duration": "Poisson",
    "init_with_pca": True,
    "d_max": 50,
    "one_dim_init": False,
    "normalize": False,
    "init_iters": 100,
    "one_dim_init_scale": 1.,
    "complex_init": False,
    "in_state_path": "",
    "state_dir": "../results/edslds/bee/state/{timestamp}/",
    "extended_kalman": False,
    "nobeam": False,
    "loopy": False,
    "dyn_eps": 0.1,
    "emi_eps": 0.1,
    "tran_eps": 0.1,
    "dur_eps": 0.1,
    "adaptive_priors": False,
    "mixed_priors": False,
    "difference_priors": False,
    "emi_affine": False,
    "dyn_affine": False,
    "tran_affine": False,
    "dur_affine": False,
    "fast": False,
    "rotate": False,
    "normalize_emission": False,
    "dyn_As": None,
    "dyn_bs": None,
    "dyn_Sigmas": None,
    "dyn_mus_0": None,
    "dyn_Sigmas_0": None,
    "dyn_nu_0": None,
    "dyn_V_0": None,
    "dyn_Lambda_0": None,
    "dyn_B_0": None,
    "emi_Cs": None,
    "emi_Sigmas": None,
    "emi_mu_0": None,
    "emi_Sigma_0": None,
    "emi_nu_0": None,
    "emi_V_0": None,
    "emi_Lambda_0": None,
    "emi_B_0": None,
    "emi_b": None,
    "arhmm_init_iters": 1000,
    "dyn_cov_multiplier": 0.75,
    "emi_cov_multiplier": 0.075,
    "S0_multiplier": 0.75,
    "init_with_arhmm": True,
    "kmeans_arhmm": True,
    "freeze_dynamics": False,
    "freeze_transition": False,
    "freeze_emission": False,
    "freeze_duration": False,
    "run_suffix": "",
    "save_space": True,
    "predict": True,
    "legacy_priors": False,
    "nonswitching_duration" : False,
    "nonswitching_transition": True,
    "skip_duration_message": False,
    "skip_transition_message": False,
    "seed": 0,
    "ignore_switches": False,
    "json_path": None,
    "forward_discrete_states_sample": False,
    # frozen-parameter inference sweeps on the test split (held-out log-lik + calibration)
    "heldout_its": 50,
    "heldout_burnin": 10,
}


def _load_json_config(args):
    """JSON config over the runner defaults, then explicit CLI overrides on top.

    A flag counts as explicit when its value differs from the runner default,
    so the SLURM scripts' ``--seed``/``--run_suffix`` reach the run instead of
    being discarded (mirrors run_segmentation_edslds_jax.py).
    """
    with open(args.json_path) as ifile:
        config = {**default_config, **json.load(ifile)}
    for key, value in vars(args).items():
        if key != "json_path" and value != default_config.get(key):
            config[key] = value
    return config


def clean_wandb(wandb_id):
    try:
        run_folders = [x for x in listdir("wandb") if wandb_id in x]
        for f in run_folders:
            shutil.rmtree(join("wandb", f))
    except:
        logging.error(f"Error removing run {wandb_id}")
        logging.error(traceback.format_exc())


def _plot_predictions(path, name, Z, X_est, Y_est, Z_est, D_est, states_n, wandb_log):
    plot_dict = {}
    file_path = os.path.join(path, f"{name}_dynamics_actual_shift.png")
    plot_actual_shifts(X_est, Z_est, states_n, file_path, title=f"{name} - as")
    if wandb_log != "":
        plot_dict[f"{name}_dynamics_actual_shift"] = wandb.Image(file_path)
    file_path = os.path.join(path, f"{name}_dynamics_time.png")
    plot_observations(Z_est[:1000], X_est[:1000], fname=file_path, title=f"{name} - estimated_dynamics_time")
    if wandb_log != "":
        plot_dict[f"{name}_estimated_dynamics_time"] = wandb.Image(file_path)
    file_path = os.path.join(path, f"{name}_estimated_emissions.png")
    plot_emission(Y_est, Z_est, states_n, file_path, title=f"{name} - estimated_emission")
    if wandb_log != "":
        plot_dict[f"{name}_estimated_emissions"] = wandb.Image(file_path)
    file_path = os.path.join(path, f"{name}_estimated_emissions_time.png")
    plot_observations(Z_est[:1000], Y_est[:1000], fname=file_path, title=f"{name} - estimated_emission_time")
    if wandb_log != "":
        plot_dict[f"{name}_estimated_emissions_time"] = wandb.Image(file_path)
    file_path = os.path.join(path, f"{name}_durations.png")
    fig, ax = plot_durations(Z_est, D_est, states_n, title=f"{name} - durations")
    fig.savefig(file_path, bbox_inches='tight', dpi=300)
    if "pdf" not in file_path:
        file_path = file_path[:-3] + "pdf"
        fig.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    if wandb_log != "":
        plot_dict["durations"] = wandb.Image(file_path)
    file_path = os.path.join(path, f"{name}_states_comparison.png")
    plot_states(Z, Z_est, f"{name} states", file_path)
    if wandb_log != "":
        plot_dict[f"{name}_states_comparison"] = wandb.Image(file_path)
    if wandb_log != "":
        wandb.log(plot_dict)

def setup_parser():
    parser = argparse.ArgumentParser()
    for k, v in default_config.items():
        type_v = type(v)
        type_v = str2bool if type_v == bool else type_v
        if type_v == type(None):
            type_v = str
        parser.add_argument(f'--{k}', default=v, type=type_v)
    return parser


if __name__ == "__main__":
    wandb_id = None
    matplotlib.use("Agg")
    parser = setup_parser()
    args = parser.parse_args()
    if args.json_path:
        config = _load_json_config(args)
    elif args.wandb_log != "":
        import wandb
        name = f"{args.experiment}"
        wandb.init(project=name, config=args)
        wandb_id = wandb.run.id
        config = wandb.config
    else:
        config = vars(args)
    name = config["experiment"]
    seed = config["seed"] % 10
    random.seed(seed)
    np.random.seed(seed)
    if config["in_state_path"]:
        with open(config["in_state_path"], "rb") as ifile:
            st0 = pickle.load(ifile)
        np.random.set_state(st0)
    st0 = np.random.get_state()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if len(config["run_suffix"]) > 0:
        timestamp = timestamp + "_" + config["run_suffix"]
    log_dir = config["log_dir"].format(
        timestamp=timestamp)
    model_dir = config["model_dir"].format(
        timestamp=timestamp)
    plots_dir = config["plots_dir"].format(
        timestamp=timestamp)
    timing = {
        "status": "running",
        "seed": seed,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "started_at": datetime.now().isoformat(),
    }
    run_start_perf = time.perf_counter()
    mem_sampler = TreeMemorySampler(checkpoint_path=join(log_dir, "memory_peak.json"))
    mem_sampler.start()
    train_diag = {}  # filled by REDSLDS-family beam(diagnostics=...); empty for HMM runs
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    state_dir = os.path.join(model_dir, "state")
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    with open(join(state_dir, "state.pickle"), "wb") as ofile:
        pickle.dump(st0, ofile)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    assert config["model"].upper() in available_models, f"Model not known: {config['model']}"
    try:
        with open(os.path.join(log_dir, "config.json"), "w") as fp:
            json.dump(dict(config), fp)
        chunks = config["chunks"]
        train_dataset, test_dataset = get_dataset(config["dataset"], config["normalize"])
        Y = train_dataset.get_data(chunks=chunks)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        # DATA
        logging.basicConfig(
            filename=os.path.join(log_dir, f"{name}.log"),
            filemode="w",
            level=logging.DEBUG
        )
        model_type = config["model"].upper()
        eps = config["eps"]
        K = config["discrete_dim"]
        x_dim = config["x_dim"]
        obs_dim = config["obs_dim"]
        if "HMM" in model_type:
            x_dim = obs_dim
        transition_type = config["transition"]
        d_max = config["d_max"]
        if transition_type == "Categorical":
            transition = Transition(
                K=K, D=x_dim
            )
        elif (transition_type == "Loopy"):
            transition = LoopyTransition(
                K=K,
                D=x_dim
            )
        elif (transition_type == "Recurrent") or (transition_type == "RecurrentTransition"):
            transition = RecurrentTransition(K=K, D=x_dim, loopy=config["loopy"], eps=config["tran_eps"],
                                             affine=config["tran_affine"], nonswitching=config["nonswitching_transition"])
        elif (transition_type == "HDPTransition"):
            transition = HDPTransition(L = K, D=x_dim)
        else:
            raise Exception("Unknown trainsition")
        reps = K if config["emission"] == "LinearGaussian" else 0
        Cs = _get_diagonal_matrix(config["emi_Cs"], obs_dim, reps)
        Sigmas = _get_diagonal_matrix(config["emi_Sigmas"], obs_dim, reps)
        mu_0 = _get_vectors(config["emi_mu_0"], obs_dim, reps)
        b = _get_vectors(config["emi_b"], obs_dim, reps)
        Sigma_0 = _get_diagonal_matrix(config["emi_Sigma_0"], obs_dim, reps)
        nu_0_n = 1 if reps == 0 else reps
        nu_0 = _get_vectors(config["emi_mu_0"], nu_0_n, 0)
        V_0 = _get_diagonal_matrix(config["emi_V_0"], obs_dim, reps)
        temp_dim = x_dim if not config["emi_affine"] else x_dim + 1
        Lambda_0 = _get_diagonal_matrix(config["emi_Lambda_0"], temp_dim, reps)
        B_0 = _get_diagonal_matrix(config["emi_B_0"], obs_dim, reps)
        eps = config["emi_eps"]
        if config["adaptive_priors"]:
            if config["legacy_priors"]:
                Y_ = np.concatenate(Y)
                if obs_dim != x_dim:
                    pca = PCA(n_components=x_dim)
                    Y_ = pca.fit_transform(Y_)
                if config["difference_priors"]:
                    Y_ = Y_[1:] - Y_[:-1]
                    # ic(Y_.shape)
                S0 = config["S0_multiplier"] * np.cov(Y_.T)
                emi_cov = config["emi_cov_multiplier"] * S0
                if reps > 0:
                    Lambda_0 = np.tile(invert(emi_cov)[np.newaxis], (reps, 1, 1))
                else:
                    Lambda_0 = invert(emi_cov)
            else:
                Y_ = np.concatenate(Y)
                if config["difference_priors"]:
                    Y_ = Y_[1:] - Y_[:-1]
                    # ic(Y_.shape)
                S0 = config["S0_multiplier"] * np.cov(Y_.T)
                emi_cov = config["emi_cov_multiplier"] * S0
                if reps > 0:
                    V_0 = np.tile(emi_cov[np.newaxis], (reps, 1, 1))
                else:
                    V_0 = emi_cov
        if config["emission"] == "LinearGaussian":
            emission = LinearGaussianEmission(Cs=Cs, Sigmas=Sigmas, mu_0=mu_0, Sigma_0=Sigma_0, nu_0=nu_0, V_0=V_0,
                                              Lambda_0=Lambda_0, B_0=B_0,
                                              obs_dim=obs_dim, x_dim=x_dim, K=K, eps=eps)
        else:
            emission = NonswitchingLinearGaussianEmission(K=K, C=Cs, Sigma=Sigmas, mu_0=mu_0, Sigma_0=Sigma_0,
                                                          nu_0=nu_0, V_0=V_0, Lambda_0=Lambda_0, B_0=B_0,
                                                          eps=config["emi_eps"], obs_dim=obs_dim, x_dim=x_dim,
                                                          updatable=True, bs=b, affine=config["emi_affine"])

        As = _get_diagonal_matrix(config["dyn_As"], x_dim, K)
        bs = _get_vectors(config["dyn_bs"], x_dim, K)
        Sigmas = _get_diagonal_matrix(config["dyn_Sigmas"], x_dim, K)
        mus_0 = _get_vectors(config["dyn_mus_0"], x_dim, K)
        Sigmas_0 = _get_diagonal_matrix(config["dyn_Sigmas_0"], x_dim, K)
        nu_0 = _get_vectors(config["dyn_nu_0"], x_dim, K)
        V_0 = _get_diagonal_matrix(config["dyn_V_0"], x_dim, K)
        Lambda_0 = _get_diagonal_matrix(config["dyn_Lambda_0"], x_dim, K)
        B_0 = _get_diagonal_matrix(config["dyn_B_0"], x_dim, K)
        Lambda_0_init = None
        V_0_init = None
        if config["adaptive_priors"] or config["mixed_priors"]:
            if config["legacy_priors"]:
                if not "HMM" in model_type:
                    dyn_cov = 0.75 * S0
                else:
                    dyn_cov = S0
                if config["adaptive_priors"]:
                    Lambda_0 = np.tile(invert(dyn_cov)[np.newaxis], (K, 1, 1))
                else:
                    Lambda_0_init = np.tile(invert(dyn_cov)[np.newaxis], (K, 1, 1))
            else:
                Y_ = np.concatenate(Y)
                if obs_dim != x_dim:
                    pca = PCA(n_components=x_dim)
                    Y_ = pca.fit_transform(Y_)
                if config["difference_priors"]:
                    Y_ = Y_[1:] - Y_[:-1]
                    # ic(Y_.shape)
                S0 = config["S0_multiplier"] * np.cov(Y_.T)
                if not "HMM" in model_type:
                    dyn_cov = config["dyn_cov_multiplier"] * S0
                else:
                    dyn_cov = S0
                if config["adaptive_priors"]:
                    V_0 = np.tile(dyn_cov[np.newaxis], (K, 1, 1))
                else:
                    V_0_init = np.tile(dyn_cov[np.newaxis], (K, 1, 1))
        dynamics = LinearGaussianDynamics(As=As, bs=bs, Sigmas=Sigmas, mus_0=mus_0, Sigmas_0=Sigmas_0, nu_0=nu_0,
                                          V_0=V_0, Lambda_0=Lambda_0, B_0=B_0, K=K, x_dim=x_dim, eps=config["dyn_eps"],
                                          affine=config["dyn_affine"])
        if (config["duration"] == "RecurrentDuration") or (config["duration"] == "Recurrent"):
            D = RecurrentDuration(K=K, D=x_dim, D_max=d_max, eps=config["dur_eps"], affine=config["dur_affine"], nonswitching=config["nonswitching_duration"])
        elif config["duration"] == "Poisson":
            D = TestPoisson(
                mu=np.ones(K),
                support_step=20,
                D=x_dim
            )
        elif config["duration"] == "LoopyPoisson":
            D = LoopyPoisson(
                mu=np.ones(K),
                support_step=20,
                D=x_dim
            )
        elif config["duration"] == "Categorical" or config["duration"] == "CategoricalDuration":
            D = CategoricalDuration(K, d_max, D=x_dim)
        elif (config["duration"] == "Dummy") or (config["duration"] == "DummyDuration") or (model_type == "RARHMM"):
            D = DummyDuration(K, obs_dim)
        else:
            raise Exception("Unknown duration")
        assert config["initial"] in ["exponential", "learnable"]
        pi: AbstractInitial
        if config["initial"] == "exponential":
            initial_beta = config["initial_beta"]
            pi = Initial(K=K, beta=initial_beta)
        elif config["initial"] == "learnable":
            pi = LearnableInitial(K=K, dur_dist=D, dynamics_dist=dynamics)
        else:
            raise Exception("Unknown initial")

        if model_type == "EDSLDS":
            m = EDSLDS(pi, transition, emission, dynamics, D)
        elif model_type == "REDSLDS":
            m = REDSLDS(pi, transition, emission, dynamics, D, extended_kalman=config["extended_kalman"],
                        nobeam=config["nobeam"], ignore_switches=config["ignore_switches"], forward_discrete_states_sample=config["forward_discrete_states_sample"])
        elif model_type == "SLDS":
            m = SLDS(K, emission, dynamics)
        elif model_type == "SLDSV2":
            m = SLDSv2(K, emission, dynamics, extended_kalman=config["extended_kalman"], nobeam=config["nobeam"], forward_discrete_states_sample=config["forward_discrete_states_sample"])
        elif model_type == "RSLDS":
            m = RSLDS(K, emission, dynamics, extended_kalman=config["extended_kalman"], nobeam=config["nobeam"],
                      affine=config["tran_affine"], eps=config["tran_eps"], nonswitching=config["nonswitching_transition"], forward_discrete_states_sample=config["forward_discrete_states_sample"])
        elif model_type == "NOBEAMEDSLDS":
            m = NoBeamEDSLDS(pi, transition, emission, dynamics, D)
        elif model_type == "REDHMM":
            m = REDHMM(initial=pi, transition=transition, duration=D, dynamics=dynamics, nobeam=config["nobeam"], forward_discrete_states_sample=config["forward_discrete_states_sample"])

        elif model_type == "RARHMM":
            D = DummyDuration(K, obs_dim)
            m = REDHMM(initial=pi, transition=transition, duration=D, dynamics=dynamics, nobeam=config["nobeam"], forward_discrete_states_sample=config["forward_discrete_states_sample"])

        else:
            raise Exception("Unknown model")

        actual_Z = train_dataset.get_labels(chunks=chunks)
        if "HMM" in model_type:
            timing["fit_started_at"] = datetime.now().isoformat()
            fit_start_perf = time.perf_counter()
            Z, _ = m.beam(Y,
                          its=config["its"],
                          burnin=config["burnin"],
                          name=name,
                          wandb_log=config["wandb_log"],
                          decay=config["decay"],
                          plot=config["plot"],
                          plot_folder=plots_dir,
                          dump_period=config["dump_period"],
                          dump_path=model_dir,
                          num_of_workers=config["num_workers"],
                          actual_Z=actual_Z,
                          double_sample=config["double_sample"],
                          init_with_kmeans=config["kmeans_arhmm"],
                          init_iters=config["init_iters"],
                          save_space=config["save_space"])
            timing["fit_finished_at"] = datetime.now().isoformat()
            timing["fit_wall_seconds"] = time.perf_counter() - fit_start_perf
            m.dump(name + "_final", model_dir)
            with open(os.path.join(model_dir, "Z.pickle"), "wb") as ofile:
                pickle.dump(Z, ofile)
            if config["predict"]:
                timing["predict_started_at"] = datetime.now().isoformat()
                predict_start_perf = time.perf_counter()
                log_pred = logging.getLogger('prediction')
                Z = Z[-1]
                Y = Y[-1]
                Z_test = test_dataset.get_labels(chunks=chunks)
                Y_test = test_dataset.get_data(chunks=chunks)
                if config["debug"]:
                    if Z_test is not None:
                        Z_test = np.concatenate(Z_test)[:15]
                    Y_test = np.concatenate(Y_test)[:15]
                else:
                    if Z_test is not None:
                        Z_test = np.concatenate(Z_test)
                    Y_test = np.concatenate(Y_test)
                T = len(Y_test)
                Z_pred, Y_pred, D_pred = m.sim(T, init=(Z[-1][0], Z[-1][1], Y[-1]))
                with open(os.path.join(model_dir, "Z_pred.pickle"), "wb") as ofile:
                    pickle.dump(Z_pred, ofile)
                with open(os.path.join(model_dir, "Y_pred.pickle"), "wb") as ofile:
                    pickle.dump(Y_pred, ofile)
                with open(os.path.join(model_dir, "D_pred.pickle"), "wb") as ofile:
                    pickle.dump(D_pred, ofile)
                res = {}
                res["sampled_val_rmse"] = float(np.sqrt(mean_squared_error(Y_test, Y_pred)))
                if Z_test is not None:
                    res["sampled_val_accuracy_score"] = accuracy_score(Z_test, Z_pred)
                    res["sampled_val_confusion_matrix"] = confusion_matrix(Z_test, Z_pred)
                used_ks = set()
                for k, v in res.items():
                    log_pred.debug(f"{k}: {v}")
                    used_ks.add(k)

                # wandb.log(res)
                Z_pred = np.array(Z_pred)
                D_pred = np.array(D_pred)
                Y_pred = np.stack(Y_pred)
                _plot_predictions(plots_dir, f"{name}_long_predicted", Z=Z_test, X_est=X_pred, Y_est=Y_pred, Z_est=Z_pred,
                                  D_est=D_pred, states_n=K, wandb_log=config["wandb_log"])
                # Z_pred = []
                # Y_pred = []
                # D_pred = []
                # for t in range(T):
                #     Y_cur = np.concatenate([Y, Y_test[:t]])
                #     Z_cur = m.infer([Y_cur], its=10, decay=0.01, cache=cache)
                #     Z_cur = Z_cur[0]
                #     _Z_pred, _Y_pred, _D_pred = m.sim(1, init=(Z_cur[-1][0], Z_cur[-1][1], X_cur[-1]))
                #     Z_pred.append(scipy.stats.mode(_Z_pred, axis=0))
                #     Y_pred.append(np.mean(_Y_pred, axis=0))
                #     D_pred.append(_D_pred[-1])
                # Z_pred = np.concatenate(Z_pred)
                # Y_pred = np.concatenate(Y_pred)
                # D_pred = np.concatenate(D_pred)
                # res["short_sampled_val_rmse"] = mean_squared_error(Y_test, Y_pred, squared=False)
                # if Z_test is not None:
                #     res["short_sampled_val_accuracy_score"] = accuracy_score(Z_test, Z_pred)
                #     res["short_sampled_val_confusion_matrix"] = confusion_matrix(Z_test, Z_pred)
                # _plot_predictions(plots_dir, f"{name}_short_predicted", Z=Z_test, X_est=Y_pred, Y_est=Y_pred, Z_est=Z_pred[0],
                #                   D_est=D_pred, states_n=K, wandb_log=config["wandb_log"])
                timing["predict_finished_at"] = datetime.now().isoformat()
                timing["predict_wall_seconds"] = time.perf_counter() - predict_start_perf
        else:
            timing["fit_started_at"] = datetime.now().isoformat()
            fit_start_perf = time.perf_counter()
            # Only the REDSLDS family accepts metrics_file; EDSLDS/SLDS override beam() without it.
            metrics_kwargs = ({"metrics_file": join(log_dir, "sweep_metrics.jsonl"), "diagnostics": train_diag}
                              if isinstance(m, REDSLDS) else {})
            Z, X = m.beam(
                Y, its=config["its"], burnin=config["burnin"], name=name,
                online=True, sample_U=True, wandb_log=config["wandb_log"], plot=config["plot"], decay=config["decay"],
                plot_folder=plots_dir, dump_period=config["dump_period"], dump_path=model_dir,
                num_of_workers=config["num_workers"], actual_Z=actual_Z,
                init_with_pca=config["init_with_pca"],
                init_iters=config["init_iters"],
                fast=config["fast"],
                rotate=config["rotate"],
                normalize_emission=config["normalize_emission"],
                arhmm_iters=config["arhmm_init_iters"],
                init_arhmm_with_kmeans=config["kmeans_arhmm"],
                init_with_arhmm=config["init_with_arhmm"],
                freeze_dynamics=config["freeze_dynamics"],
                freeze_transition=config["freeze_transition"],
                freeze_duration=config["freeze_duration"],
                freeze_emission=config["freeze_emission"],
                log_file=join(log_dir, "training.output"),
                **metrics_kwargs,
                save_space=config["save_space"],
                skip_duration_message = config["skip_duration_message"],
                skip_transition_message = config["skip_transition_message"],
                dyn_Lambda_0_init = Lambda_0_init,
                dyn_V_0_init = V_0_init
            )
            timing["fit_finished_at"] = datetime.now().isoformat()
            timing["fit_wall_seconds"] = time.perf_counter() - fit_start_perf
            m.dump(name + "_final", model_dir)
            with open(os.path.join(model_dir, "Z.pickle"), "wb") as ofile:
                pickle.dump(Z, ofile)
            with open(os.path.join(model_dir, "X.pickle"), "wb") as ofile:
                pickle.dump(X, ofile)
            if config["predict"]:
                timing["predict_started_at"] = datetime.now().isoformat()
                predict_start_perf = time.perf_counter()
                log_pred = logging.getLogger('prediction')
                Z = Z[-1]
                X = X[-1]
                Y = Y[-1]
                Z_test = test_dataset.get_labels(chunks=chunks)
                Y_test = test_dataset.get_data(chunks=chunks)
                if config["debug"]:
                    if Z_test is not None:
                        Z_test = np.concatenate(Z_test)[:15]
                    Y_test = np.concatenate(Y_test)[:15]
                else:
                    if Z_test is not None:
                        Z_test = np.concatenate(Z_test)
                    Y_test = np.concatenate(Y_test)
                T = len(Y_test)
                assert (T == len(Z_test)), (T, len(Z_test))
                Z_pred, X_pred, Y_pred, D_pred = m.sim(T, init=(Z[-1][0], Z[-1][1], X[-1]))
                with open(os.path.join(model_dir, "Z_pred.pickle"), "wb") as ofile:
                    pickle.dump(Z_pred, ofile)
                with open(os.path.join(model_dir, "X_pred.pickle"), "wb") as ofile:
                    pickle.dump(X_pred, ofile)
                with open(os.path.join(model_dir, "Y_pred.pickle"), "wb") as ofile:
                    pickle.dump(Y_pred, ofile)
                with open(os.path.join(model_dir, "D_pred.pickle"), "wb") as ofile:
                    pickle.dump(D_pred, ofile)
                res = {}
                res["sampled_val_rmse"] = float(np.sqrt(mean_squared_error(Y_test, Y_pred)))
                if Z_test is not None:
                    res["sampled_val_accuracy_score"] = 0
                    for Z_ in Z_pred:
                        res["sampled_val_accuracy_score"] += accuracy_score(Z_test, Z_)
                    res["sampled_val_accuracy_score"] /= len(Z_pred)
                    res["sampled_val_confusion_matrix"] = confusion_matrix(Z_test, Z_pred[0])
                used_ks = set()
                for k, v in res.items():
                    log_pred.debug(f"{k}: {v}")
                    used_ks.add(k)
                # wandb.log(res)
                Z_pred = np.array(Z_pred)
                D_pred = np.array(D_pred)
                X_pred = np.stack(X_pred)
                Y_pred = np.stack(Y_pred)
                _plot_predictions(plots_dir, f"{name}_long_predicted", Z=Z_test, X_est=X_pred, Y_est=Y_pred, Z_est=Z_pred[0],
                                  D_est=D_pred[0], states_n=K, wandb_log=config["wandb_log"])
                # Z_pred = []
                # Y_pred = []
                # X_pred = []
                # D_pred = []
                # for t in range(T):
                #     Y_cur = np.concatenate([Y, Y_test[:t]])
                #     Z_cur, X_cur = m.infer([Y_cur], its=10, decay=0.01)
                #     Z_cur = Z_cur[0]
                #     X_cur = X_cur[0]
                #     _Z_pred, _X_pred, _Y_pred, _D_pred = m.sim(1, init=(Z_cur[-1][0], Z_cur[-1][1], X_cur[-1]))
                #     Z_pred.append(scipy.stats.mode(_Z_pred, axis=0))
                #     Y_pred.append(np.mean(_Y_pred, axis=0))
                #     X_pred.append(np.mean(_X_pred, axis=0))
                #     D_pred.append(_D_pred[-1])
                # Z_pred = np.concatenate(Z_pred)
                # Y_pred = np.concatenate(Y_pred)
                # D_pred = np.concatenate(D_pred)
                # X_pred = np.concatenate(X_pred)
                # with open(os.path.join(model_dir, "Z_short_pred.pickle"), "wb") as ofile:
                #     pickle.dump(Z_pred, ofile)
                # with open(os.path.join(model_dir, "X_short_pred.pickle"), "wb") as ofile:
                #     pickle.dump(X_pred, ofile)
                # with open(os.path.join(model_dir, "Y_short_pred.pickle"), "wb") as ofile:
                #     pickle.dump(Y_pred, ofile)
                # with open(os.path.join(model_dir, "D_short_pred.pickle"), "wb") as ofile:
                #     pickle.dump(D_pred, ofile)
                # res["short_sampled_val_rmse"] = mean_squared_error(Y_test, Y_pred, squared=False)
                # if Z_test is not None:
                #     res["short_sampled_val_accuracy_score"] = 0
                #     accuracy_score(Z_test, Z_pred)
                #     for Z_ in Z_pred:
                #         res["short_sampled_val_accuracy_score"] += accuracy_score(Z_test, Z_)
                #     res["short_sampled_val_accuracy_score"] /= len(Z_pred)
                #     res["short_short_sampled_val_confusion_matrix"] = confusion_matrix(Z_test, Z_pred[0])
                # for k, v in res.items():
                #     if k in used_ks:
                #         continue
                #     log_pred.debug(f"{k}: {v}")
                #     used_ks.add(k)
                # _plot_predictions(plots_dir, f"{name}_short_predicted", Z=Z_test, X_est=X_pred, Y_est=Y_pred, Z_est=Z_pred,
                #                   D_est=D_pred, states_n=K, wandb_log=config["wandb_log"])
                timing["predict_finished_at"] = datetime.now().isoformat()
                timing["predict_wall_seconds"] = time.perf_counter() - predict_start_perf
        if train_diag.get("state_counts") is not None and actual_Z is not None:
            np.savez(join(log_dir, "posterior_train.npz"), labels=np.concatenate(actual_Z),
                     **posterior_arrays(train_diag))
        if config["predict"] and isinstance(m, REDSLDS):
            # held-out log-likelihood and state marginals: frozen-parameter inference on the test split
            held = {}
            heldout_start = time.perf_counter()
            try:
                m.infer([Y_test], its=config["heldout_its"], burnin=config["heldout_burnin"],
                        decay=config["decay"], num_of_workers=config["num_workers"], fast=config["fast"],
                        skip_duration_message=config["skip_duration_message"],
                        skip_transition_message=config["skip_transition_message"], diagnostics=held)
                b = config["heldout_burnin"]
                with open(join(log_dir, "heldout_loglikelihood_per_sweep.json"), "w") as fp:
                    json.dump(held["loglikelihoods"], fp)
                timing.update(heldout_status="success", heldout_wall_seconds=time.perf_counter() - heldout_start,
                              heldout_its=config["heldout_its"], heldout_burnin=b,
                              heldout_loglikelihood_mean=float(np.mean(held["loglikelihoods"][b:])))
                if Z_test is not None and held.get("state_counts") is not None:
                    np.savez(join(log_dir, "posterior_test.npz"), labels=np.asarray(Z_test),
                             **posterior_arrays(held))
            except Exception as e:
                timing.update(heldout_status="failed", heldout_error=f"{type(e).__name__}: {e}",
                              heldout_traceback=traceback.format_exc()[-3000:])
        timing["status"] = "success"
    except Exception as e:
        timing["status"] = "failed"
        timing["error_type"] = type(e).__name__
        timing["error_message"] = str(e)
        raise
    finally:
        timing["finished_at"] = datetime.now().isoformat()
        timing["total_wall_seconds"] = time.perf_counter() - run_start_perf
        mem_sampler.stop()
        mem_sampler.join(timeout=5.0)
        timing["peak_tree_pss_bytes"] = mem_sampler.peak_pss_bytes
        timing["peak_tree_rss_bytes"] = mem_sampler.peak_rss_bytes
        timing["mem_samples"] = mem_sampler.samples
        try:
            timing["ru_maxrss_self_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            timing["ru_maxrss_children_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        except Exception:
            pass
        _write_timing_artifacts(log_dir, timing)
        if wandb_id is not None and config["clean_wandb"]:
            clean_wandb(wandb_id)
        if config["clean_plots"]:
            shutil.rmtree(plots_dir)
        if config["clean_logs"]:
            shutil.rmtree(log_dir)
        if config["clean_models"]:
            shutil.rmtree(model_dir)
