import argparse
import importlib
from typing import Any

import optuna
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from torch import nn

ALGO = "ppo"


def sample_ppo_params(
    trial: optuna.Trial,
    _n_actions: int,
    _n_envs: int,
    _additional_args: dict,
) -> dict[str, Any]:
    # From 2**5=32 to 2**10=1024
    batch_size = 2 ** trial.suggest_int("batch_size_pow", 5, 10)
    # From 2**5=32 to 2**12=4096
    n_steps = 2 ** trial.suggest_int("n_steps_pow", 5, 12)
    gamma = 1 - trial.suggest_float(
        "one_minus_gamma",
        0.0001,
        0.03,
        log=True,
    )
    gae_lambda = 1 - trial.suggest_float(
        "one_minus_gae_lambda",
        0.0001,
        0.1,
        log=True,
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)

    # Display true values
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("gae_lambda", gae_lambda)

    return {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            },
            "activation_fn": nn.ReLU,
        },
    }


def optimize_hyperparams(env_id: str) -> None:
    HYPERPARAMS_SAMPLER[ALGO] = sample_ppo_params
    importlib.import_module("gymnasium_search_race")
    exp_manager = ExperimentManager(
        args=argparse.Namespace(env=env_id),
        algo=ALGO,
        env_id=env_id,
        log_folder="logs",
        n_timesteps=500_000,
        n_eval_episodes=50,
        env_kwargs={"sequential_maps": True},
        optimize_hyperparameters=True,
        storage="logs/hyperparams_tuning.log",
        study_name=env_id,
        n_trials=500,
        n_startup_trials=20,
        n_evaluations=1,
        config="hyperparams/ppo.yml",
    )
    exp_manager.setup_experiment()
    exp_manager.hyperparameters_optimization()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_search_race/SearchRaceDiscrete-v3",
        help="environment id",
    )
    args = parser.parse_args()

    optimize_hyperparams(env_id=args.env)
