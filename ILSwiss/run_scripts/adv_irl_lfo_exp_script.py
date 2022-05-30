import yaml
import argparse
import numpy as np
import os
import sys
import inspect
import gym
import pickle
import random
import joblib

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env, get_envs

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianLfOPolicy
from rlkit.torch.algorithms.sac.sac_lfo import SoftActorCritic
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc
from rlkit.torch.algorithms.adv_irl.adv_irl_lfo import AdvIRL_LfO
from rlkit.data_management.replay_buffer.unified_buffer import UnifiedReplayBuffer
from rlkit.data_management.replay_buffer.dpo_buffer import DPOUnifiedReplayBuffer
from rlkit.envs.wrappers import ProxyEnv, NormalizedBoxActEnv, ObsScaledEnv, EPS
from rlkit.samplers import DPOPathSampler

import torch


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    demos_path = listings[variant["expert_name"]]["file_paths"][0]
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    if variant["traj_num"] > 0:
        traj_list = random.sample(traj_list, variant["traj_num"])

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print(
        "\n\nEnv: {}: {}".format(env_specs["env_creator"], env_specs["scenario_name"])
    )
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    assert variant["adv_irl_params"]["state_only"]

    expert_replay_buffer = UnifiedReplayBuffer(
        max_replay_buffer_size=variant["adv_irl_params"].get(
            "expert_buffer_size", variant["adv_irl_params"]["replay_buffer_size"]
        ),
        env=env,
        random_seed=np.random.randint(10000),
    )

    if "expert_buffer_size" in variant["adv_irl_params"]:
        variant["adv_irl_params"].pop("expert_buffer_size")

    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    if variant.get("share_policy", True):
        policy_mapping_dict = dict(
            zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
        )
    else:
        policy_mapping_dict = dict(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_trainer_n = {}
    policy_n = {}
    disc_model_n = {}

    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    if "spectral_norm_inv" not in variant["adv_irl_params"]:
        variant["adv_irl_params"]["spectral_norm_inv"] = False

    if "sample_num" not in variant["adv_irl_params"]:
        variant["adv_irl_params"]["sample_num"] = 1

    if "inv_noise" not in variant["adv_irl_params"]:
        variant["adv_irl_params"]["inv_noise"] = False

    if "sp_net_size" not in variant["adv_irl_params"]:
        variant["sp_net_size"] = net_size

    if "inv_net_size" not in variant["adv_irl_params"]:
        variant["inv_net_size"] = net_size

    if "sp_num_hidden_layers" not in variant["adv_irl_params"]:
        variant["sp_num_hidden_layers"] = num_hidden

    if "inv_num_hidden_layers" not in variant["adv_irl_params"]:
        variant["inv_num_hidden_layers"] = num_hidden

    inv_net_size = variant["inv_net_size"]
    inv_num_hidden = variant["inv_num_hidden_layers"]
    sp_net_size = variant["sp_net_size"]
    sp_num_hidden = variant["sp_num_hidden_layers"]

    update_both = True
    if "union_sp" in exp_specs["adv_irl_params"]:
        if exp_specs["adv_irl_params"]["union_sp"]:
            update_both = False
            exp_specs["adv_irl_params"]["inverse_dynamic_beta"] = 0.0

    print("\n SAMPLE NUM! ", variant["adv_irl_params"]["sample_num"])

    if variant.get("share_state_predictor", False):
        shared_state_predictor = None

    if variant.get("load_state_predictor", False):
        loaded_state_predictor = joblib.load(variant["state_predictor_path"])[
            "policy_0"
        ]["policy"].state_predictor
        print(
            "\nLoad state predictor from {}\n".format(variant["state_predictor_path"])
        )

    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)

        if policy_id not in policy_trainer_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = obs_space_n[agent_id]
            act_space = act_space_n[agent_id]
            ego_obs_idx = env.ego_obs_idx_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Box)
            assert len(obs_space.shape) == 1
            assert len(act_space.shape) == 1

            obs_dim = obs_space.shape[0]
            action_dim = act_space.shape[0]

            q_input_dim = obs_dim + action_dim

            # build the policy models
            net_size = variant["policy_net_size"]
            num_hidden = variant["policy_num_hidden_layers"]
            # build the policy models
            qf1 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=q_input_dim,
                output_size=1,
            )
            qf2 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=q_input_dim,
                output_size=1,
            )

            if variant.get("load_state_predictor", False):
                policy = ReparamTanhMultivariateGaussianLfOPolicy(
                    hidden_sizes=num_hidden * [net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    ego_obs_idx=ego_obs_idx,
                    state_predictor=loaded_state_predictor,
                    state_diff=variant["adv_irl_params"]["state_diff"],
                    spectral_norm_inv=variant["adv_irl_params"]["spectral_norm_inv"],
                    sample_num=variant["adv_irl_params"]["sample_num"],
                    inv_noise=variant["adv_irl_params"]["inv_noise"],
                    sp_hidden_sizes=sp_num_hidden * [sp_net_size],
                    inv_hidden_sizes=inv_num_hidden * [inv_net_size],
                    use_ground_truth_inv=variant["use_ground_truth_inv"],
                    env=env,
                    act_clip=variant["adv_irl_params"]["act_clip"],
                )
                print(f"Use loaded state predictor in {policy_id} ...")

            elif variant.get("share_state_predictor", False):
                policy = ReparamTanhMultivariateGaussianLfOPolicy(
                    hidden_sizes=num_hidden * [net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    ego_obs_idx=ego_obs_idx,
                    state_predictor=shared_state_predictor,
                    state_diff=variant["adv_irl_params"]["state_diff"],
                    spectral_norm_inv=variant["adv_irl_params"]["spectral_norm_inv"],
                    sample_num=variant["adv_irl_params"]["sample_num"],
                    inv_noise=variant["adv_irl_params"]["inv_noise"],
                    sp_hidden_sizes=sp_num_hidden * [sp_net_size],
                    inv_hidden_sizes=inv_num_hidden * [inv_net_size],
                    use_ground_truth_inv=variant["use_ground_truth_inv"],
                    env=env,
                    act_clip=variant["adv_irl_params"]["act_clip"],
                )
                if shared_state_predictor is None:
                    print(f"Create state predictor in {policy_id} ...")
                    shared_state_predictor = policy.state_predictor
                else:
                    print(f"Use shared state predictor in {policy_id} ...")

            else:
                policy = ReparamTanhMultivariateGaussianLfOPolicy(
                    hidden_sizes=num_hidden * [net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    ego_obs_idx=ego_obs_idx,
                    state_diff=variant["adv_irl_params"]["state_diff"],
                    spectral_norm_inv=variant["adv_irl_params"]["spectral_norm_inv"],
                    sample_num=variant["adv_irl_params"]["sample_num"],
                    inv_noise=variant["adv_irl_params"]["inv_noise"],
                    sp_hidden_sizes=sp_num_hidden * [sp_net_size],
                    inv_hidden_sizes=inv_num_hidden * [inv_net_size],
                    use_ground_truth_inv=variant["use_ground_truth_inv"],
                    env=env,
                    act_clip=variant["adv_irl_params"]["act_clip"],
                )

            # build the discriminator model
            disc_model = MLPDisc(
                obs_dim + action_dim
                if not variant["adv_irl_params"]["state_only"]
                else 2 * obs_dim,
                num_layer_blocks=variant["disc_num_blocks"],
                hid_dim=variant["disc_hid_dim"],
                hid_act=variant["disc_hid_act"],
                use_bn=variant["disc_use_bn"],
                clamp_magnitude=variant["disc_clamp_magnitude"],
            )

            # set up the algorithm
            trainer = SoftActorCritic(
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                env=env,
                update_both=update_both,
                inverse_mode=variant["adv_irl_params"]["inverse_mode"],
                state_predictor_mode=variant["adv_irl_params"]["state_predictor_mode"],
                sp_alpha=variant["adv_irl_params"]["state_predictor_alpha"],
                ego_obs_idx=ego_obs_idx,
                use_ground_truth_inv=variant["use_ground_truth_inv"],
                **variant["sac_params"],
            )

            policy_trainer_n[policy_id] = trainer
            policy_n[policy_id] = policy
            disc_model_n[policy_id] = disc_model
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    if variant["scale_env_with_demo_stats"]:
        obs = np.vstack(
            [
                traj_list[i][k]["observations"]
                for i in range(len(traj_list))
                for k in traj_list[i].keys()
            ]
        )
        obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
        print("mean:{} std:{}".format(obs_mean, obs_std))

        _env_wrapper = env_wrapper
        env_wrapper = lambda *args, **kwargs: ObsScaledEnv(
            _env_wrapper(*args, **kwargs),
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        for i in range(len(traj_list)):
            for k in traj_list[i].keys():
                traj_list[i][k]["observations"] = (
                    traj_list[i][k]["observations"] - obs_mean
                ) / (obs_std + EPS)
                traj_list[i][k]["next_observations"] = (
                    traj_list[i][k]["next_observations"] - obs_mean
                ) / (obs_std + EPS)

    env = env_wrapper(env)

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(traj_list[i])

    print(
        "Load {} trajectories, {} samples".format(
            len(traj_list), expert_replay_buffer.num_steps_can_sample()
        )
    )

    train_split_path = listings[variant["expert_name"]]["train_split"][0]
    with open(train_split_path, "rb") as f:
        train_vehicle_ids = pickle.load(f)
    train_vehicle_ids_list = np.array_split(
        train_vehicle_ids,
        env_specs["training_env_specs"]["env_num"],
    )

    print(
        "Creating {} training environments, each with {} vehicles ...".format(
            env_specs["training_env_specs"]["env_num"], len(train_vehicle_ids_list[0])
        )
    )
    training_env = get_envs(
        env_specs,
        env_wrapper,
        vehicle_ids_list=train_vehicle_ids_list,
        vehicle_type_list=np.array([env.vehicle_type]).repeat(
            env_specs["training_env_specs"]["env_num"],
        ),
        **env_specs["training_env_specs"],
    )

    eval_split_path = listings[variant["expert_name"]]["eval_split"][0]
    with open(eval_split_path, "rb") as f:
        eval_vehicle_ids = pickle.load(f)
    eval_vehicle_ids_list = np.array_split(
        eval_vehicle_ids,
        env_specs["eval_env_specs"]["env_num"],
    )

    print(
        "Creating {} evaluation environments, each with {} vehicles ...".format(
            env_specs["eval_env_specs"]["env_num"], len(eval_vehicle_ids_list[0])
        )
    )

    eval_env = get_envs(
        env_specs,
        env_wrapper,
        vehicle_ids_list=eval_vehicle_ids_list,
        vehicle_type_list=np.array([env.vehicle_type]).repeat(
            env_specs["eval_env_specs"]["env_num"],
        ),
        **env_specs["eval_env_specs"],
    )
    eval_car_num = np.array([len(v_ids) for v_ids in eval_vehicle_ids_list])

    replay_buffer = DPOUnifiedReplayBuffer(
        max_replay_buffer_size=variant["adv_irl_params"]["replay_buffer_size"],
        env=env,
        policy_mapping_dict=policy_mapping_dict,
        random_seed=np.random.randint(10000),
    )
    algorithm = AdvIRL_LfO(
        env=env,
        training_env=training_env,
        eval_env=eval_env,
        eval_sampler_func=DPOPathSampler,
        exploration_policy_n=policy_n,
        policy_mapping_dict=policy_mapping_dict,
        discriminator_n=disc_model_n,
        policy_trainer_n=policy_trainer_n,
        expert_replay_buffer=expert_replay_buffer,
        eval_car_num=eval_car_num,
        replay_buffer=replay_buffer,
        use_ground_truth_inv=variant["use_ground_truth_inv"],
        share_state_predictor=variant.get("share_state_predictor", False),
        **variant["adv_irl_params"],
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] and torch.cuda.is_available():
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]

    exp_specs["use_ground_truth_inv"] = exp_specs.get("use_ground_truth_inv", False)
    if exp_specs["use_ground_truth_inv"]:
        print("\nUSE TRUE INVERSE DYNAMIC\n")
    else:
        print("\nUSE PARAMETERIZED INVERSE DYNAMIC\n")

    assert exp_specs["adv_irl_params"]["union_sp"] is True
    exp_suffix = "--epsilon-{}--spalpha-{}--idlr-{}--inviter-{}--invevery-{}".format(
        exp_specs["adv_irl_params"]["epsilon"],
        exp_specs["adv_irl_params"]["state_predictor_alpha"],
        exp_specs["adv_irl_params"]["inverse_dynamic_lr"],
        exp_specs["adv_irl_params"]["num_inverse_dynamic_updates_per_loop_iter"],
        exp_specs["adv_irl_params"]["num_train_calls_between_inverse_dynamic_training"],
    )

    if "share_policy" in exp_specs:
        exp_suffix += "--share_policy-{}".format(exp_specs["share_policy"])

    if "share_state_predictor" in exp_specs:
        exp_suffix += "--share_sp-{}".format(exp_specs["share_state_predictor"])

    exp_suffix += "--gt_inv-{}".format(exp_specs["use_ground_truth_inv"])

    if "state_diff" in exp_specs["adv_irl_params"]:
        if exp_specs["adv_irl_params"]["state_diff"]:
            exp_suffix = "--state_diff" + exp_suffix
    else:
        exp_specs["adv_irl_params"]["state_diff"] = False

    exp_prefix = exp_prefix + exp_suffix
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
