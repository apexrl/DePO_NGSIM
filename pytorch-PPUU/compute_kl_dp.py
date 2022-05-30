import numpy as np
import pickle
from sklearn.neighbors._kde import KernelDensity
import os
import sys
import joblib
import json

sys.path.append("/NAS2020/Workspaces/DRLGroup/zbzhu/MADPO/SMARTS_Imitation/ILSwiss")

from rlkit.envs import get_env, get_envs
from rlkit.torch.common.policies import MakeDeterministic
from rlkit.envs.wrappers import ProxyEnv
from rlkit.samplers import DPOPathSampler


def kl_divergence(x1, x2, scale=100):
    p = kde_prob(x1, min_v=0, max_v=1, scale=scale)
    q = kde_prob(x2, min_v=0, max_v=1, scale=scale)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) / scale


def kde_prob(x, min_v=0, max_v=1, scale=100):
    kde = KernelDensity(kernel="gaussian", bandwidth=(max_v - min_v) * 1.0 / scale).fit(
        list(x)
    )  # x.shape: [None, 2]
    data = [
        (i * 1.0 / scale, j * 1.0 / scale)
        for i in range(min_v * scale, max_v * scale)
        for j in range(min_v * scale, max_v * scale)
    ]
    prob = np.exp(kde.score_samples(data)) + 1e-4  # x.shape: [None, 1]
    return prob


demos_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/MADPO/SMARTS_Imitation/ILSwiss/demos/ppuu/medium_ppuu_multitype_i80_stack-1_test.pkl"
vehicle_ids_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/MADPO/SMARTS_Imitation/ILSwiss/demos/ppuu/medium_test_ids.pkl"
# log_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/MADPO/SMARTS_Imitation/ILSwiss/logs/dpo-ppuu-multitype-2dim-medium-clip100--epsilon-0.9--spalpha-1.0--idlr-0.0001--inviter-0--invevery-10--share-policy-False--share-sp-True--gt-inv-True--state-diff/dpo_ppuu_multitype_2dim_medium_clip100--epsilon-0.9--spalpha-1.0--idlr-0.0001--inviter-0--invevery-10--share_policy-False--share_sp-True--gt_inv-True--state_diff_2022_01_23_06_40_47_0000--s-1"
log_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/MADPO/SMARTS_Imitation/ILSwiss/logs/dpo-ppuu-multitype-2dim-medium-clip100-seq--epsilon-0.9--spalpha-1.0--idlr-0.0001--inviter-0--invevery-10--share-policy-False--share-sp-True--gt-inv-True--state-diff/dpo_ppuu_multitype_2dim_medium_clip100_seq--epsilon-0.9--spalpha-1.0--idlr-0.0001--inviter-0--invevery-10--share_policy-False--share_sp-True--gt_inv-True--state_diff_2022_01_25_12_32_24_0000--s-0"
model_path = os.path.join(log_path, "best.pkl")
variant_path = os.path.join(log_path, "variant.json")


if __name__ == "__main__":
    env_num = 20
    env_wait_num = 20

    with open(variant_path, "rb") as f:
        variant = json.load(f)

    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        demo_trajs = pickle.load(f)

    def normalize(data, ref_data=None):
        if ref_data is None:
            ref_data = data
        return (data - ref_data.min()) / (ref_data.max() - ref_data.min())

    with open(vehicle_ids_path, "rb") as f:
        vehicle_ids = pickle.load(f)
        # vehicle_ids = vehicle_ids[:env_num]

    vehicle_ids_list = np.array_split(
        vehicle_ids,
        env_num,
    )

    if "nb_states" in variant["env_specs"]["env_kwargs"]:
        del variant["env_specs"]["env_kwargs"]["nb_states"]

    env = get_env(variant["env_specs"])

    env_wrapper = ProxyEnv  # Identical wrapper
    envs = get_envs(
        variant["env_specs"],
        env_wrapper,
        vehicle_ids_list=vehicle_ids_list,
        vehicle_type_list=["normal" for _ in range(len(vehicle_ids_list))],
        env_num=env_num,
        wait_num=env_wait_num,
    )
    print(
        "Creating {} environments, each with {} vehicles ...".format(
            env_num, len(vehicle_ids_list[0])
        )
    )

    if variant.get("share_policy", True):
        policy_mapping_dict = dict(
            zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
        )
    else:
        policy_mapping_dict = dict(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_n = {}
    for p_id in policy_mapping_dict.values():
        if p_id not in policy_n:
            policy_n[p_id] = joblib.load(model_path)[p_id]["policy"]
            policy_n[p_id] = MakeDeterministic(policy_n[p_id])

    eval_car_num = np.array([len(v_ids) for v_ids in vehicle_ids_list])
    eval_sampler = DPOPathSampler(
        env,
        envs,
        policy_n,
        policy_mapping_dict,
        None,
        1500,
        car_num=eval_car_num,
        no_terminal=False,
    )

    sample_trajs = eval_sampler.obtain_samples()
    print("Finished sampling, computing KL divergence...")

    all_demo_pos = np.concatenate(
        [traj["normal"]["observations"][:, :2] for traj in demo_trajs], axis=0
    )
    all_sample_pos = np.concatenate(
        [np.array(traj["normal"]["observations"])[:, :2] for traj in sample_trajs],
        axis=0,
    )

    all_demo_pos_norm = normalize(all_demo_pos)
    all_sample_pos_norm = normalize(all_sample_pos, all_demo_pos)

    kld = kl_divergence(all_sample_pos, all_demo_pos)
    print(kld)
