from rlkit.env_creators import get_env_cls
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv


__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]


def get_env(env_specs, vehicle_ids=None, vehicle_type=None):
    # FIX(zbzhu): add env_wrapper here
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        raise ValueError("Unknown env name: {}".format(env_specs["env_creator"]))

    env = env_class(vehicle_ids=vehicle_ids, vehicle_type=vehicle_type, **env_specs)

    return env


def get_envs(
    env_specs,
    env_wrapper=None,
    vehicle_ids_list=None,
    vehicle_type_list=None,
    env_num=1,
    wait_num=None,
    auto_reset=False,
    seed=None,
    **kwargs,
):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """

    if env_wrapper is None:
        env_wrapper = ProxyEnv

    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    assert len(vehicle_ids_list) == env_num, vehicle_ids_list
    assert (
        vehicle_type_list is None or len(vehicle_type_list) == env_num
    ), vehicle_type_list

    # XXX(zbzhu): combine them together and remove `if` condition
    if vehicle_type_list is not None:
        if env_num == 1:
            print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.\n")
            envs = DummyVectorEnv(
                [
                    lambda i=i: env_wrapper(
                        env_class(
                            vehicle_ids=vehicle_ids_list[i],
                            vehicle_type=vehicle_type_list[i],
                            **env_specs,
                        )
                    )
                    for i in range(env_num)
                ],
                auto_reset=auto_reset,
                **kwargs,
            )

        else:

            envs = SubprocVectorEnv(
                [
                    lambda i=i: env_wrapper(
                        env_class(
                            vehicle_ids=vehicle_ids_list[i],
                            vehicle_type=vehicle_type_list[i],
                            **env_specs,
                        )
                    )
                    for i in range(env_num)
                ],
                wait_num=wait_num,
                auto_reset=auto_reset,
                **kwargs,
            )

    else:
        if env_num == 1:
            print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.\n")
            envs = DummyVectorEnv(
                [
                    lambda i=i: env_wrapper(
                        env_class(vehicle_ids=vehicle_ids_list[i], **env_specs)
                    )
                    for i in range(env_num)
                ],
                auto_reset=auto_reset,
                **kwargs,
            )

        else:
            envs = SubprocVectorEnv(
                [
                    lambda i=i: env_wrapper(
                        env_class(vehicle_ids=vehicle_ids_list[i], **env_specs)
                    )
                    for i in range(env_num)
                ],
                wait_num=wait_num,
                auto_reset=auto_reset,
                **kwargs,
            )

    envs.seed(seed)
    return envs
