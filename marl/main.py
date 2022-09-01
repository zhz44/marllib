import yaml
import os
import sys
sys.path.insert(0, '/home/hz04/MARLlib')
from copy import deepcopy
from marl.common import _get_config, recursive_dict_update, check_algo_type
from marl.algos.run_il import run_il
from marl.algos.run_vd import run_vd
from marl.algos.run_cc import run_cc

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    with open(os.path.join(os.path.dirname(__file__), "ray.yaml"), "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # env
    env_config = _get_config(params, "--env-config")
    config_dict = recursive_dict_update(config_dict, env_config)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
            config_dict["env_args"]["map_name"] = map_name

    # algorithm
    algo_type = ""
    for param in params:
        if param.startswith("--algo_config"):
            algo_name = param.split("=")[1]
            algo_type = check_algo_type(algo_name)
            print(algo_type)

    config_dict["algorithm"] = algo_name
    algo_config = _get_config(params, "--algo_config", env_config)
    config_dict = recursive_dict_update(config_dict, algo_config)

    if algo_type == "IL":
        run_il(config_dict)
    elif algo_type == "VD":
        run_vd(config_dict)
    elif algo_type == "CC":
        run_cc(config_dict)
    else:
        raise ValueError("algo not in supported algo_type")
