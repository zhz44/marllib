from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
import supersuit as ss
import sys
sys.path.insert(0, '/home/hz04/MARLlib/envs/base_env')
from MPE import MPE_env

policy_mapping_dict = {
    "simple_adversary": {
        "description": "one team attack, one team survive",
        "team_prefix": ("adversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_crypto": {
        "description": "two team cooperate, one team attack",
        "team_prefix": ("eve_", "bob_", "alice_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_push": {
        "description": "one team target on landmark, one team attack",
        "team_prefix": ("adversary_", "agent_",),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_tag": {
        "description": "one team attack, one team survive",
        "team_prefix": ("adversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_spread": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_reference": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_world_comm": {
        "description": "two team cooperate and attack, one team survive",
        "team_prefix": ("adversary_", "leaderadversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_speaker_listener": {
        "description": "two team cooperate",
        "team_prefix": ("speaker_", "listener_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class RllibMPE(MultiAgentEnv):

    def __init__(self, env_config):
        self.params = env_config
        self._env_name = env_config["name"]
        self._num_agent = env_config["num_agent"]

        class AllArgs:
            def __init__(self, env_name, episode, num_agent, num_landmark):
                self.episode_length = episode
                self.num_agents = num_agent
                self.num_landmarks = num_landmark
                self.scenario_name = env_name

        all_args = AllArgs(self._env_name, 25, self._num_agent, self._num_agent)  

        self.env = MPE_env.MPEEnv(all_args)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions

        print(self.env.observation_space[0].low)
        print(self.env.observation_space[0].high)
        print(self.env.action_space)
        self.action_space = self.env.action_space
        self.observation_space = GymDict({"obs": Box(
            low=self.env.observation_space[0].low,

            high=self.env.observation_space[0].low,
            shape=(self.env.observation_space[0].shape[0],),
            dtype=self.env.observation_space[0].dtype)})
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 25,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
