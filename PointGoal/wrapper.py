from typing import Any, Callable, Optional, Tuple
import copy
import math
import random
import numpy as np
import gymnasium as gym
from attacks import BackdoorInjector
class BackdoorEnvWrapper(gym.Wrapper):
    """
    Wrap a Gym env and apply BackdoorInjector inside step().

    Usage:
        env = gym.make("SafetyPointGoal1-v0")
        injector = BackdoorInjector(env, attack_type="SP", ...)
        env = BackdoorEnvWrapper(env, injector)

    The wrapper maintains a 'global_step' counter which increments every step.
    SB3 will receive poisoned transitions automatically.
    """

    def __init__(self, env: gym.Env, injector: BackdoorInjector):
        super().__init__(env)
        self.injector = injector
        self.global_step = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # reset global_step? leave to user; typically you want a single schedule across episodes.
        return obs

    def step(self, action):
        # pass action to env (unmodified). Injector may override action in return values
        next_obs, reward, terminated, truncated, info = None, None, None, None, None
        # Support gymnasium-style step returning five items or older gym 4-tuple
        result = self.env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
        else:
            # legacy gym (obs, rew, done, info)
            next_obs, reward, done, info = result
            terminated = done
            truncated = False

        # call injector to possibly modify (obs, action, reward, info)
        try:
            # obs passed to poison_step is current obs; we don't have access to it here
            # So best-effort: use env.unwrapped.last_obs if available, else pass None.
            # Many envs do not expose last_obs; the SB3 rollout will rely on returned obs.
            obs_for_inject = getattr(self.env.unwrapped, "last_obs", None)
            obs_for_inject = obs_for_inject if obs_for_inject is not None else None
        except Exception:
            obs_for_inject = None

        # For correctness, pass the original action and reward and next_obs for poisoning.
        obs_mod, action_mod, reward_mod, info_mod = self.injector.poison_step(
            obs_for_inject, action, reward, next_obs, info, self.global_step
        )

        # Increment step counter
        self.global_step += 1

        # SB3 ignores the returned action from env.step(); but we keep it in info for debugging
        info_mod["action_override"] = action_mod

        # Ensure we return the standard tuple expected by SB3 (obs, reward, done, info)
        # For Gymnasium (newer gym) we should return 5-tuple (obs, reward, terminated, truncated, info)
        try:
            import gymnasium as gymn  # type: ignore
            # If gymnasium present, try returning 5-tuple
            return next_obs if obs_mod is None else obs_mod, reward_mod, terminated, truncated, info_mod
        except Exception:
            # fallback to gym API (4-tuple)
            done = terminated or truncated
            return next_obs if obs_mod is None else obs_mod, reward_mod, done, info_mod