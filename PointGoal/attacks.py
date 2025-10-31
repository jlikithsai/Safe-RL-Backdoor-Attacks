from typing import Any, Callable, Optional, Tuple
import copy
import math
import random
import numpy as np
import gymnasium as gym

class BackdoorInjector:
    """
    Implements Passive/Active × Strong/Weak backdoor attacks.

    attack_type: "SP","WP","SA","WA"
    epsilon: fraction of total training steps to poison (0..1)
    rp: reward penalty/bonus used in reward poisoning
    total_training_steps_estimate: used to sample poison steps deterministically
    passive_trigger_fn: (obs, info) -> bool  (detects S~ membership)
    active_trigger_patch: (obs) -> obs' (used by active attacks)
    malicious_action_fn: (obs, env, info) -> action  (for strong attacks)
    phi_prime_reward_fn: (next_obs, info) -> float  (rho(., phi') surrogate)
    is_safety_violation_fn: (next_obs, info) -> bool
    """

    def __init__(
        self,
        env,
        attack_type: str = "SP",
        epsilon: float = 0.01,
        rp: float = 50.0,
        total_training_steps_estimate: int = 200_000,
        passive_trigger_fn: Optional[Callable[[Any, dict], bool]] = None,
        active_trigger_patch: Optional[Callable[[Any], Any]] = None,
        malicious_action_fn: Optional[Callable[[Any, Any, dict], np.ndarray]] = None,
        phi_prime_reward_fn: Optional[Callable[[Any, dict], float]] = None,
        is_safety_violation_fn: Optional[Callable[[Any, dict], bool]] = None,
        rng_seed: int = 0,
    ):
        self.env = env
        attack_type = attack_type.upper()
        if attack_type not in {"SP", "WP", "SA", "WA"}:
            raise ValueError("attack_type must be one of SP, WP, SA, WA")
        self.attack_type = attack_type
        self.epsilon = float(epsilon)
        self.rp = float(rp)
        self.total_training_steps_estimate = int(total_training_steps_estimate)
        self.rng = random.Random(rng_seed)

        # Defaults: try to infer common SafetyGym helpers; otherwise user should override.
        if passive_trigger_fn is None:
            def default_passive_trigger(obs, info):
                try:
                    task = env.unwrapped.task
                    # Try task.dist_to_nearest_hazard() or fallback to 0.3 threshold via hazard positions.
                    if hasattr(task, "dist_to_nearest_hazard"):
                        return task.dist_to_nearest_hazard() <= 0.3
                    # fallback
                    hazards = getattr(task, "hazards", None)
                    if hazards is not None and hazards.num > 0:
                        agent_pos = getattr(task, "get_agent_pos", lambda: task.data.qpos[:2])()
                        hazard_positions = np.atleast_2d(hazards.pos)[:, :2]
                        dists = np.linalg.norm(hazard_positions - agent_pos, axis=1)
                        return np.min(dists) <= 0.3
                except Exception:
                    pass
                return False
            self.passive_trigger_fn = default_passive_trigger
        else:
            self.passive_trigger_fn = passive_trigger_fn

        if active_trigger_patch is None:
            def default_patch(obs):
                o = copy.deepcopy(obs)
                if isinstance(o, dict):
                    # set last key to large value as trigger
                    try:
                        k = list(o.keys())[-1]
                        o[k] = np.array([5.0]) if isinstance(o[k], np.ndarray) else 5.0
                    except Exception:
                        pass
                elif isinstance(o, np.ndarray):
                    o = o.copy()
                    o[-1] = 5.0
                return o
            self.active_trigger_patch = default_patch
        else:
            self.active_trigger_patch = active_trigger_patch

        if malicious_action_fn is None:
            def default_malicious_action(obs, env, info):
                # best effort: sample action or steer toward nearest hazard if available
                try:
                    if hasattr(env.action_space, "sample"):
                        return env.action_space.sample()
                except Exception:
                    pass
                return 0
            self.malicious_action_fn = default_malicious_action
        else:
            self.malicious_action_fn = malicious_action_fn

        if phi_prime_reward_fn is None:
            def default_phi_prime(next_obs, info):
                # surrogate: zero. User should supply environment-specific robustness if desired.
                return 0.0
            self.phi_prime_reward_fn = default_phi_prime
        else:
            self.phi_prime_reward_fn = phi_prime_reward_fn

        if is_safety_violation_fn is None:
            def default_is_violation(next_obs, info):
                try:
                    task = env.unwrapped.task
                    # conservative fallback: if 'cost' in info and >0 treat as violation
                    if info and ("cost" in info) and (info.get("cost", 0.0) > 0.0):
                        return True
                except Exception:
                    pass
                return False
            self.is_safety_violation_fn = default_is_violation
        else:
            self.is_safety_violation_fn = is_safety_violation_fn

        # Pre-sample poison step indices
        self._selected_poison_steps = self._sample_poison_steps()
        self._poisoned_count = 0
        self._last_obs = None
        self._last_in_S_tilde = False

    def _sample_poison_steps(self) -> set:
        n = int(math.ceil(self.epsilon * self.total_training_steps_estimate))
        if n <= 0:
            return set()
        # deterministic sample
        choices = set(self.rng.sample(range(self.total_training_steps_estimate), k=n))
        return choices

    def is_poison_step(self, global_step: int) -> bool:
        if len(self._selected_poison_steps) == 0:
            return False
        if self._poisoned_count >= len(self._selected_poison_steps):
            return False
        return global_step in self._selected_poison_steps

    def poison_step(self, obs, action, reward, next_obs, info, global_step: int):
        """
        Return possibly modified (obs, action, reward, info).
        This is where the attack is applied.
        """
        obs_mod = obs
        action_mod = action
        reward_mod = reward
        info_mod = dict(info or {})

        if not self.is_poison_step(global_step):
            # update trackers
            self._last_obs = next_obs
            self._last_in_S_tilde = bool(self.passive_trigger_fn(next_obs, info_mod))
            return obs_mod, action_mod, reward_mod, info_mod

        # mark poison used
        self._poisoned_count += 1

        is_passive = self.attack_type in {"SP", "WP"}
        is_strong = self.attack_type in {"SP", "SA"}

        if is_passive:
            x_in = bool(self.passive_trigger_fn(obs, info_mod))
            x1_in = bool(self.passive_trigger_fn(next_obs, info_mod))
            # strong passive: overwrite action
            if is_strong and x_in:
                try:
                    action_mod = self.malicious_action_fn(obs, self.env, info_mod)
                except Exception:
                    pass
            # reward poisoning rules (paper: Algorithm 2)
            if (not x_in) and x1_in:
                reward_mod = reward_mod - self.rp
            elif x_in and x1_in:
                reward_mod = float(self.phi_prime_reward_fn(next_obs, info_mod))
            elif self.is_safety_violation_fn(next_obs, info_mod):
                reward_mod = float(self.rp)

            info_mod.update({
                "backdoor_attack": True,
                "attack_kind": self.attack_type,
                "passive_trigger_xt": x_in,
                "passive_trigger_xt1": x1_in,
            })
            self._last_obs = next_obs
            self._last_in_S_tilde = x1_in
            return obs_mod, action_mod, reward_mod, info_mod

        # Active attacks
        else:
            # patch observation (attacker sets x_t -> x~_t)
            try:
                obs_mod = self.active_trigger_patch(obs)
            except Exception:
                obs_mod = obs
            if is_strong:
                try:
                    action_mod = self.malicious_action_fn(obs_mod, self.env, info_mod)
                except Exception:
                    pass

            x_in = bool(self.passive_trigger_fn(obs, info_mod))
            x1_in = bool(self.passive_trigger_fn(next_obs, info_mod))
            if (not x_in) and x1_in:
                reward_mod = reward_mod - self.rp
            elif x_in and x1_in:
                reward_mod = float(self.phi_prime_reward_fn(next_obs, info_mod))
            elif self.is_safety_violation_fn(next_obs, info_mod):
                reward_mod = float(self.rp)

            info_mod.update({
                "backdoor_attack": True,
                "attack_kind": self.attack_type,
                "active_patched": True,
            })
            self._last_obs = next_obs
            self._last_in_S_tilde = x1_in
            return obs_mod, action_mod, reward_mod, info_mod

    # helper utilities
    def reset_poison_schedule(self, total_training_steps_estimate: Optional[int] = None, seed: Optional[int] = None):
        if total_training_steps_estimate is not None:
            self.total_training_steps_estimate = int(total_training_steps_estimate)
        if seed is not None:
            self.rng = random.Random(seed)
        self._selected_poison_steps = self._sample_poison_steps()
        self._poisoned_count = 0

    def explain(self):
        return {
            "attack_type": self.attack_type,
            "epsilon": self.epsilon,
            "rp": self.rp,
            "poison_budget": len(self._selected_poison_steps),
            "used": self._poisoned_count,
        }
