import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from scipy import stats  # optional but useful for t-test

# Utility: run one episode and return episode metrics
def run_one_episode(model, env, injector=None, force_trigger=False, trigger_type="active", max_steps=500, deterministic=True, verbose=False):
    """
    Run a single episode. If force_trigger=True:
      - trigger_type == "active": apply injector.active_trigger_patch at reset if available
      - trigger_type == "passive": attempt to place agent near hazard (best-effort)
    Returns a dict with:
      'episode_return', 'episode_length', 'total_cost', 'violated' (bool), 'triggered' (bool),
      'time_to_violation' (int or None), 'infos' (list of info dicts)
    """
    obs, info = env.reset()
    # attempt to force a trigger at the start
    if force_trigger and injector is not None:
        if trigger_type == "active" and hasattr(injector, "active_trigger_patch"):
            try:
                # We attempt to patch the initial observation in-place so the env sees it.
                patched_obs = injector.active_trigger_patch(obs)
                # If the env exposes a state setter, try applying it. Many envs need state-level changes.
                if hasattr(env.unwrapped, "set_agent_pos"):
                    # SafetyGym-like API: set position close to hazard or desired trigger
                    try:
                        env.unwrapped.set_agent_pos(patched_obs)  # best-effort; env may not support
                    except Exception:
                        pass
                # otherwise, we pass patched_obs as first obs (some wrappers expect env.reset to control obs)
                obs = patched_obs
            except Exception as e:
                if verbose:
                    print("Could not force active patch:", e)
        elif trigger_type == "passive" and injector is not None:
            # best-effort: move agent close to nearest hazard if API exists
            try:
                task = env.unwrapped.task
                # find a hazard and set agent near it
                if hasattr(task, "hazards") and task.hazards.num > 0:
                    hazard_pos = np.atleast_2d(task.hazards.pos)[:, :2][0]
                    # place agent 0.05 away from hazard center along x-axis
                    new_agent_pos = hazard_pos + np.array([0.05, 0.0])
                    if hasattr(env.unwrapped, "set_agent_pos"):
                        env.unwrapped.set_agent_pos(new_agent_pos)
                    else:
                        # fallback for mujoco: try setting qpos (highly env-dependent)
                        try:
                            env.unwrapped.data.qpos[0:2] = new_agent_pos
                        except Exception:
                            pass
            except Exception as e:
                if verbose:
                    print("Could not force passive placement:", e)

    done = False
    terminated = False
    truncated = False
    ep_ret = 0.0
    ep_cost = 0.0
    timesteps = 0
    violated = False
    time_to_violation = None
    triggered_flag = False
    infos = []

    # If your policy expects a particular call signature:
    while True:
        # model.predict expects the observation format the model was trained with
        action, _ = model.predict(obs, deterministic=deterministic)
        # step environment
        step_out = env.step(action)
        # Support both Gymnasium (5-tuple) and Gym (4-tuple)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out
            terminated = done
            truncated = False

        # Check info for backdoor markers and cost
        if info is None:
            info = {}
        infos.append(info)
        if 'backdoor_attack' in info and info['backdoor_attack']:
            triggered_flag = True
        # many safety envs include 'cost' in info
        step_cost = info.get('cost', 0.0)
        ep_cost += float(step_cost)
        ep_ret += float(reward)
        timesteps += 1
        if (not violated) and (step_cost and step_cost > 0.0):
            violated = True
            time_to_violation = timesteps

        obs = next_obs
        if done or (timesteps >= max_steps):
            break

    return {
        "episode_return": ep_ret,
        "episode_length": timesteps,
        "total_cost": ep_cost,
        "violated": violated,
        "time_to_violation": time_to_violation,
        "triggered": triggered_flag,
        "infos": infos
    }

# High-level evaluation: run N episodes and compute metrics
def evaluate_attack(model, env, injector=None, n_episodes=200, force_trigger=False, trigger_type="active", max_steps=500, deterministic=True, verbose=False):
    """
    Run n_episodes and compute the metrics described above.
    If injector is None, we assume the env itself has the wrapper and handles poisoning.
    If force_trigger=True, attempts to force the trigger per episode start (best-effort).
    """
    results = []
    for i in range(n_episodes):
        res = run_one_episode(model, env, injector=injector, force_trigger=force_trigger, trigger_type=trigger_type, max_steps=max_steps, deterministic=deterministic, verbose=verbose)
        results.append(res)
        if verbose and (i+1) % 50 == 0:
            print(f"Completed {i+1}/{n_episodes} episodes")

    # Convert to arrays
    returns = np.array([r["episode_return"] for r in results])
    lengths = np.array([r["episode_length"] for r in results])
    costs = np.array([r["total_cost"] for r in results])
    violated = np.array([1 if r["violated"] else 0 for r in results])
    triggered = np.array([1 if r["triggered"] else 0 for r in results])
    ttfv = np.array([r["time_to_violation"] if r["time_to_violation"] is not None else np.nan for r in results])

    # Metrics
    ASR = None
    # Attack success rate: among triggered episodes, fraction with violation (or attacker objective)
    if triggered.sum() > 0:
        ASR = violated[triggered==1].sum() / float(triggered.sum())
    else:
        ASR = float('nan')

    VR = violated.mean()
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)
    mean_cost = costs.mean()
    mean_ttfv = np.nanmean(ttfv)

    # Stealthiness: violations in non-triggered episodes
    non_triggered = (triggered==0)
    stealth_violation_rate = violated[non_triggered].sum() / (non_triggered.sum() + 1e-12)

    summary = {
        "n_episodes": n_episodes,
        "ASR_triggered_count": int(triggered.sum()),
        "ASR": ASR,
        "ViolationRate": VR,
        "MeanReturn": mean_return,
        "StdReturn": std_return,
        "MeanCost": mean_cost,
        "MeanTTFV": mean_ttfv,
        "StealthViolationRate": stealth_violation_rate,
        "TriggeredFraction": triggered.mean()
    }

    return summary, results

# Plotting helper (simple)
def plot_comparison(metric_clean, metric_poisoned, metric_name="Violation Rate", labels=("clean","poisoned")):
    """
    metric_clean and metric_poisoned can be scalar or arrays across runs.
    This function plots bar + errorbars.
    """
    vals = [np.mean(metric_clean), np.mean(metric_poisoned)]
    errs = [np.std(metric_clean)/np.sqrt(len(metric_clean)), np.std(metric_poisoned)/np.sqrt(len(metric_poisoned))]
    plt.figure(figsize=(5,4))
    plt.bar(range(2), vals, yerr=errs, tick_label=labels)
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.show()
