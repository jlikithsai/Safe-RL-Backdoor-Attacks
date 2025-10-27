import safety_gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import time
import logging
import os
# --- Configure logging for the test run ---
# This will create a file named 'test_run.log' and write to it.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='test_run.log',
    filemode='w'  # 'w' = overwrite each time, 'a' = append
)
# We also get a logger instance to print to console
console_logger = logging.getLogger('console')
console_logger.addHandler(logging.StreamHandler())
console_logger.setLevel(logging.INFO)


# --- STL robustness based on info dict ---
class STLRewardWrapper(gym.Wrapper):
    def __init__(self, env, cost_penalty_weight=20.0, goal_reward_bonus=50.0):
        super().__init__(env)
        self.cost_penalty = cost_penalty_weight
        self.goal_reward = goal_reward_bonus
        
        # We don't need goal_threshold anymore because we
        # can just read 'goal_achieved' from the task.
        
        self.last_goal_dist = 100.0  # Will be set in reset()
        
        print(f"Wrapper initialized with cost_penalty = {self.cost_penalty}, "
              f"goal_reward = {self.goal_reward}")

    def step(self, action):
        obs, _, cost, terminated, truncated, info = self.env.step(action)
        
        # --- THIS IS THE FIX ---
        # Read the distance and goal_met status directly
        # from the unwrapped environment's task.
        current_goal_dist = self.env.unwrapped.task.dist_goal()
        goal_met = self.env.unwrapped.task.goal_achieved
        # -----------------------
        goal_progress_reward = self.last_goal_dist - current_goal_dist
        
        # 2. Safety penalty
        safety_penalty = -cost * self.cost_penalty
        
        # 3. Goal bonus
        goal_bonus = 0.0
        if goal_met:
            goal_bonus = self.goal_reward
            
        # 4. Define total reward
        reward = goal_progress_reward + safety_penalty + goal_bonus
        
        # 5. Update state for next step
        self.last_goal_dist = current_goal_dist
        
        # 6. Update the info dict FOR THE LOGGER
        #    This ensures your console log and Monitor file get the
        #    correct values that match the GUI.
        info['cost'] = cost
        info['goal_dist'] = current_goal_dist
        info['goal_met'] = goal_met
            
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        
        # --- THIS IS THE FIX ---
        # Get the initial distance from the task
        self.last_goal_dist = self.env.unwrapped.task.dist_goal()
        # -----------------------
        
        # Update info dict for logging
        info['goal_dist'] = self.last_goal_dist
        info['cost'] = 0.0
        info['goal_met'] = False
        
        return obs, info
class PaperSTLRewardWrapper(gym.Wrapper):
    """
    Implements the STL-based reward logic as described in the paper.
    (v6: Fixes IndexError by using np.atleast_2d)

    - Atomic Goal (phi_g): d_g < r_g (distance to goal < goal radius)
      - Robustness (rho_g) = r_g - d_g
    - Atomic Safety (phi_s): d_c > r_c (distance to hazard > hazard radius)
      - Robustness (rho_s) = min_i(distance_to_hazard_i - radius_of_hazard_i)
    """
    
    def __init__(self, env, goal_radius=0.3, conj_v=1.0):
        super().__init__(env)
        self.goal_radius = goal_radius
        self.conj_v = conj_v

    def _calculate_conjunction_robustness(self, rho_list):
        """
        Implements the conjunction robustness from [25], cited by the paper.
        This metric is a weighted average to avoid the "shadow-lifting"
        problem of the standard 'min' function.
        """
        rho_array = np.array(rho_list)
        rho_min = np.min(rho_array)
        
        if self.conj_v <= 0:
             return rho_min

        if rho_min > 0:
            weights = np.exp(rho_array / self.conj_v)
        elif rho_min < 0:
            weights = np.exp(-rho_array / self.conj_v)
        else: # rho_min == 0
            return 0.0
        
        numerator = np.sum(rho_array * weights)
        denominator = np.sum(weights)
        
        if denominator == 0:
            return rho_min 
            
        return numerator / denominator

    def _get_safety_robustness(self):
        """
        Calculates the continuous safety robustness (rho_s = d_c - r_c).
        """
        # 1. Get agent's position (x, y) from the physics data
        agent_pos = self.env.unwrapped.task.data.qpos[:2]
        
        # 2. Get the Hazards object
        hazards = self.env.unwrapped.task.hazards
        
        # 3. Get the number of hazards
        num_hazards = hazards.num 
        
        if num_hazards == 0:
            return 1.0 # No hazards, so we are "safe"

        # --- THIS IS THE FIX ---
        # 4. Use np.atleast_2d() to ensure arrays are 2D before slicing.
        #    This gracefully handles both N=1 and N>1 cases.
        
        # np.atleast_2d([x,y,z]) -> [[x,y,z]]
        # np.atleast_2d([[x1,y1,z1], [x2,y2,z2]]) -> (unchanged)
        hazard_positions = np.atleast_2d(hazards.pos)[:, :2] 
        
        # np.atleast_2d([r,r,r]) -> [[r,r,r]]
        # np.atleast_2d([[r1,r1,r1], [r2,r2,r2]]) -> (unchanged)
        hazard_radii = np.atleast_2d(hazards.size)[:, 0]     
        # --- END OF FIX ---

        # 5. Calculate distances from agent to all hazards (efficiently)
        distances = np.linalg.norm(agent_pos - hazard_positions, axis=1)
        
        # 6. Calculate robustness (d - r) for all hazards
        robustnesses = distances - hazard_radii
        
        # 7. The overall safety robustness is the minimum
        return np.min(robustnesses)

    def step(self, action):
        # The base safety-gymnasium env returns 6 values
        obs, base_reward, base_cost, terminated, truncated, info = self.env.step(action)
        
        # --- 1. Get robustness for atomic safety proposition (phi_s) ---
        rho_s = self._get_safety_robustness()
        
        # --- 2. Get robustness for atomic goal proposition (phi_g) ---
        dist_goal = self.env.unwrapped.task.dist_goal()
        rho_g = self.goal_radius - dist_goal
        
        # --- 3. Calculate final reward using the paper's conjunction ---
        stl_reward = self._calculate_conjunction_robustness([rho_g, rho_s])
        
        # --- 4. Update info dict for logging ---
        info['cost'] = base_cost 
        info['goal_dist'] = dist_goal
        info['goal_met'] = self.env.unwrapped.task.goal_achieved
        info['reward_stl'] = stl_reward
        info['rho_s_atomic'] = rho_s
        info['rho_g_atomic'] = rho_g
            
        # The wrapper's step function must also return 6 values
        return obs, stl_reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        
        # Update info dict for logging
        info['goal_dist'] = self.env.unwrapped.task.dist_goal()
        info['cost'] = 0.0 # Cost is 0 at reset
        info['goal_met'] = False
        
        return obs, info

class NormalizedPaperSTLRewardWrapper(gym.Wrapper):
    """
    Implements the STL-based reward from the paper, but adds
    normalization to fix the reward scaling issue.
    
    The robustness values are scaled to a [-1, 1] range before
    being combined.
    """
    
    def __init__(self, env, goal_radius=0.3, conj_v=1.0, max_env_dist=10.0):
        super().__init__(env)
        self.goal_radius = goal_radius
        self.conj_v = conj_v
        # Estimate of the max distance from goal/hazard.
        # Tune this based on your arena size!
        self.max_dist = max_env_dist 
        
        # Pre-calculate for normalization
        self.goal_norm = self.max_dist - self.goal_radius
        self.safety_norm = self.max_dist # Assumes min hazard radius is small

    def _calculate_conjunction_robustness(self, rho_list):
        # (This function is unchanged from v6)
        rho_array = np.array(rho_list)
        rho_min = np.min(rho_array)
        
        if self.conj_v <= 0:
             return rho_min
        if rho_min > 0:
            weights = np.exp(rho_array / self.conj_v)
        elif rho_min < 0:
            weights = np.exp(-rho_array / self.conj_v)
        else: # rho_min == 0
            return 0.0
        
        numerator = np.sum(rho_array * weights)
        denominator = np.sum(weights)
        
        if denominator == 0:
            return rho_min 
        return numerator / denominator

    def _get_safety_robustness(self):
        # (This function is unchanged from v6)
        agent_pos = self.env.unwrapped.task.data.qpos[:2]
        hazards = self.env.unwrapped.task.hazards
        num_hazards = hazards.num 
        if num_hazards == 0:
            return 1.0 
        hazard_positions = np.atleast_2d(hazards.pos)[:, :2] 
        hazard_radii = np.atleast_2d(hazards.size)[:, 0]     
        distances = np.linalg.norm(agent_pos - hazard_positions, axis=1)
        robustnesses = distances - hazard_radii
        return np.min(robustnesses)

    def step(self, action):
        obs, base_reward, base_cost, terminated, truncated, info = self.env.step(action)
        
        # --- 1. Get RAW robustness values ---
        raw_rho_s = self._get_safety_robustness()
        raw_rho_g = self.goal_radius - self.env.unwrapped.task.dist_goal()
        
        # --- 2. NORMALIZE the robustness values ---
        # Scale rho_g: [goal_radius, -(max_dist-goal_radius)] -> [~0, -max_dist]
        # We normalize it to approx [1, -1]
        norm_rho_g = raw_rho_g / self.goal_norm
        
        # Scale rho_s: [-hazard_radius, max_dist]
        # We normalize it to approx [-1, 1]
        # We can just divide by max_dist as a simple approximation
        norm_rho_s = raw_rho_s / self.safety_norm
        
        # Clip to ensure they are strictly in the [-1, 1] range
        norm_rho_g = np.clip(norm_rho_g, -1.0, 1.0)
        norm_rho_s = np.clip(norm_rho_s, -1.0, 1.0)
        
        # --- 3. Calculate final reward from NORMALIZED values ---
        stl_reward = self._calculate_conjunction_robustness([norm_rho_g, norm_rho_s])
        
        # --- 4. Update info dict for logging ---
        info['cost'] = base_cost 
        info['goal_dist'] = self.env.unwrapped.task.dist_goal()
        info['goal_met'] = self.env.unwrapped.task.goal_achieved
        info['reward_stl'] = stl_reward
        # Log both raw and normalized values for debugging
        info['rho_s_raw'] = raw_rho_s
        info['rho_g_raw'] = raw_rho_g
        info['rho_s_norm'] = norm_rho_s
        info['rho_g_norm'] = norm_rho_g
            
        return obs, stl_reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        # (This function is unchanged from v6)
        obs, info = self.env.reset(*args, **kwargs)
        info['goal_dist'] = self.env.unwrapped.task.dist_goal()
        info['cost'] = 0.0
        info['goal_met'] = False
        return obs, info

TRAIN_LOG_DIR = "./ppo_safety_logs/"
os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode=None)
env = NormalizedPaperSTLRewardWrapper(env)
# 4. ADDED: Use Monitor wrapper for training logs
#    This saves episode reward, length, time, and cost to a .csv file
env = Monitor(env, TRAIN_LOG_DIR, info_keywords=("cost",))

# --- Train PPO ---
# Set a random seed for reproducibility
model = PPO("MlpPolicy", env, verbose=1, seed=42)
print("--- STARTING TRAINING ---")
# Using 2e4 for a quicker test, change back to 2e5 for full run
model.learn(total_timesteps=2e6) 
model.save(f"{TRAIN_LOG_DIR}/ppo_pointgoal_stl")
print("--- TRAINING COMPLETE ---")


# --- Test policy ---
print("--- STARTING TEST ---")
env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode="human")
env = STLRewardWrapper(env) # Use the same wrapper for testing
obs, info = env.reset(seed=42)
terminated = truncated = False
step = 0
total_reward = 0
cumulative_cost = 0

while not (terminated or truncated): # Increased test steps
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    cumulative_cost += info.get('cost', 0.0)
    total_reward += reward
    
    log_msg = (
        f"Step {step:03d} | "
        f"STL Reward: {reward:6.3f} | "
        f"Total Reward: {total_reward:6.3f} | "
        f"Step Cost: {info.get('cost', 0):.1f} | "
        f"Cumulative Cost: {cumulative_cost:.1f} | "
        f"Goal Dist: {info.get('goal_dist', 0):.3f}"
    )
    
    # 5. ADDED: Log to both file and console
    logging.info(log_msg)     # Writes to 'test_run.log'
    console_logger.info(log_msg) # Writes to console
    
    time.sleep(0.05) # Reduced sleep time for faster testing
    step += 1

env.close()
print("--- TEST COMPLETE ---")
print(f"Test results saved to 'test_run.log'")
print(f"Training statistics saved in '{TRAIN_LOG_DIR}'")