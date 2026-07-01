import unittest
import numpy as np
import torch
import pytest
from rl.evaluation.types import EvalCondition
from rl.evaluation.router_ablation import configure_condition
from plot.eval_rollout import run_eval_episodes

class DummySpace:
    def __init__(self, shape):
        self.shape = shape

class MockVecEnv:
    def __init__(self, n_envs=1, max_steps=10):
        self.observation_space = type("DummySpace", (object,), {
            "spaces": {
                "grid": DummySpace((1, 1, 10, 10)),
                "vec": DummySpace((1, 10))
            }
        })()
        self.action_space = type("DummySpace", (object,), {
            "nvec": [4]
        })()
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.steps = 0
        
    def reset(self):
        self.steps = 0
        return {
            "grid": np.zeros((self.n_envs, 1, 10, 10), dtype=np.float32),
            "vec": np.zeros((self.n_envs, 10), dtype=np.float32),
            "agent_mask": np.ones((self.n_envs, 4), dtype=np.float32),
            "mask": np.ones((self.n_envs, 4), dtype=np.float32),
        }
        
    def step_async(self, actions):
        pass
        
    def step_wait(self):
        self.steps += 1
        done = np.array([self.steps >= self.max_steps] * self.n_envs)
        obs = self.reset() if done.any() else {
            "grid": np.zeros((self.n_envs, 1, 10, 10), dtype=np.float32),
            "vec": np.zeros((self.n_envs, 10), dtype=np.float32),
            "agent_mask": np.ones((self.n_envs, 4), dtype=np.float32),
            "mask": np.ones((self.n_envs, 4), dtype=np.float32),
        }
        rew = np.zeros((self.n_envs,))
        infos = [{"episode_result": {"blue_score": 10, "red_score": 5, "decision_steps": self.steps}}] * self.n_envs
        return obs, rew, done, infos

    def env_method(self, name, *args, **kwargs):
        if name == "get_opponent_key":
            return ["OP5"]
        return [None]


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.uses_latent_strategy = True
        self.latent_k = 4
        self.strategy_interval = 2
        self.eval_allow_switching = True
        self.eval_selection_rule = "qphi"
        self.fixed_latent_strategy = False
        self.fixed_latent_strategy_id = 0
        self.latent_eval_mode = "normal"
        self.opportunity_trace_log = []
        self._opportunity_counter = np.zeros((1,), dtype=np.int64)
        
    def reset_strategy(self, done_mask=None):
        if done_mask is None:
            self._opportunity_counter.fill(0)
            self.opportunity_trace_log = []
            
    def set_sampling_generators(self, strategy=None, action=None):
        pass
        
    def set_latent_eval_mode(self, mode, seed=None):
        self.latent_eval_mode = mode
        
    def set_eval_episode_context(self, opponent, eval_seed, environment_seed, env_index=0):
        pass
        
    def set_current_decision_step(self, step):
        pass
        
    def predict(self, obs, deterministic=True):
        # Log opportunity trace
        self.opportunity_trace_log.append({
            "opponent": "OP5",
            "seed": 2000,
            "environment_seed": 2000,
            "episode_index": 0,
            "opportunity_index": len(self.opportunity_trace_log),
            "step": len(self.opportunity_trace_log) * 2,
            "logits": [0.0]*4,
            "probabilities": [0.25]*4,
            "selected_z": 0,
            "prev_z": -1,
            "switch_occurred": 0
        })
        return torch.zeros((1, 4), dtype=torch.long), None

    def strategy_info(self):
        return {
            "strategy": 0,
            "strategy_entropy": 1.38,
            "strategy_k": 4,
            "strategy_resampled": True,
        }


class TestOrderInvariance(unittest.TestCase):
    def test_condition_order_invariance(self):
        # Define condition order A and B
        order_a = [
            EvalCondition("qphi_initial_only_no_switch", "qphi", 0, False),
            EvalCondition("shuffled_qphi_outputs", "shuffled_qphi", 2, True),
            EvalCondition("uniform_episode_fixed", "uniform", 0, False),
            EvalCondition("uniform_random_at_router_opportunities", "uniform", 2, True),
        ]
        order_b = list(reversed(order_a))
        
        env = MockVecEnv()
        
        results_a = {}
        traces_a = {}
        for cond in order_a:
            model = MockModel()
            configure_condition(model, cond)
            episodes = run_eval_episodes(
                "dummy_path", env, 1, "cpu", "OP5",
                deterministic=True,
                logical_eval_seed=2000,
                preloaded_model=model,
                expected_strategy_interval=cond.strategy_interval,
                expected_allow_switching=cond.allow_switching,
                condition_name=cond.name,
                checkpoint_name="dummy_checkpoint"
            )
            results_a[cond.name] = episodes[0]
            traces_a[cond.name] = list(model.opportunity_trace_log)
            
        results_b = {}
        traces_b = {}
        for cond in order_b:
            model = MockModel()
            configure_condition(model, cond)
            episodes = run_eval_episodes(
                "dummy_path", env, 1, "cpu", "OP5",
                deterministic=True,
                logical_eval_seed=2000,
                preloaded_model=model,
                expected_strategy_interval=cond.strategy_interval,
                expected_allow_switching=cond.allow_switching,
                condition_name=cond.name,
                checkpoint_name="dummy_checkpoint"
            )
            results_b[cond.name] = episodes[0]
            traces_b[cond.name] = list(model.opportunity_trace_log)
            
        for name in results_a.keys():
            res_a = results_a[name]
            res_b = results_b[name]
            
            self.assertEqual(res_a["success"], res_b["success"])
            self.assertEqual(res_a["steps"], res_b["steps"])
            self.assertAlmostEqual(res_a["return"], res_b["return"], places=7)
            
            # Compare selection trace logs:
            trace_a_sel = [entry["selected_z"] for entry in traces_a[name]]
            trace_b_sel = [entry["selected_z"] for entry in traces_b[name]]
            self.assertEqual(trace_a_sel, trace_b_sel)

if __name__ == "__main__":
    unittest.main()
