from __future__ import annotations

import unittest

from rl.evaluation.episode_runner import EpisodeRunnerRuntime, run_episode


class EvaluationEpisodeRunnerTests(unittest.TestCase):
    def test_run_episode_restores_model_training_state_and_row_schema(self) -> None:
        class Model:
            def __init__(self):
                self.training = True
            def eval(self):
                self.training = False
            def train(self, value=True):
                self.training = value
        class Env:
            def __init__(self):
                self.closed = False
            def reset(self):
                return {"obs": 1}
            def step(self, action):
                return {"obs": 2}, 0.0, True, {"info": 1}
            def metrics(self, info):
                return {"episode_steps": 1}
            def close(self):
                self.closed = True
        model = Model()
        env = Env()
        runtime = EpisodeRunnerRuntime(
            adapt_obs_for_policy=lambda obs, policy: obs,
            done=lambda value: bool(value),
            first_info=lambda infos: infos,
            get_opponent_key=lambda env: "OP8",
            make_env=lambda **kwargs: env,
            model=lambda policy: model,
            predict=lambda policy, obs: 0,
            reset_obs=lambda value: value,
            scores=lambda env, info: (1.0, 0.0),
            set_opponent=lambda env, opponent: opponent,
            unpack_step=lambda value: value,
            validate_opponent_name=lambda opponent: opponent,
        )
        row = run_episode(runtime=runtime, policy=object(), policy_name="candidate", map_name="map_a_open", opponent="OP8", seed=7, device="cpu", n_agents=2, max_steps=1)
        self.assertEqual(row["policy"], "candidate")
        self.assertEqual(row["resolved_opponent"], "OP8")
        self.assertEqual(row["win"], 1)
        self.assertTrue(model.training)
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
