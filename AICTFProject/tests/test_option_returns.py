"""Focused tests for ``rl.custom_ppo.option_returns.compute_option_returns``.

These tests intentionally do **not** touch ``CustomPPOTrainer`` or any env —
the whole point of extracting this recursion was to make it trivially
testable. Each test sets up a tiny ``(T, N)`` rollout by hand and asserts
the option-return / option-advantage tensor matches a hand-computed
discounted sum with explicit boundary handling.
"""

from __future__ import annotations

import unittest

import torch

from rl.custom_ppo.option_returns import compute_option_returns


def _b(values: list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _f(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class ComputeOptionReturnsTests(unittest.TestCase):
    def test_continues_until_next_z_boundary(self) -> None:
        """No env-level dones, but z is resampled at t=2 → option ends there.

        Single env, T=4, rewards=[1, 2, 3, 4], gamma=0.5.
        z_resampled = [True, False, True, False]  → option windows: {0, 1} and {2, 3}.

        Window 1 (t=0..1): return at t=1 bootstraps from V(s_2, z_2)=values[2]=100,
            so option_returns[1] = 2 + 0.5 * 100 = 52
            option_returns[0] = 1 + 0.5 * 52 = 27.
        Window 2 (t=2..3): t=3 is the last buffered step (no done), bootstraps
            from next_values[3]=50,
            so option_returns[3] = 4 + 0.5 * 50 = 29
            option_returns[2] = 3 + 0.5 * 29 = 17.5.
        """
        rewards = _f([[1.0], [2.0], [3.0], [4.0]])
        values = _f([[10.0], [20.0], [100.0], [40.0]])
        next_values = _f([[20.0], [100.0], [40.0], [50.0]])
        terminated = _b([[0], [0], [0], [0]])
        truncated = _b([[0], [0], [0], [0]])
        z_resampled = _b([[1], [0], [1], [0]])

        returns, advantages = compute_option_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            z_resampled=z_resampled,
            gamma=0.5,
        )

        expected_returns = _f([[27.0], [52.0], [17.5], [29.0]])
        torch.testing.assert_close(returns, expected_returns)
        torch.testing.assert_close(advantages, expected_returns - values)

    def test_zero_bootstrap_on_termination(self) -> None:
        """Termination at step 2 → option_returns[2] = reward only, no bootstrap.

        Single env, T=4, gamma=0.9.
        terminated = [0, 0, 1, 0], truncated = [0, 0, 0, 0].
        z_resampled = all False (option spans whole rollout) BUT termination
        forces the boundary anyway.

        - t=3 (no done, last step) → bootstrap next_values[3]=10:
            returns[3] = 5 + 0.9 * 10 = 14
        - t=2 (terminated) → next_val = 0:
            returns[2] = 3 + 0.9 * 0 = 3       (THE key property)
        - t=1 (no done) → carry = returns[2] = 3 (option continues into t=2):
            returns[1] = 2 + 0.9 * 3 = 4.7
        - t=0 (no done) → carry = returns[1] = 4.7:
            returns[0] = 1 + 0.9 * 4.7 = 5.23
        """
        rewards = _f([[1.0], [2.0], [3.0], [5.0]])
        values = _f([[0.0], [0.0], [0.0], [0.0]])
        next_values = _f([[0.0], [0.0], [999.0], [10.0]])
        terminated = _b([[0], [0], [1], [0]])
        truncated = _b([[0], [0], [0], [0]])
        z_resampled = _b([[0], [0], [0], [0]])

        returns, _adv = compute_option_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            z_resampled=z_resampled,
            gamma=0.9,
        )

        expected = _f([[5.23], [4.7], [3.0], [14.0]])
        torch.testing.assert_close(returns, expected, atol=1e-6, rtol=0)

    def test_bootstrap_on_truncation(self) -> None:
        """Truncation at step 2 → bootstrap from V(s'), not from carry.

        Single env, T=4, gamma=1.0 (easy arithmetic).
        truncated = [0, 0, 1, 0], terminated = [0, 0, 0, 0].
        z_resampled = all False (would otherwise extend the option).

        - t=3 (no done, last) → bootstrap next_values[3]=4: returns[3] = 5 + 4 = 9
        - t=2 (truncated, not terminated) → bootstrap next_values[2]=7:
            returns[2] = 3 + 7 = 10            (THE key property)
        - t=1: returns[1] = 2 + returns[2] = 12
        - t=0: returns[0] = 1 + 12 = 13
        """
        rewards = _f([[1.0], [2.0], [3.0], [5.0]])
        values = _f([[0.0], [0.0], [0.0], [0.0]])
        next_values = _f([[0.0], [0.0], [7.0], [4.0]])
        terminated = _b([[0], [0], [0], [0]])
        truncated = _b([[0], [0], [1], [0]])
        z_resampled = _b([[0], [0], [0], [0]])

        returns, _adv = compute_option_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            z_resampled=z_resampled,
            gamma=1.0,
        )

        expected = _f([[13.0], [12.0], [10.0], [9.0]])
        torch.testing.assert_close(returns, expected)

    def test_vectorized_multi_env_branches_dont_crash(self) -> None:
        """Regression: with n_envs > 1 the inner conditional must use torch.where.

        This is exactly the bug PR-3 protects against: an earlier inline
        version branched on ``if done_t:`` which raised
        ``RuntimeError: Boolean value of Tensor with more than one value is
        ambiguous`` once ``n_envs > 1``.

        Each env hits a different boundary type at the same timestep:
            env 0: terminates at t=1 (zero bootstrap)
            env 1: truncates at t=1 (bootstrap V(s'))
            env 2: nothing at t=1 (carries forward)
            env 3: z is resampled at t=2 (option boundary)
        """
        T, N = 3, 4
        rewards = _f([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ])
        values = _f([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
        ])
        next_values = _f([
            [0.0, 0.0, 0.0, 0.0],
            [99.0, 5.0, 0.0, 0.0],
            [7.0, 7.0, 7.0, 7.0],
        ])
        terminated = _b([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        truncated = _b([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ])
        z_resampled = _b([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],   # env 3: new z sampled at t=2
        ])

        returns, advantages = compute_option_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            z_resampled=z_resampled,
            gamma=1.0,
        )

        self.assertEqual(tuple(returns.shape), (T, N))
        self.assertEqual(tuple(advantages.shape), (T, N))
        # All four envs: last step is no-done last-row → bootstrap next_values[2]=7.
        # returns[2, :] = 3 + 7 = 10.
        torch.testing.assert_close(returns[2], _f([[10.0, 10.0, 10.0, 10.0]])[0])
        # env 0 at t=1 (terminated): returns[1, 0] = 2 + 0 = 2.
        self.assertAlmostEqual(float(returns[1, 0]), 2.0, places=5)
        # env 1 at t=1 (truncated, not terminated): bootstrap next_values=5.
        # returns[1, 1] = 2 + 5 = 7.
        self.assertAlmostEqual(float(returns[1, 1]), 7.0, places=5)
        # env 2 at t=1 (no done, no boundary): carry = returns[2, 2] = 10.
        # returns[1, 2] = 2 + 10 = 12.
        self.assertAlmostEqual(float(returns[1, 2]), 12.0, places=5)
        # env 3 at t=1 (no done, but z_resampled at t=2): bootstrap values[2, 3]=10.
        # returns[1, 3] = 2 + 10 = 12. (same arithmetic as env 2 since values==returns[2] here.)
        self.assertAlmostEqual(float(returns[1, 3]), 12.0, places=5)

    def test_advantages_equal_returns_minus_values(self) -> None:
        torch.manual_seed(0)
        T, N = 8, 3
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        next_values = torch.randn(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        z_resampled = torch.zeros(T, N, dtype=torch.bool)

        returns, advantages = compute_option_returns(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            z_resampled=z_resampled,
            gamma=0.99,
        )
        torch.testing.assert_close(advantages, returns - values)

    def test_rejects_non_2d_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_option_returns(
                rewards=torch.zeros(4),
                values=torch.zeros(4),
                next_values=torch.zeros(4),
                terminated=torch.zeros(4, dtype=torch.bool),
                truncated=torch.zeros(4, dtype=torch.bool),
                z_resampled=torch.zeros(4, dtype=torch.bool),
                gamma=0.99,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
