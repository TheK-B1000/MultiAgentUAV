"""
Module: decision_proximal_features.py

This module computes instantaneous geometric and tactical features from the 
environment core at a single decision step. 

It is designed for use in the C3 decision-proximal discovery pipeline, replacing 
aggregate-fraction features with instantaneous decision-proximal geometry. This 
shift is motivated by C1 and C2 rejections which highlighted that episode-level 
aggregates mask crucial split-second tactical transitions (e.g., carrier-pressure 
onsets, intercept margins).

The DecisionProximalExtractor tracks minimal state (e.g., previous positions 
for velocity estimation) to compute dynamic features like relative closing 
velocity and pressure trends.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import torch
import math

# Default constants
ESCORT_RADIUS_FRAC = 0.22
PRESSURE_RADIUS_FRAC = 0.18
DEFAULT_SPEED = 0.15

def _np(x):
    """Bridge pattern to convert torch tensors to numpy arrays safely."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

@dataclasses.dataclass
class DecisionProximalFeatures:
    """
    Features representing the instantaneous geometric/tactical state at a 
    single decision step, centered around flag carrier dynamics.
    """
    time_to_intercept: float
    relative_closing_velocity: float
    carrier_dist_home: float
    nearest_ready_defender_dist: float
    escort_dist: float
    cooldown_remaining: float
    carrier_progress_frac: float
    pressure_trend: float
    commitment_imbalance: float
    mate_intervention_eta: float
    intercept_margin: float
    agents_forward: int
    formation_spread: float
    is_carrier_pressure_onset: bool
    score_diff: float
    time_remaining_frac: float

class DecisionProximalExtractor:
    """
    Extracts DecisionProximalFeatures from a BatchedCTFCore instance (B=1).
    Maintains per-step state to compute derivatives like velocity and pressure trend.
    """
    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self):
        """Clears all internal state. Must be called at episode start."""
        self._prev_red_pos = None          # np.ndarray shape (N, 2)
        self._prev_blue_pos = None         # np.ndarray shape (N, 2)
        self._prev_carrier_pressure = 0.0
        self._pressure_ema = 0.0
        self._prev_carrying = None         # np.ndarray shape (N,) bool
        self._prev_under_pressure = False

    def extract(self, core) -> DecisionProximalFeatures:
        """
        Reads core state and returns DecisionProximalFeatures.
        Assumes core has batch size B=1.
        """
        # Extract basic arrays and convert to numpy
        b_x = _np(core.blue_x)[0]
        b_y = _np(core.blue_y)[0]
        r_x = _np(core.red_x)[0]
        r_y = _np(core.red_y)[0]
        
        b_alive = _np(core.blue_alive)[0].astype(bool)
        r_alive = _np(core.red_alive)[0].astype(bool)
        b_carrying = _np(core.blue_carrying)[0].astype(bool)
        
        r_tag_cd = _np(core.red_tag_cooldown)[0]
        b_tag_cd = _np(core.blue_tag_cooldown)[0]
        
        b_home = _np(core.blue_flag_home)[0]
        
        cols = float(core.cols)
        
        b_pos = np.stack([b_x, b_y], axis=-1)
        r_pos = np.stack([r_x, r_y], axis=-1)
        
        N = len(b_x)
        
        # Initialize default NaN values for carrier-specific features
        time_to_intercept = float('nan')
        rel_closing_velocity = float('nan')
        carrier_dist_home = float('nan')
        nearest_ready_def_dist = float('nan')
        escort_dist = float('nan')
        cd_remaining = float('nan')
        carrier_progress_frac = float('nan')
        mate_interv_eta = float('nan')
        intercept_margin = float('nan')
        is_pressure_onset = False
        
        carrier_idx = np.argmax(b_carrying) if np.any(b_carrying) else -1
        
        current_carrier_pressure = 0.0
        currently_under_pressure = False
        
        if carrier_idx >= 0:
            c_pos = b_pos[carrier_idx]
            
            # Carrier to home distance
            dist_home = np.linalg.norm(c_pos - b_home) / cols
            carrier_dist_home = dist_home
            carrier_progress_frac = max(0.0, 1.0 - dist_home)
            cd_remaining = float(b_tag_cd[carrier_idx])
            
            # Nearest red to carrier
            min_r_dist = float('inf')
            nearest_r_idx = -1
            for i in range(N):
                if r_alive[i]:
                    d = np.linalg.norm(r_pos[i] - c_pos) / cols
                    if d < min_r_dist:
                        min_r_dist = d
                        nearest_r_idx = i
            
            # Escort logic
            min_mate_dist = float('inf')
            for i in range(N):
                if i != carrier_idx and b_alive[i]:
                    d = np.linalg.norm(b_pos[i] - c_pos) / cols
                    if d < min_mate_dist:
                        min_mate_dist = d
            
            escort_dist = min_mate_dist
            if np.isinf(min_mate_dist):
                mate_interv_eta = float('inf')
            else:
                mate_interv_eta = escort_dist / DEFAULT_SPEED
                
            # Nearest ready defender
            min_ready_dist = float('inf')
            for i in range(N):
                if r_alive[i] and r_tag_cd[i] <= 0:
                    d = np.linalg.norm(r_pos[i] - c_pos) / cols
                    if d < min_ready_dist:
                        min_ready_dist = d
            nearest_ready_def_dist = min_ready_dist if not np.isinf(min_ready_dist) else float('nan')
            
            # Closing velocity & time to intercept
            if nearest_r_idx >= 0:
                d_nearest = min_r_dist
                r_vel = np.zeros(2)
                if self._prev_red_pos is not None:
                    # simplistic velocity: dp/dt, normalized by map size
                    r_vel = (r_pos[nearest_r_idx] - self._prev_red_pos[nearest_r_idx]) / cols
                
                # Direction vector from red to carrier
                r_to_c = (c_pos - r_pos[nearest_r_idx]) / cols
                norm_r_to_c = np.linalg.norm(r_to_c)
                
                if norm_r_to_c > 1e-6:
                    r_to_c_dir = r_to_c / norm_r_to_c
                    # closing vel: positive if red is moving towards carrier
                    closing = float(np.dot(r_vel, r_to_c_dir))
                else:
                    closing = 0.0
                
                rel_closing_velocity = closing
                
                if closing > 0:
                    time_to_intercept = d_nearest / closing
                else:
                    time_to_intercept = (d_nearest / DEFAULT_SPEED) + 20.0
                
                intercept_margin = time_to_intercept - mate_interv_eta
                
                current_carrier_pressure = 1.0 / (d_nearest + 1e-3)
                if d_nearest <= PRESSURE_RADIUS_FRAC:
                    currently_under_pressure = True
            else:
                rel_closing_velocity = 0.0
                time_to_intercept = float('inf')
                intercept_margin = float('inf')
            
            # Onset detection
            just_picked_up = (self._prev_carrying is not None) and (not self._prev_carrying[carrier_idx])
            became_pressured = currently_under_pressure and (not self._prev_under_pressure)
            
            is_pressure_onset = bool(just_picked_up or became_pressured)
        
        # Pressure Trend (EMA derivative)
        pressure_diff = current_carrier_pressure - self._prev_carrier_pressure
        self._pressure_ema = self.ema_alpha * pressure_diff + (1 - self.ema_alpha) * self._pressure_ema
        pressure_trend = self._pressure_ema
        
        # Update state buffers
        self._prev_red_pos = r_pos
        self._prev_blue_pos = b_pos
        self._prev_carrier_pressure = current_carrier_pressure
        self._prev_under_pressure = currently_under_pressure
        self._prev_carrying = b_carrying
        
        # Commitment imbalance
        # Attackers: blue agents on red half (x > cols/2)
        # Defenders: blue agents on blue half (x <= cols/2)
        n_attackers = sum(1 for i in range(N) if b_alive[i] and b_x[i] > cols / 2)
        n_defenders = sum(1 for i in range(N) if b_alive[i] and b_x[i] <= cols / 2)
        commitment_imbalance = float(abs(n_attackers - n_defenders))
        
        agents_forward = n_attackers
        
        # Formation spread
        live_blue_pos = [b_pos[i] for i in range(N) if b_alive[i]]
        if len(live_blue_pos) > 1:
            ys = [p[1] for p in live_blue_pos]
            formation_spread = (max(ys) - min(ys)) / cols
        else:
            formation_spread = 0.0
            
        # Score and Time
        try:
            score_diff = float(_np(core.blue_score)[0] - _np(core.red_score)[0])
        except AttributeError:
            score_diff = 0.0
            
        step_count = float(_np(core.step_count)[0])
        max_steps = float(core.cfg.max_decision_steps)
        time_rem = max(0.0, 1.0 - (step_count / max_steps))
        
        return DecisionProximalFeatures(
            time_to_intercept=time_to_intercept,
            relative_closing_velocity=rel_closing_velocity,
            carrier_dist_home=carrier_dist_home,
            nearest_ready_defender_dist=nearest_ready_def_dist,
            escort_dist=escort_dist,
            cooldown_remaining=cd_remaining,
            carrier_progress_frac=carrier_progress_frac,
            pressure_trend=pressure_trend,
            commitment_imbalance=commitment_imbalance,
            mate_intervention_eta=mate_interv_eta,
            intercept_margin=intercept_margin,
            agents_forward=agents_forward,
            formation_spread=formation_spread,
            is_carrier_pressure_onset=is_pressure_onset,
            score_diff=score_diff,
            time_remaining_frac=time_rem
        )
