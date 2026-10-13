"""Lightweight tests for the isolated HoST safe-recovery reward delta."""

import math
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import torch
from minipi_getup.host_safe_recovery.asset import ASSET_XML, EXPECTED_ASSET_SHA256
from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg
from minipi_getup.host_safe_recovery.env import (
  SafeRecoveryEnv,
  _com_ascent_cost,
  _contact_tangential_speed_cost,
  _foot_support_score,
  _knee_sole_progress,
  _loaded_coverage_deficit,
  _loaded_hip_roll_excess,
  _loaded_knee_flexion,
  _loaded_knee_hyperextension,
  _quat_mul,
  _rotation_speed_cost,
  _saturating_support,
  _sole_contact_coverage,
  _sole_tilt_cost,
)
from minipi_getup.host_safe_recovery.optimization import install_action_std_guard


class RewardTests(unittest.TestCase):
  def test_coverage_deficit_requires_same_foot_load_and_is_squared(self):
    coverage = torch.tensor([[0.,1.], [.25,1.], [.5,.5], [0.,0.]])
    support = torch.tensor([[1.,0.], [1.,0.], [1.,1.], [0.,0.]])
    cost = _loaded_coverage_deficit(coverage, support, .5)
    self.assertTrue(torch.allclose(cost, torch.tensor([.5,.125,0.,0.])))
    self.assertTrue(torch.isfinite(cost).all())
    self.assertLess(SafeRecoveryCfg.constraints.scales.style_loaded_sole_coverage_deficit, 0)

  def test_sole_tilt_cost_is_zero_flat_and_grows_for_roll_and_pitch(self):
    def axis_quat(axis, angle):
      q = torch.zeros(4)
      q[0] = math.cos(angle / 2)
      q[1 + axis] = math.sin(angle / 2)
      return q
    yaw = axis_quat(2, 1.0)
    quats = torch.stack([yaw, axis_quat(0, .05), axis_quat(0, .3), axis_quat(1, .3), axis_quat(1, 1.4)])
    cost = _sole_tilt_cost(quats, .15)
    self.assertAlmostEqual(float(cost[0]), 0.0, places=6)
    self.assertLess(float(cost[1]), .2)
    self.assertTrue(torch.allclose(cost[2], cost[3], atol=1e-5))
    self.assertGreater(float(cost[2]), .9)
    self.assertLessEqual(float(cost.max()), 1.0)
    self.assertEqual(_sole_tilt_cost(quats.view(1, 5, 4), .15).shape, (1, 5))
    self.assertLess(SafeRecoveryCfg.constraints.scales.style_loaded_sole_tilt, 0)
    self.assertTrue(callable(SafeRecoveryEnv._reward_loaded_sole_tilt))

  def test_scratch_regularization_ramps_from_original_strength(self):
    env = self.make_env()
    env.cfg = SafeRecoveryCfg()
    env.dof_vel[:] = 6.0
    env._penalty_curriculum_strength = lambda: torch.tensor(0.0)
    initial = env._reward_joint_overspeed()
    env._penalty_curriculum_strength = lambda: torch.tensor(1.0)
    full = env._reward_joint_overspeed()
    self.assertTrue(torch.allclose(full, 3 * initial))

  def test_slip_cost_uses_contact_point_and_ignores_vertical_speed(self):
    linear = torch.tensor([[0.,0.,2.], [0.,0.,0.], [0.02,0.,0.]])
    angular = torch.tensor([[0.,0.,0.], [0.,1.,0.], [0.,0.,0.]])
    offset = torch.tensor([[0.,0.,0.], [0.,0.,-.05], [0.,0.,0.]])
    cost = _contact_tangential_speed_cost(linear, angular, offset, .03, .1)
    self.assertTrue(torch.allclose(cost, torch.tensor([0.,.2,0.])))

  def test_knee_flexion_credit_and_hyperextension_cost_need_load_and_gate(self):
    calf = torch.tensor([[1., 1.], [-.65, -.65], [1., 1.], [-.2, -.2]])
    support = torch.tensor([[1., 1.], [1., 1.], [0., 0.], [1., 1.]])
    gate = torch.tensor([1., 1., 1., 1.])
    flex = _loaded_knee_flexion(calf, support, gate, .6)
    self.assertGreater(float(flex[0]), .8)
    self.assertTrue(torch.equal(flex[1:], torch.zeros(3)))
    hyper = _loaded_knee_hyperextension(calf, support, gate, .3, .35)
    self.assertTrue(torch.allclose(hyper, torch.tensor([0., 1., 0., 0.])))
    self.assertTrue(torch.equal(_loaded_knee_flexion(calf, support, torch.zeros(4), .6), torch.zeros(4)))
    scales = SafeRecoveryCfg.constraints.scales
    self.assertGreater(scales.target_loaded_knee_flexion, 0)
    self.assertLess(scales.style_loaded_knee_hyperextension, 0)
    for name in ("loaded_knee_flexion", "loaded_knee_hyperextension"):
      self.assertTrue(callable(getattr(SafeRecoveryEnv, f"_reward_{name}")))

  def test_hip_roll_excess_only_costs_wide_loaded_legs(self):
    outward = torch.tensor([[1.6, 1.6], [1.6, .3], [.5, .5], [1.6, 1.6]])
    support = torch.tensor([[1., 1.], [0., 1.], [1., 1.], [0., 0.]])
    cost = _loaded_hip_roll_excess(outward, support, .6)
    self.assertTrue(torch.allclose(cost, torch.tensor([1., 0., 0., 0.])))
    scales = SafeRecoveryCfg.constraints.scales
    self.assertLess(scales.style_loaded_hip_roll_excess, 0)
    self.assertFalse(hasattr(scales, "target_recovery_hip_roll"))
    self.assertTrue(callable(SafeRecoveryEnv._reward_loaded_hip_roll_excess))

  def test_scale_finetuning_only_reduces_successful_unassisted_envs(self):
    env = object.__new__(SafeRecoveryEnv)
    env.cfg = SafeRecoveryCfg()
    env.old_headheight = torch.zeros(3, 1)
    env.force = torch.tensor([[0.], [0.], [10.]])
    env.action_rescale = torch.tensor([[.25], [.25], [.25]])
    env.penalty_success_ema = torch.tensor(.90)
    env.penalty_stood_on_feet = torch.tensor([True, False, True])
    env.update_force_curriculum(torch.arange(3))
    self.assertTrue(torch.allclose(env.action_rescale[:,0], torch.tensor([.245,.25,.25])))
    env.action_rescale[0] = .101
    env.update_force_curriculum(torch.arange(3))
    self.assertAlmostEqual(env.action_rescale[0].item(), .10, places=6)
    env.penalty_success_ema.fill_(.50)
    env.update_force_curriculum(torch.arange(3))
    self.assertAlmostEqual(env.action_rescale[1].item(), .25, places=6)

  def test_rotation_cost_uses_all_axes_and_soft_deadband(self):
    velocity = torch.tensor([[0.,0.,0.], [1.5,0.,0.], [0.,0.,3.], [3.,0.,0.]])
    cost = _rotation_speed_cost(velocity, 1.5)
    self.assertTrue(torch.allclose(cost, torch.tensor([0.,0.,2.25,2.25])))
    self.assertEqual(SafeRecoveryCfg.constraints.scales.regu_joint_overspeed, -0.30)
    self.assertEqual(SafeRecoveryCfg.constraints.scales.regu_action_rate, -0.03)
    self.assertEqual(SafeRecoveryCfg.constraints.scales.regu_smoothness, -0.03)

  def test_immediate_com_cost_is_zero_below_threshold_and_grows_strongly(self):
    cost = _com_ascent_cost(torch.tensor([-1.0, 0.0, 0.22, 0.30, 1.0]), 0.22, 0.50)
    self.assertTrue(torch.equal(cost[:3], torch.zeros(3)))
    self.assertGreater(cost[3], 0)
    self.assertGreater(cost[4], 100 * cost[3])
    self.assertTrue(torch.isfinite(cost).all())

  def test_new_contact_rewards_are_registered_in_existing_target_group(self):
    self.assertEqual(SafeRecoveryCfg.constraints.scales.target_knee_sole_progress, 6.0)
    for name in ("sole_contact_coverage", "knee_sole_progress"):
      self.assertTrue(callable(getattr(SafeRecoveryEnv, f"_reward_{name}")))
    self.assertEqual(SafeRecoveryCfg.rewards.reward_groups, ["task", "regu", "style", "target"])

  def test_knee_progress_initial_sample_and_stationary_pose_have_no_bonus(self):
    env = object.__new__(SafeRecoveryEnv)
    env.cfg = SimpleNamespace(safe_recovery=SimpleNamespace(knee_contact_progress_scale_rad=0.04))
    env.knee_contact_indices = torch.tensor([0, 1])
    env.dof_pos = torch.tensor([[0.5, 0.5]])
    env.previous_contact_knees = torch.zeros(1, 2)
    env.previous_sole_coverage = torch.zeros(1, 2)
    env.knee_contact_progress_valid = torch.zeros(1, dtype=torch.bool)
    coverage = torch.tensor([[0.2, 0.2]])
    env._sole_coverage = lambda: coverage
    env._standing_gate = lambda: torch.zeros(1)
    self.assertEqual(env._reward_knee_sole_progress().item(), 0.0)
    env.dof_pos += 0.04
    coverage += 0.2
    self.assertAlmostEqual(env._reward_knee_sole_progress().item(), 0.2, places=5)
    self.assertEqual(env._reward_knee_sole_progress().item(), 0.0)
    env.knee_contact_progress_valid.fill_(False)
    env.dof_pos += 0.04
    coverage += 0.2
    self.assertEqual(env._reward_knee_sole_progress().item(), 0.0)

  def test_knee_bonus_requires_same_foot_contact_improvement_and_flexion(self):
    delta = torch.tensor([0.04, -0.04, 0.0, 0.04, 4.0])
    contact = torch.tensor([0.2, 0.2, 0.2, -0.2, 0.2])
    score = _knee_sole_progress(delta, contact, 0.04)
    self.assertTrue(torch.allclose(score, torch.tensor([0.2, 0.0, 0.0, 0.0, 0.2])))

  def test_sole_coverage_rewards_broad_persistent_contact_not_one_impact(self):
    history = torch.zeros(4, 5, 4, 3)
    history[1, 0, :, 2] = 1000.0  # one overloaded strip
    history[2, :, :, 2] = 2.0     # all strips, modest force
    history[3, :, 0, 2] = 1000.0 # one-substep impact
    score = _sole_contact_coverage(history, torch.ones(4))
    self.assertEqual(score[0].item(), 0.0)
    self.assertAlmostEqual(score[1].item(), 0.2, places=6)
    self.assertAlmostEqual(score[3].item(), 0.25, places=6)
    self.assertGreater(score[2], score[1])
    self.assertGreater(score[2], score[3])
    self.assertTrue(torch.isfinite(score).all())
    self.assertTrue(((score >= 0) & (score <= 1)).all())

  def test_host_like_leg_collisions(self):
    root = ET.parse(ASSET_XML).getroot()
    geoms = {geom.get("name"): geom for geom in root.findall(".//geom") if geom.get("name")}
    for side in ("l", "r"):
      thigh = geoms[f"{side}_thigh_collision"]
      self.assertEqual(thigh.get("type"), "box")
      self.assertEqual(thigh.get("size"), "0.0005 0.0005 0.0005")
      calf = geoms[f"{side}_calf_collision"]
      self.assertEqual(calf.get("type"), "box")
      self.assertEqual(calf.get("size"), "0.029 0.0325 0.0275")
      self.assertEqual(
        geoms[f"{side}_ankle_pitch_collision"].get("size"), "0.039 0.04 0.025"
      )
      self.assertEqual(
        geoms[f"{side}_ankle_roll_collision"].get("size"), "0.029 0.025 0.025"
      )
    self.assertEqual(
      EXPECTED_ASSET_SHA256,
      "a0a961cbd522d7836b48b4e77160128df0c3f640f01de19617b43657800eb533",
    )

  def test_recovery_pull_force_curriculum_is_40_to_0_by_10(self):
    self.assertTrue(SafeRecoveryCfg.curriculum.pull_force)
    self.assertEqual(SafeRecoveryCfg.curriculum.force, 40)
    self.assertEqual(SafeRecoveryCfg.safe_recovery.force_curriculum_step_n, 10.0)

  def test_force_support_is_deadbanded_monotonic_and_saturating(self):
    weight = torch.full((5,), 100.0)
    force = torch.tensor([0.0, 2.0, 12.0, 27.0, 1000.0])
    support = _saturating_support(force, weight, 0.02, 0.25)
    self.assertEqual(support[0], 0)
    self.assertEqual(support[1], 0)
    self.assertTrue(torch.all(support[1:] >= support[:-1]))
    self.assertTrue(torch.all((support >= 0) & (support <= 1)))
    self.assertGreater(support[3], support[2])
    self.assertGreater(support[-1], 0.99)

  def test_foot_score_allows_one_foot_and_rewards_two(self):
    zero = torch.tensor([0.0])
    partial = torch.tensor([0.7])
    one_foot = _foot_support_score(partial, zero)
    two_feet = _foot_support_score(partial, partial)
    self.assertGreater(one_foot, zero)
    self.assertGreater(two_feet, one_foot)
    self.assertEqual(_foot_support_score(torch.ones(1), torch.ones(1)), 1.5)

  def test_force_curriculum_decrements_by_10_and_clamps_at_zero(self):
    env = object.__new__(SafeRecoveryEnv)
    env.cfg = SimpleNamespace(
      curriculum=SimpleNamespace(threshold_height=0.37),
      safe_recovery=SafeRecoveryCfg.safe_recovery,
    )
    env.old_headheight = torch.full((4, 1), 0.5)
    env.penalty_success_ema = torch.tensor(0.0)
    env.force = torch.tensor([30.0, 20.0, 10.0, 0.0])
    env.action_rescale = torch.ones(4)
    ids = torch.arange(4)
    env.update_force_curriculum(ids)
    self.assertTrue(torch.equal(env.force, torch.tensor([20.0, 10.0, 0.0, 0.0])))
    self.assertTrue(torch.allclose(env.action_rescale, torch.full((4,), 0.98)))

  def make_env(self):
    env = object.__new__(SafeRecoveryEnv)
    env.cfg = SimpleNamespace(
      safe_recovery=SimpleNamespace(
        overspeed_soft_rad_s=4.0,
        gate_height_center=0.25,
        gate_height_width=0.045,
        gate_upright_center=0.65,
        gate_upright_width=0.12,
        stable_joint_velocity_weight=1.0,
        stable_base_angular_velocity_weight=0.25,
        standing_posture_penalty_multiplier=2.0,
        penalty_initial_regu_fraction=1.0/3.0,
        standing_motion_penalty_multiplier=2.0,
        standing_ascent_penalty_multiplier=3.0,
      )
    )
    env.dof_vel = torch.zeros(2, 12)
    env.dof_pos = torch.zeros(2, 12)
    env.default_dof_pos = torch.zeros(1, 12)
    env.pose_sigma = torch.ones(1, 12)
    env.pose_weight = torch.ones(1, 12)
    env.root_states = torch.zeros(2, 13)
    env.projected_gravity = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    env.base_ang_vel = torch.zeros(2, 3)
    env.support_memory = torch.zeros(2, 2)
    env._penalty_curriculum_strength = lambda: torch.tensor(1.0)
    env._normalized_supports = lambda: (torch.zeros(2), torch.zeros(2), torch.zeros(2))
    return env

  def test_penalty_success_curriculum_is_smooth_and_bounded(self):
    env = object.__new__(SafeRecoveryEnv)
    env.cfg = SafeRecoveryCfg()
    for success, expected in ((0.0, 0.0), (0.60, 0.0), (0.725, 0.5), (0.85, 1.0), (1.0, 1.0)):
      env.penalty_success_ema = torch.tensor(success)
      self.assertAlmostEqual(env._penalty_curriculum_strength().item(), expected, places=5)

  def test_standing_penalty_boost_requires_sustained_foot_support(self):
    env = self.make_env()
    env._standing_gate = lambda: torch.ones(2)
    env._normalized_supports = lambda: (torch.ones(2), torch.zeros(2), torch.zeros(2))
    env.support_memory[1, 0] = 1.0
    self.assertTrue(torch.equal(env._standing_penalty_boost(2.0), torch.tensor([1.0, 2.0])))
    env.dof_pos[:] = 0.2
    env.dof_vel[:] = 0.3
    env.pose_weight = torch.ones(1, 12)
    env.ascent_cost_history = torch.ones(2, 4)
    posture = env._reward_default_posture()
    motion = env._reward_stable_standing()
    self.assertAlmostEqual((posture[1] / posture[0]).item(), 2.0)
    self.assertAlmostEqual((motion[1] / motion[0]).item(), 2.0)
    self.assertTrue(torch.equal(env._reward_ballistic_ascent(), torch.ones(2)))
    env._standing_gate = lambda: torch.zeros(2)
    self.assertTrue(torch.equal(env._standing_penalty_boost(3.0), torch.ones(2)))

  def test_overspeed_threshold_and_mean_max_square(self):
    env = self.make_env()
    env.dof_vel[0, 0] = 4.0
    env.dof_vel[1, 0] = -6.0
    cost = env._reward_joint_overspeed()
    self.assertEqual(cost[0], 0)
    expected = 0.5 * (4.0 / 12.0) + 0.5 * 4.0
    self.assertTrue(torch.allclose(cost[1], torch.tensor(expected)))

  def test_single_joint_spike_is_penalized_per_environment(self):
    env = self.make_env()
    env.dof_vel[0, 3] = 10.0
    env.dof_vel[1, :] = 5.0
    cost = env._reward_joint_overspeed()
    old_mean_only_cost = 36.0 / 12.0
    self.assertGreater(cost[0], old_mean_only_cost * 6.0)
    self.assertTrue(torch.allclose(cost[1], torch.tensor(1.0)))
    self.assertEqual(cost.shape, torch.Size([2]))
    self.assertTrue(torch.isfinite(cost).all())

  def test_sustained_support_increases_height_credit_without_removing_early_signal(self):
    env = self.make_env()
    env.cfg.safe_recovery.foot_height_center = 0.18
    env.cfg.safe_recovery.foot_height_width = 0.06
    env.root_states[:, 2] = 0.25
    env._normalized_supports = lambda: (torch.ones(2), torch.ones(2), torch.zeros(2))
    env.support_memory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    reward = env._reward_foot_supported_height()
    self.assertGreater(reward[0], 0)
    self.assertTrue(torch.allclose(reward[1], 5 * reward[0]))

  def test_ballistic_cost_averages_substeps_per_environment(self):
    env = self.make_env()
    env.ascent_cost_history = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    self.assertTrue(torch.equal(env._reward_ballistic_ascent(), torch.tensor([0.0, 1.0])))

  def test_knee_cost_filters_other_geoms_and_increases_during_lift(self):
    env = self.make_env()
    env.cfg = SafeRecoveryCfg()
    history = torch.zeros(2, 8, 4, 3)
    env.nonfoot_ground_sensor = SimpleNamespace(data=SimpleNamespace(force_history=history))
    env.knee_force_indices = torch.tensor([1, 5])
    env.robot = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([0])))
    env.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(2, 1)))
    env.root_states[:, 2] = 0.08
    history[:, 0, :, 2] = 100.0
    self.assertTrue(torch.equal(env._reward_knee_load_support(), torch.zeros(2)))
    history[:, 1, :, 2] = 10.0
    env.root_states[1, 9] = 0.3
    cost = env._reward_knee_load_support()
    self.assertGreater(cost[1], cost[0])
    self.assertTrue(torch.all((cost >= 0) & (cost <= 1)))

  def test_height_credit_reduces_knee_dominance_but_preserves_floor(self):
    env = self.make_env()
    env.cfg = SafeRecoveryCfg()
    env.root_states[:, 2] = 0.35
    history = torch.zeros(2, 8, 4, 3)
    history[:, 1, :, 2] = 20.0
    env.nonfoot_ground_sensor = SimpleNamespace(data=SimpleNamespace(force_history=history))
    env.knee_force_indices = torch.tensor([1, 5])
    env.robot = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([0])))
    env.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(2, 1)))
    env._ground_support_forces = lambda: (torch.tensor([0.0, 100.0]), torch.zeros(2), torch.zeros(2))
    credit = env._knee_height_credit()
    self.assertLess(credit[0], credit[1])
    self.assertTrue(torch.all((credit >= 0.35) & (credit <= 1.0)))
    history.zero_()
    self.assertTrue(torch.equal(env._knee_height_credit(), torch.ones(2)))

  def test_standing_gate_is_smooth_and_state_dependent(self):
    env = self.make_env()
    env.root_states[:, 2] = torch.tensor([0.34, 0.15])
    gate = env._standing_gate()
    self.assertTrue(torch.isfinite(gate).all())
    self.assertTrue(torch.all((gate > 0) & (gate < 1)))
    self.assertGreater(gate[0], gate[1])

  def test_pose_and_stability_costs_are_nonnegative_and_finite(self):
    env = self.make_env()
    env.root_states[:, 2] = 0.34
    env.dof_pos[0] = 0.5
    env.dof_vel[0] = 2.0
    env.base_ang_vel[0] = 1.0
    for value in (env._reward_default_posture(), env._reward_stable_standing()):
      self.assertTrue(torch.isfinite(value).all())
      self.assertTrue((value >= 0).all())
      self.assertGreater(value[0], value[1])

  def test_quaternion_product_stays_unit(self):
    a = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
    b = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
    self.assertTrue(torch.allclose(torch.linalg.vector_norm(_quat_mul(a, b), dim=1), torch.ones(1)))

  def test_action_std_guard_clamps_initial_and_post_update_values(self):
    actor = SimpleNamespace(std=torch.nn.Parameter(torch.tensor([-1.0, 2.0, 0.5])))

    class Algorithm:
      actor_critic = actor

      def update(self):
        with torch.no_grad():
          self.actor_critic.std.copy_(torch.tensor([0.0, 50.0, 0.7]))
        return "unchanged-result"

    algorithm = Algorithm()
    install_action_std_guard(algorithm, 0.05, 1.0)
    self.assertTrue(torch.allclose(actor.std, torch.tensor([0.05, 1.0, 0.5])))
    self.assertEqual(algorithm.update(), "unchanged-result")
    self.assertTrue(torch.allclose(actor.std, torch.tensor([0.05, 1.0, 0.7])))

  def test_action_std_guard_rejects_invalid_bounds(self):
    algorithm = SimpleNamespace(
      actor_critic=SimpleNamespace(std=torch.nn.Parameter(torch.ones(1))),
      update=lambda: None,
    )
    with self.assertRaises(ValueError):
      install_action_std_guard(algorithm, 1.0, 0.5)


if __name__ == "__main__":
  unittest.main()
