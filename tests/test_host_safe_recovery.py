"""Lightweight tests for the isolated HoST safe-recovery reward delta."""

import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import torch
from minipi_getup.host_safe_recovery.asset import ASSET_XML, EXPECTED_ASSET_SHA256
from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg
from minipi_getup.host_safe_recovery.env import (
  SafeRecoveryEnv,
  _foot_support_score,
  _quat_mul,
  _saturating_support,
)
from minipi_getup.host_safe_recovery.optimization import install_action_std_guard


class RewardTests(unittest.TestCase):
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
      "de53addd1184b579557c84c9511752a894a1e34a6aa3621555af3b092b961fd3",
    )

  def test_recovery_pull_force_curriculum_is_30_to_0_by_10(self):
    self.assertTrue(SafeRecoveryCfg.curriculum.pull_force)
    self.assertEqual(SafeRecoveryCfg.curriculum.force, 30)
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
      safe_recovery=SimpleNamespace(force_curriculum_step_n=10.0),
    )
    env.old_headheight = torch.full((4, 1), 0.5)
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
    return env

  def test_overspeed_threshold_and_mean_square(self):
    env = self.make_env()
    env.dof_vel[0, 0] = 4.0
    env.dof_vel[1, 0] = -6.0
    cost = env._reward_joint_overspeed()
    self.assertEqual(cost[0], 0)
    self.assertTrue(torch.allclose(cost[1], torch.tensor(4.0 / 12.0)))

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
