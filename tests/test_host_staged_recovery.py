"""Tests for the staged-recovery geometry and reward delta."""

import math
import unittest

import torch
from minipi_getup.host_staged_recovery import geometry as geo
from minipi_getup.host_staged_recovery.asset import (
  ASSET_XML,
  EXPECTED_ASSET_SHA256,
  asset_sha256,
)
from minipi_getup.host_staged_recovery.config import StagedRecoveryCfg
from minipi_getup.host_staged_recovery.env import StagedRecoveryEnv

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def yaw_quat(angle):
  return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def two_feet(y_left, y_right, yaw_left=0.0, yaw_right=0.0):
  pos = torch.tensor([[[0.0, y_left, 0.048], [0.0, y_right, 0.048]]])
  quat = torch.stack([yaw_quat(yaw_left), yaw_quat(yaw_right)])[None]
  return geo.sole_corners_world(pos, quat), quat


class GeometryTests(unittest.TestCase):
  def test_sole_corners_lie_on_ground_for_flat_foot(self):
    corners, _ = two_feet(0.08, -0.08)
    self.assertTrue(torch.allclose(corners[..., 2], torch.zeros(1, 2, 4), atol=1e-6))

  def test_polygon_distance_inside_outside_and_single_foot(self):
    corners, _ = two_feet(0.08, -0.08)
    verts = corners[..., :2].reshape(1, 8, 2).expand(4, 8, 2)
    valid = torch.ones(4, 8, dtype=torch.bool)
    valid[2, 4:] = False  # left foot only
    valid[3] = False
    point = torch.tensor([[-0.05, 0.0], [-0.05, 0.30], [-0.05, 0.0], [0.0, 0.0]])
    out, centre, anyv = geo.support_polygon_distance(point, verts, valid)
    self.assertEqual(out[0].item(), 0.0)  # between the feet: inside the hull
    # The outer sole edge is slanted: half-width 0.029 at the toe, 0.0233 at the heel.
    half = 0.029 - (0.026 + 0.05) / 0.155 * (0.029 - 0.0233)
    self.assertAlmostEqual(out[1].item(), 0.30 - 0.08 - half, places=3)
    self.assertAlmostEqual(out[2].item(), 0.08 - half, places=3)  # outside the left sole
    self.assertFalse(bool(anyv[3]))
    self.assertTrue(math.isinf(out[3].item()))
    score = geo.com_support_score(out, centre, anyv, 0.03, 0.06)
    self.assertGreater(score[0].item(), 0.9)
    self.assertLess(score[1].item(), 1e-6)
    self.assertEqual(score[3].item(), 0.0)

  def test_edge_contact_gives_a_line_support(self):
    corners, _ = two_feet(0.08, -0.08)
    verts = corners[..., :2].reshape(1, 8, 2)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[0, [0, 3]] = True  # left sole outer edge only
    on_line = verts[0, [0, 3]].mean(0, keepdim=True)
    out, *_ = geo.support_polygon_distance(on_line + torch.tensor([[0.0, -0.02]]), verts, valid)
    self.assertAlmostEqual(out.item(), 0.02, places=3)

  def test_lateral_gap_detects_touching_and_toe_in_feet(self):
    base = IDENTITY[None]
    apart, q = two_feet(0.08, -0.08)
    touching, q2 = two_feet(0.029, -0.029)
    toe_in, q3 = two_feet(0.05, -0.05, yaw_left=-0.6, yaw_right=0.6)
    gap = lambda c, fq: geo.foot_lateral_gap(c, geo.lateral_axis(base, fq)).item()  # noqa: E731
    self.assertAlmostEqual(gap(apart, q), 0.16 - 0.058, places=3)
    self.assertLess(gap(touching, q2), 1e-3)
    width = torch.linalg.vector_norm(toe_in.mean(2)[:, 0, :2] - toe_in.mean(2)[:, 1, :2], dim=-1)
    self.assertGreater(width.item(), 0.09)  # centre distance alone looks acceptable
    self.assertLess(gap(toe_in, q3), 0.04)

  def test_lateral_axis_follows_torso_even_with_opposite_foot_yaw(self):
    _, q = two_feet(0.08, -0.08, yaw_left=1.57, yaw_right=-1.57)  # headings cancel
    axis = geo.lateral_axis(yaw_quat(0.3)[None], q)
    self.assertTrue(torch.allclose(axis, torch.tensor([[-math.sin(0.3), math.cos(0.3)]]), atol=1e-5))
    rolled = torch.tensor([[math.cos(0.75), math.sin(0.75), 0.0, 0.0]])  # torso rolled 1.5 rad
    _, q2 = two_feet(0.08, -0.08)
    self.assertTrue(torch.allclose(geo.lateral_axis(rolled, q2), torch.tensor([[0.0, 1.0]]), atol=1e-5))

  def test_stance_cost_band(self):
    gap = torch.tensor([0.10, 0.0, 0.10, 0.10])
    width = torch.tensor([0.16, 0.16, 0.42, 0.9])
    cost = geo.stance_width_cost(gap, width, 0.04, 0.26, 0.03, 0.08, 4.0)
    self.assertEqual(cost[0].item(), 0.0)
    self.assertAlmostEqual(cost[1].item(), (0.04 / 0.03) ** 2, places=4)
    self.assertAlmostEqual(cost[2].item(), 4.0, places=4)
    self.assertEqual(cost[3].item(), 4.0)

  def test_feet_under_com_and_unsupported_ascent(self):
    feet = torch.tensor([[[0.0, 0.08], [0.0, -0.08]], [[0.3, 0.08], [0.3, -0.08]]])
    score = geo.feet_under_com_score(feet, torch.zeros(2, 2), 0.12, 0.08)
    self.assertEqual(score[0].item(), 1.0)
    self.assertLess(score[1].item(), 0.05)
    vz = torch.tensor([0.25, 0.25, 0.0])
    cost = geo.unsupported_ascent_cost(vz, torch.tensor([0.0, 1.0, 0.0]), torch.ones(3), 0.05, 0.2)
    self.assertTrue(torch.allclose(cost, torch.tensor([1.0, 0.0, 0.0])))

  def test_knee_extension_and_knee_ramp(self):
    calf = torch.tensor([[-0.65, -0.65], [0.5, 0.5], [-0.65, -0.65]])
    support = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    cost = geo.knee_extension_cost(calf, support, 0.25, 0.40)
    self.assertTrue(torch.allclose(cost, torch.tensor([1.0, 0.0, 0.0])))
    ramp = geo.knee_contact_ramp(torch.tensor([0.0, 0.4, 2.0]), 0.8, 0.25)
    self.assertTrue(torch.allclose(ramp, torch.tensor([0.25, 0.625, 1.0])))

  def test_load_imbalance_penalizes_one_foot_standing(self):
    left = torch.tensor([34.0, 64.0, 0.0])
    right = torch.tensor([34.0, 4.0, 0.0])
    cost = geo.load_imbalance_cost(left, right, torch.tensor(68.0), 0.1)
    self.assertEqual(cost[0].item(), 0.0)
    self.assertGreater(cost[1].item(), 0.6)  # (60 / 74.8)^2
    self.assertEqual(cost[2].item(), 0.0)

  def test_two_foot_share_and_ready_need_every_pipeline_step(self):
    left = torch.tensor([34.0, 64.0, 0.0, 20.0])
    right = torch.tensor([34.0, 4.0, 0.0, 48.0])
    share = geo.two_foot_share_score(left, right, 0.3)
    self.assertEqual(share[0].item(), 1.0)
    self.assertAlmostEqual(share[1].item(), (4 / 68) / 0.3, places=4)  # one-foot standing ~0.2
    self.assertEqual(share[2].item(), 0.0)
    self.assertAlmostEqual(share[3].item(), (20 / 68) / 0.3, places=4)
    gate = geo.two_foot_support_gate(share, torch.ones(4))
    self.assertEqual(gate[0].item(), 1.0)
    self.assertLess(gate[1].item(), 0.11)  # one-foot standing: smoothstep(0.196) = 0.100
    self.assertEqual(gate[2].item(), 0.0)
    cov = torch.ones(4)
    ready = geo.rise_ready_score(cov, torch.tensor([1.0, 1.0, 1.0, 0.0]), gate)
    self.assertEqual(ready[0].item(), 1.0)
    self.assertLess(ready[1].item(), 0.11)
    self.assertEqual(ready[2].item(), 0.0)
    self.assertEqual(ready[3].item(), 0.0)  # COM outside the support
    half = geo.rise_ready_score(torch.tensor([0.25]), torch.tensor([1.0]), torch.tensor([1.0]))
    self.assertAlmostEqual(half.item(), 0.5, places=6)  # geometric mean of coverage and COM support

  def test_height_credit_has_an_intermediate_two_foot_step(self):
    gate = torch.tensor([0.0, 1.0, 1.0, 0.1])
    ready = torch.tensor([0.0, 0.0, 1.0, 0.1])
    credit = geo.height_credit(gate, ready, 0.2, 0.3)
    self.assertTrue(torch.allclose(credit, torch.tensor([0.2, 0.5, 1.0, 0.28])))  # one foot / both / ready

  def test_coverage_pair_weights_the_worse_sole(self):
    pair = geo.coverage_pair_score(torch.tensor([1.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.5]), 0.3)
    self.assertTrue(torch.allclose(pair, torch.tensor([1.0, 0.15, 0.5])))

  def test_polygon_margin_is_signed(self):
    corners, _ = two_feet(0.08, -0.08)
    verts = corners[..., :2].reshape(1, 8, 2).expand(3, 8, 2)
    valid = torch.ones(3, 8, dtype=torch.bool)
    valid[2] = False
    point = torch.tensor([[-0.05, 0.0], [-0.05, 0.30], [0.0, 0.0]])
    out, _, _, margin = geo.support_polygon_distance(point, verts, valid, with_margin=True)
    # Between the feet, 5 cm behind the toe line (x = 0.026): nearest edge is the toe edge.
    self.assertAlmostEqual(margin[0].item(), 0.076, places=3)
    self.assertAlmostEqual(margin[1].item(), -out[1].item(), places=6)
    self.assertTrue(math.isinf(margin[2].item()) and margin[2].item() < 0)
    score = geo.com_margin_score(torch.tensor([0.05, 0.0125, -0.02, -math.inf, math.nan]), 0.025)
    self.assertTrue(torch.allclose(score, torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])))

  def test_torso_lateral_error_uses_the_lateral_axis_only(self):
    feet = torch.tensor([[[0.0, 0.08], [0.0, -0.08]], [[0.0, 0.08], [0.0, -0.08]]])
    torso = torch.tensor([[0.10, 0.0], [0.0, -0.05]])
    lateral = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    err = geo.torso_lateral_error(torso, feet, lateral)
    self.assertTrue(torch.allclose(err, torch.tensor([0.0, 0.05])))  # fore-aft offset is free

  def test_geometric_sole_coverage_tolerates_small_tilt_only(self):
    pos = torch.tensor([[[0.0, 0.08, 0.048], [0.0, -0.08, 0.048]]])
    flat = torch.stack([IDENTITY, IDENTITY])[None]
    z = geo.sole_strip_points_world(pos, flat)[..., 2]
    self.assertEqual(tuple(z.shape), (1, 2, 10))
    self.assertTrue(torch.allclose(z, torch.zeros_like(z), atol=1e-6))
    cov = geo.geometric_sole_coverage(z, torch.ones(1, 2), 0.003)
    self.assertTrue(torch.allclose(cov, torch.ones(1, 2)))
    def rolled(deg):  # roll about the foot x axis, lowest edge on the ground
      a = math.radians(deg)
      q = torch.tensor([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])
      pts = geo.sole_strip_points_world(torch.zeros(1, 1, 3), q[None, None])[..., 2]
      return geo.geometric_sole_coverage(pts - pts.amin(-1, keepdim=True), torch.ones(1, 1), 0.003).item()
    self.assertGreater(rolled(1.0), 0.9)
    self.assertLess(rolled(3.0), 0.8)
    self.assertLess(rolled(10.0), 0.4)
    half = geo.geometric_sole_coverage(torch.zeros(1, 2, 10), torch.tensor([[1.0, 0.5]]), 0.003)
    self.assertTrue(torch.allclose(half, torch.tensor([[1.0, 0.5]])))  # load gate

  def test_foot_translation_hold_survives_brief_unloading_in_contact_only(self):
    decay = math.exp(-0.02 / 0.25)
    hold = torch.tensor([[1.0, 1.0]])
    memory = torch.zeros(1, 2)  # load dropped to zero on both feet
    contact = torch.tensor([[True, False]])  # left still on the ground, right lifted
    for _ in range(5):  # 0.1 s
      hold = geo.contact_load_hold(hold, memory, contact, decay)
    self.assertAlmostEqual(hold[0, 0].item(), decay ** 5, places=5)  # ~0.67: dragging still pays
    self.assertEqual(hold[0, 1].item(), 0.0)  # stepping is free
    loaded = geo.contact_load_hold(torch.zeros(1, 2), torch.tensor([[0.8, 0.8]]), torch.tensor([[True, True]]), decay)
    self.assertTrue(torch.allclose(loaded, torch.tensor([[0.8, 0.8]])))
    cost = geo.foot_translation_cost(torch.tensor([[0.13, 0.5], [0.02, 0.02]]), torch.tensor([[1.0, 0.0], [1.0, 1.0]]), 0.03, 0.10)
    self.assertTrue(torch.allclose(cost, torch.tensor([0.5, 0.0])))

  def test_foot_anchor_follows_lifted_feet_and_charges_planted_drift(self):
    anchor = torch.zeros(1, 2, 2)
    xy = torch.tensor([[[0.04, 0.0], [0.04, 0.0]]])
    follow = torch.tensor([[False, True]])  # left planted, right lifted
    anchor = geo.foot_anchor_update(anchor, xy, follow)
    self.assertTrue(torch.allclose(anchor, torch.tensor([[[0.0, 0.0], [0.04, 0.0]]])))
    cost = geo.foot_anchor_cost(xy, anchor, torch.ones(1, 2), 0.01, 0.02, 3.0)
    self.assertAlmostEqual(cost.item(), 0.5 * 1.5, places=5)  # left drifted 4 cm, right re-anchored
    far = geo.foot_anchor_cost(torch.tensor([[[0.5, 0.0], [0.0, 0.0]]]), torch.zeros(1, 2, 2), torch.ones(1, 2), 0.01, 0.02, 3.0)
    self.assertAlmostEqual(far.item(), 1.5, places=5)  # capped at 3 per foot

  def test_stance_offset_error_is_zero_at_the_default_place(self):
    hips = torch.tensor([[[0.0, 0.05], [0.0, -0.05]]])
    soles = hips + torch.tensor([[[0.0037, 0.0568], [0.0037, -0.0568]]])
    fwd, lat = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    err = geo.stance_offset_error(soles, hips, fwd, lat, 0.0568, 0.0037)
    self.assertTrue(torch.allclose(err, torch.zeros(1, 2), atol=1e-6))
    wide = soles + torch.tensor([[[0.0, 0.04], [0.0, -0.04]]])  # both 4 cm further out
    self.assertTrue(torch.allclose(geo.stance_offset_error(wide, hips, fwd, lat, 0.0568, 0.0037), torch.full((1, 2), 0.04)))

  def test_stance_return_score_has_coarse_and_fine_kernels(self):
    score = geo.stance_return_score(torch.tensor([0.0, 0.02, 0.088]), (0.02, 0.08))
    self.assertAlmostEqual(score[0].item(), 1.0, places=6)
    self.assertGreater(score[2].item(), 0.1)  # still a gradient 8.8 cm away
    self.assertAlmostEqual(score[1].item(), 0.5 * math.exp(-1) + 0.5 * math.exp(-1 / 16), places=5)

  def test_step_placement_gain_is_signed_and_capped(self):
    gain = geo.step_placement_gain(torch.tensor([0.08, 0.02, 0.2]), torch.tensor([0.02, 0.08, 0.0]), 0.02, 3.0)
    self.assertTrue(torch.allclose(gain, torch.tensor([3.0, -3.0, 3.0])))
    out_and_back = geo.step_placement_gain(torch.tensor(0.03), torch.tensor(0.05), 0.02, 3.0) + geo.step_placement_gain(
      torch.tensor(0.05), torch.tensor(0.03), 0.02, 3.0)
    self.assertAlmostEqual(out_and_back.item(), 0.0, places=6)

  def test_stance_library_feet_flat_and_displaced(self):
    from minipi_getup.host_staged_recovery.stance_poses import build_stance_library
    names = [f"{s}_{j}_joint" for s in "rl" for j in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")]
    lib = build_stance_library(str(ASSET_XML), names, n=6, seed=1)
    self.assertEqual(lib["dof_pos"].shape, (6, 12))
    self.assertTrue(((lib["offset"][..., 0] >= 0.02) & (lib["offset"][..., 0] <= 0.08)).all())
    self.assertTrue(((lib["base_xyz"][:, 2] > 0.3) & (lib["base_xyz"][:, 2] < 0.36)).all())

  def test_swing_credit_window(self):
    credit = geo.swing_credit(torch.tensor([0.1, 0.3, 0.4, 0.6]), 0.3, 0.5)
    self.assertTrue(torch.allclose(credit, torch.tensor([1.0, 1.0, 0.5, 0.0])))

  def test_rise_speed(self):
    cost = geo.rise_speed_cost(torch.tensor([0.1, 0.3, 2.0, -1.0]), 0.2, 0.1, 9.0)
    self.assertTrue(torch.allclose(cost, torch.tensor([0.0, 1.0, 9.0, 0.0])))

  def test_one_foot_rise_cost(self):
    loaded = torch.tensor([1.0, 1.0, 0.0, 1.0])
    two_foot = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gate = torch.tensor([1.0, 1.0, 1.0, 0.0])
    cost = geo.one_foot_rise_cost(loaded, two_foot, gate)
    self.assertTrue(torch.equal(cost, torch.tensor([1.0, 0.0, 0.0, 0.0])))  # one foot / both / knees / not rising

  def test_speed_limit_fades_accelerating_torque_only(self):
    torque = torch.tensor([16.0, 16.0, 16.0, -16.0])
    vel = torch.tensor([2.0, 5.25, 7.0, 7.0])
    out = geo.speed_limited_torque(torque, vel, torch.tensor(16.0), 6.0, 1.5)
    self.assertTrue(torch.allclose(out, torch.tensor([16.0, 8.0, 0.0, -16.0])))


class TerminationTests(unittest.TestCase):
  def test_drag_termination_is_off_until_curriculum_then_tightens(self):
    env = object.__new__(StagedRecoveryEnv)
    env.cfg = StagedRecoveryCfg()
    for strength, expected in ((0.0, math.inf), (0.5, 0.13), (1.0, 0.06)):
      env._penalty_curriculum_strength = lambda s=strength: torch.tensor(s)
      self.assertAlmostEqual(env.drag_termination_limit().item(), expected, places=5)

  def test_velocity_termination_is_off_until_curriculum_then_tightens(self):
    env = object.__new__(StagedRecoveryEnv)
    env.cfg = StagedRecoveryCfg()
    for strength, expected in ((0.0, math.inf), (1e-3, 12.0 - 7e-3), (0.5, 8.5), (1.0, 5.0)):
      env._penalty_curriculum_strength = lambda s=strength: torch.tensor(s)
      self.assertAlmostEqual(env.velocity_termination_limit().item(), expected, places=4)


class ConfigTests(unittest.TestCase):
  def test_asset_is_pinned_hip120_with_left_right_shells(self):
    self.assertEqual(asset_sha256(), EXPECTED_ASSET_SHA256)
    text = ASSET_XML.read_text()
    self.assertEqual(text.count('_shell_collision" class="collision" type="box" contype="2" conaffinity="12"'), 2)
    self.assertEqual(text.count('_shell_collision" class="collision" type="box" contype="4" conaffinity="10"'), 2)
    self.assertIn('l_foot_shell_collision" class="collision" type="box" contype="66" conaffinity="12"', text)
    self.assertIn('r_foot_shell_collision" class="collision" type="box" contype="132" conaffinity="10"', text)
    self.assertIn('l_hip_shell_collision" class="collision" type="box" contype="16" conaffinity="64"', text)
    self.assertIn('r_hip_shell_collision" class="collision" type="box" contype="32" conaffinity="128"', text)
    self.assertEqual(text.count('name="base_shell_collision" class="collision" type="box" contype="8" conaffinity="6"'), 1)
    self.assertEqual(text.count('range="-1.25 1.75"'), 2)
    self.assertEqual(text.count('calf_joint" pos="0 0 0" axis="0 1 0" range="-2.5 1.65"'), 2)
    self.assertIn('r_thigh_joint" pos="0 0 0" axis="0 0 1" range="-0.28 0.3"', text)
    self.assertIn('l_thigh_joint" pos="0 0 0" axis="0 0 1" range="-0.3 0.28"', text)
    self.assertIn('l_hip_roll_joint" pos="0 0 0" axis="1 0 0" range="-0.12 2.0943951023931953"', text)

  def test_no_joint_strategy_rewards_and_pipeline_terms_registered(self):
    s = StagedRecoveryCfg.constraints.scales
    for name in ("target_loaded_knee_flexion", "target_knee_sole_progress", "target_recovery_hip_roll",
                 "style_loaded_hip_roll_excess", "style_loaded_knee_hyperextension", "regu_joint_overspeed",
                 "style_hip_roll_deviation"):
      self.assertEqual(getattr(s, name), 0.0, name)
    for name in ("target_feet_under_com", "target_two_foot_flat_support", "target_com_support",
                 "style_stance_width", "style_unsupported_ascent", "style_standing_knee_extension",
                 "style_joint_overspeed", "style_standing_load_imbalance", "style_leg_contact",
                 "style_one_foot_rise", "style_rise_speed", "target_torso_center_over_feet",
                 "target_com_support_margin", "target_sole_contact_coverage", "style_loaded_foot_translation",
                 "style_standing_foot_anchor", "target_stance_return", "target_stable_standing",
                 "target_step_placement"):
      self.assertNotEqual(getattr(s, name), 0.0)
      self.assertTrue(callable(getattr(StagedRecoveryEnv, "_reward_" + name.split("_", 1)[1])))
    self.assertEqual(StagedRecoveryCfg.safe_recovery.evaluation_ceiling_rad_s, 6.0)
    self.assertEqual(StagedRecoveryCfg.safe_recovery.overspeed_soft_rad_s, 3.0)
    self.assertEqual(StagedRecoveryCfg.staged.motor_max_speed_rad_s, 4.0)
    self.assertEqual(StagedRecoveryCfg.curriculum.force, 40)
    self.assertFalse(hasattr(StagedRecoveryCfg.staged, "force_requires_com"))

  def test_revision_13_scales_and_unique_reward_functions(self):
    s = StagedRecoveryCfg.constraints.scales
    expected = {
      "target_sole_contact_coverage": 8.0, "style_loaded_sole_coverage_deficit": -10.0,
      "style_loaded_sole_tilt": -8.0, "target_two_foot_flat_support": 6.0,
      "target_torso_center_over_feet": 4.0, "target_com_support": 6.0,
      "target_com_support_margin": 2.0, "target_stable_standing": -1.0,
      "style_loaded_foot_translation": -4.0, "style_loaded_sole_slip": -6.0,
      "style_standing_foot_anchor": -4.0, "target_stance_return": 4.0, "target_step_placement": 100.0,
    }
    for name, value in expected.items():
      self.assertEqual(getattr(s, name), value, name)
    self.assertEqual(StagedRecoveryCfg.staged.height_credit_com_floor, 0.2)
    self.assertEqual(StagedRecoveryCfg.staged.height_credit_two_foot_weight, 0.3)
    self.assertEqual(StagedRecoveryCfg.staged.stance_rise_fraction, 0.25)
    self.assertEqual(StagedRecoveryCfg.staged.revision, "13")
    self.assertFalse(hasattr(StagedRecoveryCfg.staged, "ready_com_centre_sigma_m"))
    st = StagedRecoveryCfg.staged
    self.assertEqual((st.stance_start_prob, st.drag_termination_start_m, st.drag_termination_final_m), (0.15, 0.20, 0.06))
    self.assertFalse(st.drag_termination)
    self.assertEqual(st.action_scale_floor, 0.30)
    active = [n for n in dir(s) if not n.startswith("_") and getattr(s, n) != 0]
    functions = [n.split("_", 1)[1] for n in active]
    self.assertEqual(len(functions), len(set(functions)), "a reward function is registered twice")


if __name__ == "__main__":
  unittest.main()
