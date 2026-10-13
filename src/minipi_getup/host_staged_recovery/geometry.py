"""Support-polygon geometry for staged recovery rewards and evaluation.

All quantities are horizontal (world x/y) on the flat terrain plane z = 0.
"""

from __future__ import annotations

import itertools

import torch
from mjlab.utils.lab_api.math import quat_apply

# Ground-contact footprint of one sole in the ankle_roll_link frame. The five
# sole capsules (radius 0.005 m, axis z = -0.043 m) run heel-to-toe; their lowest
# lines span x in [-0.129, 0.026] and y within +-0.029 (toe) / +-0.0233 (heel).
# Left and right soles share the same local geometry in the XML.
SOLE_CORNERS_LOCAL = (
  (0.026, 0.029, -0.048),
  (0.026, -0.029, -0.048),
  (-0.129, -0.0233, -0.048),
  (-0.129, 0.0233, -0.048),
)

_PAIRS = tuple(itertools.combinations(range(8), 2))

# Bottom lines of the 5 sole capsules (radius 0.005 at z = -0.043, so z = -0.048),
# toe (x = 0.026) and heel (x = -0.129) ends, in the ankle_roll_link frame.
SOLE_STRIP_POINTS_LOCAL = tuple(
  (x, y, -0.048)
  for y_toe, y_heel in ((-0.029, -0.0233), (-0.0145, -0.01165), (0.0, 0.0), (0.0145, 0.01165), (0.029, 0.0233))
  for x, y in ((0.026, y_toe), (-0.129, y_heel))
)


def sole_corners_world(foot_pos: torch.Tensor, foot_quat: torch.Tensor) -> torch.Tensor:
  """(N, 2, 3) foot frames -> (N, 2, 4, 3) world sole corners."""
  local = torch.tensor(SOLE_CORNERS_LOCAL, dtype=foot_pos.dtype, device=foot_pos.device)
  n, feet = foot_pos.shape[:2]
  quat = foot_quat[:, :, None].expand(n, feet, 4, 4).reshape(-1, 4)
  offset = quat_apply(quat, local.expand(n, feet, 4, 3).reshape(-1, 3))
  return foot_pos[:, :, None] + offset.reshape(n, feet, 4, 3)


def sole_strip_points_world(foot_pos: torch.Tensor, foot_quat: torch.Tensor) -> torch.Tensor:
  """(N, 2, 3) foot frames -> (N, 2, 10, 3) world sole-strip end points."""
  local = torch.tensor(SOLE_STRIP_POINTS_LOCAL, dtype=foot_pos.dtype, device=foot_pos.device)
  n, feet = foot_pos.shape[:2]
  k = local.shape[0]
  quat = foot_quat[:, :, None].expand(n, feet, k, 4).reshape(-1, 4)
  offset = quat_apply(quat, local.expand(n, feet, k, 3).reshape(-1, 3))
  return foot_pos[:, :, None] + offset.reshape(n, feet, k, 3)


def contact_slip_cost(linear, angular, offset, found, threshold, scale):
  """Mean tangential slip at actual contacts, independent of load or stage."""
  velocity = linear + torch.cross(angular, offset, dim=-1)
  excess = torch.relu(torch.linalg.vector_norm(velocity[..., :2], dim=-1) - threshold) / scale
  touching = (found > 0).to(excess.dtype)
  return (excess * touching).sum(dim=1) / touching.sum(dim=1).clamp_min(1.0)


def geometric_sole_coverage(point_z: torch.Tensor, load_gate: torch.Tensor, tolerance_m: float) -> torch.Tensor:
  """Per-foot sole coverage from geometry: mean over the strip end points of
  exp(-(clearance/tol)^2) (points at or below the ground count fully), times a
  load gate. point_z (N, 2, 10), load_gate (N, 2) -> (N, 2) in [0, 1].

  The force-based coverage is all-or-nothing: the rigid capsules sink < 1 mm, so
  0.2 deg of sole roll already puts the whole load on one edge. With a 3 mm
  tolerance, coverage >= 0.9 means about <= 1.7 deg of roll or pitch."""
  near = torch.exp(-(torch.relu(point_z) / tolerance_m).square()).mean(-1)
  return near * load_gate


def support_polygon_distance(
  point: torch.Tensor, vertices: torch.Tensor, valid: torch.Tensor, eps: float = 1e-9,
  with_margin: bool = False,
) -> tuple[torch.Tensor, ...]:
  """Distance from a 2D point to the convex hull of the valid vertices.

  point (N, 2); vertices (N, 8, 2); valid (N, 8). Returns (outside distance,
  0 inside; distance to the valid-vertex centroid; any-valid mask). Hull edges
  are the vertex pairs with every other valid vertex on one side, so a single
  foot, two feet, or an edge contact (collinear vertices) are all handled.
  Rows without valid vertices return +inf. With `with_margin`, also returns the
  signed margin: depth inside the hull (distance to the nearest hull edge), minus
  the outside distance otherwise (-inf without support).
  """
  pairs = torch.tensor(_PAIRS, device=point.device)
  a, b = vertices[:, pairs[:, 0]], vertices[:, pairs[:, 1]]  # (N, 28, 2)
  edge = b - a
  pair_valid = valid[:, pairs[:, 0]] & valid[:, pairs[:, 1]] & (edge.square().sum(-1) > eps)
  rel = vertices[:, None] - a[:, :, None]  # (N, 28, 8, 2)
  cross = edge[:, :, None, 0] * rel[..., 1] - edge[:, :, None, 1] * rel[..., 0]
  others = valid[:, None, :].expand_as(cross)
  nonneg = ((cross >= -eps) | ~others).all(-1)
  nonpos = ((cross <= eps) | ~others).all(-1)
  hull = pair_valid & (nonneg | nonpos)
  side = torch.where(nonneg & ~nonpos, 1.0, torch.where(nonpos & ~nonneg, -1.0, 0.0))
  rel_p = point[:, None] - a
  cp = edge[..., 0] * rel_p[..., 1] - edge[..., 1] * rel_p[..., 0]
  out_edge = hull & (
    ((side > 0) & (cp < -eps)) | ((side < 0) & (cp > eps)) | ((side == 0) & (cp.abs() > eps))
  )
  inside = hull.any(-1) & ~out_edge.any(-1)
  t = ((rel_p * edge).sum(-1) / edge.square().sum(-1).clamp_min(eps)).clamp(0.0, 1.0)
  seg = torch.linalg.vector_norm(rel_p - t[..., None] * edge, dim=-1)
  seg = torch.where(hull, seg, torch.inf).amin(-1)
  pts = torch.linalg.vector_norm(point[:, None] - vertices, dim=-1)
  pts = torch.where(valid, pts, torch.inf).amin(-1)
  any_valid = valid.any(-1)
  outside = torch.where(inside, torch.zeros_like(seg), torch.minimum(seg, pts))
  weight = valid.float()
  centroid = (vertices * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1.0)
  centre = torch.linalg.vector_norm(point - centroid, dim=-1)
  centre = torch.where(any_valid, centre, torch.inf)
  if with_margin:
    return outside, centre, any_valid, torch.where(inside, seg, -outside)
  return outside, centre, any_valid


def com_support_score(outside, centre, any_valid, sigma_out, sigma_centre):
  """1 for a COM centred over the support, 0.7 at its edge, ~0 well outside."""
  score = torch.exp(-(outside / sigma_out).square()) * (
    0.7 + 0.3 * torch.exp(-(centre / sigma_centre).square())
  )
  return torch.where(any_valid, score, torch.zeros_like(score))


def lateral_axis(base_quat: torch.Tensor, foot_quat: torch.Tensor) -> torch.Tensor:
  """Horizontal unit vector pointing to the robot's left.

  The torso's y axis projected on the ground; when the torso is rolled so far
  that this projection is short (< 0.3), fall back to the left of the summed foot
  headings. Feet yawed in opposite directions do not destabilize it while upright.
  """
  n = base_quat.shape[0]
  y_axis = torch.zeros(n, 3, device=base_quat.device, dtype=base_quat.dtype)
  y_axis[:, 1] = 1.0
  base_left = quat_apply(base_quat, y_axis)[:, :2]
  x_axis = torch.zeros(n * 2, 3, device=base_quat.device, dtype=base_quat.dtype)
  x_axis[:, 0] = 1.0
  heading = quat_apply(foot_quat.reshape(-1, 4), x_axis).reshape(n, 2, 3)[..., :2].sum(1)
  foot_left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
  use_base = torch.linalg.vector_norm(base_left, dim=-1, keepdim=True) > 0.3
  axis = torch.where(use_base, base_left, foot_left)
  return axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)


def foot_lateral_gap(corners: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
  """Clearance between the two sole footprints along `lateral` (robot left).

  corners (N, 2, 4, 3) with index 0 = left foot. Positive when the left sole lies
  entirely left of the right sole; <= 0 when the soles touch, overlap or cross.
  Zero joints give about 0.10 m.
  """
  side = (corners[..., :2] * lateral[:, None, None]).sum(-1)  # (N, 2, 4)
  return side[:, 0].amin(-1) - side[:, 1].amax(-1)


def stance_width_cost(gap, width, min_gap, max_width, gap_scale, width_scale, cap):
  """Zero for separated soles (gap >= min_gap) no wider than max_width (sole
  centre distance); squared normalized violation outside."""
  low = (torch.relu(min_gap - gap) / gap_scale).square()
  high = (torch.relu(width - max_width) / width_scale).square()
  return (low + high).clamp(max=cap)


def adequate_stance(gap, width, min_gap, max_width):
  return (gap >= min_gap) & (width <= max_width)


def feet_under_com_score(foot_xy, com_xy, radius, scale):
  """Per-foot exp(-(excess horizontal distance / scale)^2), averaged over feet."""
  dist = torch.linalg.vector_norm(foot_xy - com_xy[:, None], dim=-1)
  return torch.exp(-(torch.relu(dist - radius) / scale).square()).mean(-1)


def unsupported_ascent_cost(com_vz, support_score, gate, threshold, scale):
  """Upward COM speed above `threshold`, weighted by missing COM support."""
  return (torch.relu(com_vz - threshold) / scale).square() * (1.0 - support_score) * gate


def knee_extension_cost(calf, weight, soft_rad, span_rad):
  """Weighted squared knee extension past `soft_rad` (calf < -soft).

  Calf 0 is the default 37-degree flexed knee, -0.65 the straight leg, and
  below that the knee bends backwards (asset limit -2.5 = 106 degrees).
  """
  return ((torch.relu(-calf - soft_rad) / span_rad).square() * weight).mean(-1)


def two_foot_share_score(left_n, right_n, full_share, eps=1.0):
  """min(L, R) / (L + R), scaled so a weaker-foot share >= full_share scores 1.
  ~0 when standing on one foot or with no load."""
  share = torch.minimum(left_n, right_n) / (left_n + right_n).clamp_min(eps)
  return (share / full_share).clamp(0.0, 1.0)


def smoothstep01(x):
  """3x^2 - 2x^3 on x clipped to [0, 1]: a C1 ramp."""
  x = x.clamp(0.0, 1.0)
  return x.square() * (3.0 - 2.0 * x)


def coverage_pair_score(cov_left, cov_right, mean_weight=0.3):
  """Sole coverage of both feet: mostly the worse foot, so one fully planted sole
  cannot make up for a heel/edge contact on the other."""
  return mean_weight * 0.5 * (cov_left + cov_right) + (1.0 - mean_weight) * torch.minimum(cov_left, cov_right)


def two_foot_support_gate(two_foot, loaded):
  """Smooth both-feet load gate: smoothstep of the weaker-foot share score times
  the stronger foot's normalized support."""
  return smoothstep01(two_foot) * loaded


def rise_ready_score(coverage_pair, com_score, support_gate):
  """Readiness to rise: sqrt(sole coverage x COM support), gated by both feet
  sharing the load."""
  return torch.sqrt((coverage_pair * com_score).clamp_min(0.0)) * support_gate


def height_credit(support_gate, ready, floor, two_foot_weight):
  """floor + two_foot_weight * support_gate + rest * ready. ready <= support_gate,
  so the credit rises in steps: one foot -> both feet loaded -> fully ready."""
  return floor + two_foot_weight * support_gate + (1.0 - floor - two_foot_weight) * ready


def torso_lateral_error(torso_xy, foot_xy, lateral):
  """|lateral offset| of the torso from the midpoint of the two feet, along the
  robot lateral axis. torso_xy (N, 2), foot_xy (N, 2, 2), lateral (N, 2)."""
  mid = foot_xy.mean(dim=1)
  return ((torso_xy - mid) * lateral).sum(-1).abs()


def com_margin_score(margin, full_margin):
  """0 outside the support polygon or without support, smoothstep up to 1 at
  `full_margin` inside it."""
  m = torch.where(torch.isfinite(margin), margin, torch.full_like(margin, -1.0))
  return smoothstep01(m / full_margin)


def contact_load_hold(hold, support_memory, in_contact, decay):
  """Load gate for foot translation, per foot. While the sole is near the ground it
  follows support_memory up and decays slowly (`decay` per step) when the load
  drops, so briefly unloading a foot that still touches the ground does not
  switch the cost off. A lifted foot (stepping, airborne) resets it to 0."""
  return torch.where(in_contact, torch.maximum(support_memory, hold * decay), torch.zeros_like(hold))


def foot_translation_cost(foot_speed_xy, gate, threshold, scale):
  """mean over feet of gate * relu(horizontal sole speed - threshold) / scale."""
  return (gate * torch.relu(foot_speed_xy - threshold) / scale).mean(-1)


def foot_anchor_update(anchor, foot_xy, follow):
  """Anchor = where the foot last settled. It follows the foot while `follow`
  (lifted, or not yet standing) and freezes once the foot is down."""
  return torch.where(follow[..., None], foot_xy, anchor)


def foot_anchor_cost(foot_xy, anchor, hold, deadband, scale, cap):
  """mean over feet of hold * min(cap, relu(drift - deadband) / scale): a planted
  foot that slides away from where it was set down pays for as long as it stays
  there; lifting it re-anchors it for free."""
  drift = torch.linalg.vector_norm(foot_xy - anchor, dim=-1)
  return (hold * (torch.relu(drift - deadband) / scale).clamp(max=cap)).mean(-1)


def stance_offset_error(sole_xy, hip_xy, forward, lateral, default_out, default_fwd):
  """Per-foot horizontal distance (m) of the sole centre from its default place
  relative to its own hip: default_out outward along the lateral axis (left foot
  +lateral, right foot -lateral) and default_fwd along the heading.
  sole_xy, hip_xy (N, 2, 2); forward, lateral (N, 2)."""
  rel = sole_xy - hip_xy
  out = (rel * lateral[:, None]).sum(-1) * torch.tensor([1.0, -1.0], device=rel.device, dtype=rel.dtype)
  fwd = (rel * forward[:, None]).sum(-1)
  return torch.sqrt((out - default_out).square() + (fwd - default_fwd).square())


def stance_return_score(err, sigmas):
  """Mean of exp(-(err/sigma)^2) over the given sigmas (fine + coarse)."""
  return torch.stack([torch.exp(-(err / s).square()) for s in sigmas]).mean(0)


def step_placement_gain(err_at_lift, err_now, scale, cap):
  """(stance error at lift-off - at touchdown) / scale, clipped to +-cap. Signed,
  so stepping out and back in nets zero."""
  return ((err_at_lift - err_now) / scale).clamp(-cap, cap)


def swing_credit(air_time, free_s, max_s):
  """1 for a foot that has been off the ground at most `free_s`, falling to 0
  at `max_s`: a short lift to re-place a foot is exempt, standing on one foot is not."""
  return (1.0 - (air_time - free_s) / (max_s - free_s)).clamp(0.0, 1.0)


def rise_speed_cost(com_vz, threshold, scale, cap):
  """Squared upward COM speed above `threshold`: getting up slowly is free."""
  return (torch.relu(com_vz - threshold) / scale).square().clamp(max=cap)


def one_foot_rise_cost(loaded, two_foot, rise_gate):
  """Rising with the load on one foot: (most loaded foot's support) x
  (1 - two-foot share score) x rise gate. Zero for an even two-foot push, and
  for knee/body support with no foot load (limited by the knee terms)."""
  return loaded * (1.0 - two_foot) * rise_gate


def load_imbalance_cost(left_n, right_n, weight_n, floor_fraction):
  """((L - R) / (L + R + floor*W))^2: ~1 on one foot, 0 for an even split."""
  return ((left_n - right_n) / (left_n + right_n + floor_fraction * weight_n)).square()


def knee_contact_ramp(knee_time, ramp_s, initial):
  """Brief knee support costs `initial`; sustained support ramps to full cost."""
  return initial + (1.0 - initial) * (knee_time / ramp_s).clamp(0.0, 1.0)


def speed_limited_torque(torque, velocity, limit, max_speed, band):
  """Linear motor torque-speed limit: torque that accelerates a joint in its
  direction of motion fades from full at (max_speed - band) to zero at
  max_speed. Braking torque is unaffected."""
  available = limit * ((max_speed - velocity.abs()) / band).clamp(0.0, 1.0)
  same_direction = torque * velocity > 0
  return torch.where(same_direction, torch.maximum(torch.minimum(torque, available), -available), torque)
