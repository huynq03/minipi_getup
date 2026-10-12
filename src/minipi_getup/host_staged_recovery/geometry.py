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


def sole_corners_world(foot_pos: torch.Tensor, foot_quat: torch.Tensor) -> torch.Tensor:
  """(N, 2, 3) foot frames -> (N, 2, 4, 3) world sole corners."""
  local = torch.tensor(SOLE_CORNERS_LOCAL, dtype=foot_pos.dtype, device=foot_pos.device)
  n, feet = foot_pos.shape[:2]
  quat = foot_quat[:, :, None].expand(n, feet, 4, 4).reshape(-1, 4)
  offset = quat_apply(quat, local.expand(n, feet, 4, 3).reshape(-1, 3))
  return foot_pos[:, :, None] + offset.reshape(n, feet, 4, 3)


def support_polygon_distance(
  point: torch.Tensor, vertices: torch.Tensor, valid: torch.Tensor, eps: float = 1e-9
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Distance from a 2D point to the convex hull of the valid vertices.

  point (N, 2); vertices (N, 8, 2); valid (N, 8). Returns (outside distance,
  0 inside; distance to the valid-vertex centroid; any-valid mask). Hull edges
  are the vertex pairs with every other valid vertex on one side, so a single
  foot, two feet, or an edge contact (collinear vertices) are all handled.
  Rows without valid vertices return +inf.
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


def com_centred_score(centre, sigma):
  """exp(-(d/sigma)^2) of the COM distance to the centroid of the loaded sole
  corners (the midpoint between the feet when both are down); 0 without support."""
  return torch.where(torch.isfinite(centre), torch.exp(-(centre / sigma).square()), torch.zeros_like(centre))


def rise_ready_score(feet_under_com, two_foot, flat_min, com_score, centred):
  """Pipeline readiness: feet tucked under the COM, both feet down and flat
  sharing the load, COM over the support and centred between the feet.
  Product, so every step is needed."""
  return feet_under_com * two_foot * flat_min * com_score * centred


def rise_speed_cost(com_vz, threshold, scale, cap):
  """Squared upward COM speed above `threshold`: getting up slowly is free."""
  return (torch.relu(com_vz - threshold) / scale).square().clamp(max=cap)


def torso_lift_cost(base_vz, two_foot_flat, gate, vz_threshold, vz_scale, cap):
  """Raising the base while the robot is not on two flat loaded feet:
  squared excess upward speed x (1 - two_foot_flat) x gate."""
  lift = (torch.relu(base_vz - vz_threshold) / vz_scale).square().clamp(max=cap)
  return lift * (1.0 - two_foot_flat) * gate


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
