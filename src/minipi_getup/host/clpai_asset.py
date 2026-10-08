"""Use the asset-zoo Mini-Pi plant with HoST's reference-frame measurements."""
from pathlib import Path

import minipi_getup.host.env as host_env

CLPAI_XML = Path(__file__).resolve().parents[1] / 'asset_zoo/robots/hightorque_minipi/xmls/cl_pai.xml'
_HOST_FACTORY = host_env.get_robot_cfg


def use_clpai_asset():
  """Configure this process only; source XML/meshes are never written."""
  host_env.ASSET_XML = CLPAI_XML

  def factory(limit_solref=None):
    cfg = _HOST_FACTORY(limit_solref)
    source = cfg.spec_fn

    def spec_fn():
      spec = source()
      base = next(b for b in spec.bodies if b.name == 'base_link')
      base.add_body(name='keyframe_head_link', pos=[0, 0, .08])
      for side in ('r', 'l'):
        foot = next(b for b in spec.bodies if b.name == f'{side}_ankle_roll_link')
        for i, pos in enumerate(([-.04, .1, 0], [-.04, -.1, 0], [.07, 0, 0], [-.15, 0, 0]), 1):
          foot.add_body(name=f'auxiliary_{side}_ankle_roll_link{i}', pos=pos)
      return spec

    cfg.spec_fn = spec_fn
    return cfg

  host_env.get_robot_cfg = factory
  return CLPAI_XML
