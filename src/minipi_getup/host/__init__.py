"""HoST (Humanoid Standing-up Control, Huang et al. RSS 2025) Mini-Pi ``pi_ground``
reproduction on MuJoCo/MJLab.

Task id ``Mjlab-Getup-HoST-MiniPi``. The environment is a direct port of
``LeggedRobot_Pi`` and trains with the HoST fork of RSL-RL (``minipi_getup.host_rl``), so
it does not go through mjlab's manager-based env or its PPO; use ``host-train``,
``host-play`` and ``host-eval`` instead of ``uv run train/play``.
"""

TASK_ID = "Mjlab-Getup-HoST-MiniPi"
