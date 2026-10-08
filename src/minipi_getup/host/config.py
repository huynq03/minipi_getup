"""HoST ``pi_ground`` configuration, transcribed from the HoST release.

Sources (read-only reference checkout ``/home/huy/HoST``):

- ``legged_gym/legged_gym/envs/base/base_config.py``          -> ``BaseConfig``
- ``legged_gym/legged_gym/envs/base/legged_robot_config.py``  -> ``LeggedRobotCfg(PPO)``
- ``legged_gym/legged_gym/envs/pi/pi_config_ground.py``       -> ``PiCfg``, ``PiCfgPPO``

Values and class inheritance are copied unchanged, including fields that the Pi
environment never reads (terrain, commands, physx, payload, push), so the two files can
be diffed side by side. Fields that only configure Isaac Gym (``sim.physx``, asset import
options) are kept for reference; ``HOST_MJLAB_PORT_AUDIT.md`` says how each maps to MuJoCo.

Two inheritance details matter at runtime and are preserved by construction:

- ``PiCfg.constraints`` inherits ``LeggedRobotCfg.rewards`` (not ``PiCfg.rewards``), so
  ``constraints.only_positive_rewards`` is the base value ``False``.
- ``PiCfg.rewards.scales`` and ``PiCfg.constraints.scales`` do not inherit, so only the
  terms listed there are active.
"""

# ruff: noqa: E501
import inspect


class BaseConfig:
  def __init__(self) -> None:
    """Initializes all member classes recursively. Ignores all names starting with '__' (built-in methods)."""
    self.init_member_classes(self)

  @staticmethod
  def init_member_classes(obj):
    for key in dir(obj):
      if key == "__class__":
        continue
      var = getattr(obj, key)
      if inspect.isclass(var):
        i_var = var()
        setattr(obj, key, i_var)
        BaseConfig.init_member_classes(i_var)


def class_to_dict(obj) -> dict:
  """legged_gym.utils.helpers.class_to_dict."""
  if not hasattr(obj, "__dict__"):
    return obj
  result = {}
  for key in dir(obj):
    if key.startswith("_"):
      continue
    element = []
    val = getattr(obj, key)
    if isinstance(val, list):
      for item in val:
        element.append(class_to_dict(item))
    else:
      element = class_to_dict(val)
    result[key] = element
  return result


class LeggedRobotCfg(BaseConfig):
  class env:
    num_envs = 4096
    num_observations = 48
    num_privileged_obs = None
    num_actions = 12
    env_spacing = 3.0
    send_timeouts = True
    episode_length_s = 20
    test = False

  class terrain:
    mesh_type = "plane"
    horizontal_scale = 0.1
    vertical_scale = 0.005
    border_size = 25
    curriculum = True
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.0
    measure_heights = True
    measured_points_x = [
      -0.8,
      -0.7,
      -0.6,
      -0.5,
      -0.4,
      -0.3,
      -0.2,
      -0.1,
      0.0,
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7,
      0.8,
    ]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    selected = False
    terrain_kwargs = None
    max_init_terrain_level = 5
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 10
    num_cols = 20
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    slope_treshold = 0.75

  class commands:
    curriculum = False
    max_curriculum = 1.0
    num_commands = 4
    resampling_time = 10.0
    heading_command = True

    class ranges:
      lin_vel_x = [-1.0, 1.0]
      lin_vel_y = [-1.0, 1.0]
      ang_vel_yaw = [-1, 1]
      heading = [-3.14, 3.14]

  class init_state:
    pos = [0.0, 0.0, 1.0]
    rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w
    lin_vel = [0.0, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
    default_joint_angles = {"joint_a": 0.0, "joint_b": 0.0}

  class control:
    control_type = "P"
    stiffness = {"joint_a": 10.0, "joint_b": 15.0}
    damping = {"joint_a": 1.0, "joint_b": 1.5}
    action_scale = 0.5
    decimation = 4

  class asset:
    file = ""
    name = "legged_robot"
    foot_name = "None"
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    disable_gravity = False
    collapse_fixed_joints = True
    fix_base_link = False
    default_dof_drive_mode = 3
    self_collisions = 1
    replace_cylinder_with_capsule = True
    flip_visual_attachments = True
    density = 0.001
    angular_damping = 0.0
    linear_damping = 0.0
    max_angular_velocity = 1000.0
    max_linear_velocity = 1000.0
    armature = 0.0
    thickness = 0.01

  class domain_rand:
    randomize_friction = True
    friction_range = [0.5, 1.25]
    randomize_base_mass = False
    added_mass_range = [-1.0, 1.0]
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 1.0

  class rewards:
    class scales:
      termination = -0.0
      tracking_lin_vel = 1.0
      tracking_ang_vel = 0.5
      lin_vel_z = -2.0
      ang_vel_xy = -0.05
      orientation = -0.0
      torques = -0.00001
      dof_vel = -0.0
      dof_acc = -2.5e-7
      base_height = -0.0
      feet_air_time = 1.0
      collision = -1.0
      feet_stumble = -0.0
      action_rate = -0.01
      stand_still = -0.0

    only_positive_rewards = False
    tracking_sigma = 0.25
    soft_dof_pos_limit = 1.0
    soft_dof_vel_limit = 1.0
    soft_torque_limit = 1.0
    base_height_target = 1.0
    max_contact_force = 100.0

  class normalization:
    class obs_scales:
      lin_vel = 2.0
      ang_vel = 0.25
      dof_pos = 1.0
      dof_vel = 0.05
      height_measurements = 5.0

    clip_observations = 100.0
    clip_actions = 100.0

  class noise:
    add_noise = True
    noise_level = 1.0

    class noise_scales:
      dof_pos = 0.01
      dof_vel = 1.5
      lin_vel = 0.1
      ang_vel = 0.2
      gravity = 0.05
      height_measurements = 0.1

  class viewer:
    ref_env = 0
    pos = [10, 0, 6]
    lookat = [11.0, 5, 3.0]

  class sim:
    dt = 0.005
    substeps = 1
    gravity = [0.0, 0.0, -9.81]
    up_axis = 1

    class physx:
      num_threads = 10
      solver_type = 1
      num_position_iterations = 4
      num_velocity_iterations = 0
      contact_offset = 0.01
      rest_offset = 0.0
      bounce_threshold_velocity = 0.5
      max_depenetration_velocity = 1.0
      max_gpu_contact_pairs = 2**23
      default_buffer_size_multiplier = 5
      contact_collection = 2


class LeggedRobotCfgPPO(BaseConfig):
  seed = 1
  runner_class_name = "OnPolicyRunner"

  class policy:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

  class algorithm:
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

  class runner:
    policy_class_name = "ActorCritic"
    algorithm_class_name = "PPO"
    num_steps_per_env = 50
    max_iterations = 50000
    save_interval = 500
    experiment_name = "test"
    run_name = ""
    resume = False
    load_run = -1
    checkpoint = -1
    resume_path = None


class PiCfg(LeggedRobotCfg):
  class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.351]
    rot = [0.0, -1, 0, 1.0]  # x,y,z,w [quat], not normalized in the release
    target_joint_angles = {
      "l_hip_pitch_joint": -0.0,
      "l_hip_roll_joint": 0.0,
      "l_thigh_joint": 0.0,
      "l_calf_joint": 0.0,
      "l_ankle_pitch_joint": -0.0,
      "l_ankle_roll_joint": 0,
      "r_hip_pitch_joint": -0.0,
      "r_hip_roll_joint": 0.0,
      "r_thigh_joint": 0.0,
      "r_calf_joint": 0.0,
      "r_ankle_pitch_joint": -0.0,
      "r_ankle_roll_joint": 0,
    }
    default_joint_angles = {
      "l_hip_pitch_joint": -0.0,
      "l_hip_roll_joint": 0.0,
      "l_thigh_joint": 0.0,
      "l_calf_joint": 0.0,
      "l_ankle_pitch_joint": -0.0,
      "l_ankle_roll_joint": 0,
      "r_hip_pitch_joint": -0.0,
      "r_hip_roll_joint": 0.0,
      "r_thigh_joint": 0.0,
      "r_calf_joint": 0.0,
      "r_ankle_pitch_joint": -0.0,
      "r_ankle_roll_joint": 0,
    }

  class env(LeggedRobotCfg.env):
    num_one_step_observations = 43  # 3+3+3*12+1
    num_actions = 12
    num_dofs = 12
    num_actor_history = 6
    num_observations = num_actor_history * num_one_step_observations
    episode_length_s = 10
    unactuated_timesteps = 30

  class control(LeggedRobotCfg.control):
    control_type = "P"
    stiffness = {
      "hip_pitch": 30,
      "hip_roll": 15,
      "thigh": 15,
      "calf": 30,
      "ankle_pitch": 12,
      "ankle_roll": 5,
    }
    damping = {
      "hip_pitch": 0.2,
      "hip_roll": 0.2,
      "thigh": 0.2,
      "calf": 0.2,
      "ankle_pitch": 0.2,
      "ankle_roll": 0.2,
    }
    # target angle = actionRescale * action + cur_dof_pos
    action_scale = 1
    decimation = 4

  class terrain:
    mesh_type = "plane"
    horizontal_scale = 0.1
    vertical_scale = 0.005
    border_size = 25
    curriculum = True
    static_friction = 0.8
    dynamic_friction = 0.7
    restitution = 0.3
    measure_heights = True
    measured_points_x = [
      -0.8,
      -0.7,
      -0.6,
      -0.5,
      -0.4,
      -0.3,
      -0.2,
      -0.1,
      0.0,
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7,
      0.8,
    ]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    selected = False
    terrain_kwargs = None
    max_init_terrain_level = 5
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 1
    num_cols = 20
    terrain_proportions = [1, 0.0, 0, 0, 0]
    slope_treshold = 0.75

  class asset(LeggedRobotCfg.asset):
    file = "pi_12dof_release_v1.urdf"  # converted to assets/pi_12dof_host.xml
    name = "Pi"
    left_foot_name = "l_ankle_pitch"
    right_foot_name = "r_ankle_pitch"
    left_knee_name = "l_calf"
    right_knee_name = "r_calf"
    foot_name = "ankle_roll"
    penalize_contacts_on = ["calf", "hip"]
    terminate_after_contacts_on = []

    left_leg_joints = [
      "l_hip_pitch_joint",
      "l_hip_roll_joint",
      "l_thigh_joint",
      "l_calf_joint",
      "l_ankle_pitch_joint",
      "l_ankle_roll_joint",
    ]
    right_leg_joints = [
      "r_hip_pitch_joint",
      "r_hip_roll_joint",
      "r_thigh_joint",
      "r_calf_joint",
      "r_ankle_pitch_joint",
      "r_ankle_roll_joint",
    ]
    left_hip_joints = ["l_thigh_joint"]
    right_hip_joints = ["r_thigh_joint"]
    left_hip_roll_joints = ["l_hip_roll_joint"]
    right_hip_roll_joints = ["r_hip_roll_joint"]
    left_hip_pitch_joints = ["l_hip_pitch_joint"]
    right_hip_pitch_joints = ["r_hip_pitch_joint"]
    left_knee_joints = ["l_calf_joint"]
    right_knee_joints = ["r_calf_joint"]
    knee_joints = ["l_calf_joint", "r_calf_joint"]
    ankle_joints = [
      "l_ankle_pitch_joint",
      "l_ankle_roll_joint",
      "r_ankle_pitch_joint",
      "r_ankle_roll_joint",
    ]

    keyframe_name = "keyframe"
    head_name = "keyframe_head"
    trunk_names = ["base_link"]
    base_name = "base_link"
    left_lower_body_names = ["l_hip_pitch", "l_ankle_roll", "l_calf"]
    right_lower_body_names = ["r_hip_pitch", "r_ankle_roll", "r_calf"]
    left_ankle_names = ["l_ankle_roll"]
    right_ankle_names = ["r_ankle_roll"]

    density = 0.001
    angular_damping = 0.01
    linear_damping = 0.01
    max_angular_velocity = 1000.0
    max_linear_velocity = 1000.0
    armature = 0.01
    thickness = 0.01
    self_collisions = 0  # 0 = enabled
    flip_visual_attachments = False

  class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9
    soft_dof_vel_limit = 0.9
    base_height_target = 0.34
    only_positive_rewards = False
    orientation_sigma = 1
    is_gaussian = True
    target_head_height = 0.37
    target_head_margin = 0.37
    target_base_height_phase1 = 0.25
    target_base_height_phase2 = 0.25
    target_base_height_phase3 = 0.34
    orientation_threshold = 0.99
    left_foot_displacement_sigma = -2
    right_foot_displacement_sigma = -2
    target_dof_pos_sigma = -0.1
    tracking_sigma = 0.25

    reward_groups = ["task", "regu", "style", "target"]
    num_reward_groups = len(reward_groups)
    reward_group_weights = [2.5, 0.1, 1, 1]

    class scales:
      task_orientation = 1
      task_head_height = 1

  class constraints(LeggedRobotCfg.rewards):
    is_gaussian = True
    target_head_height = 0.37
    target_head_margin = 0.37
    orientation_height_threshold = 0.9
    target_base_height = 0.34
    left_foot_displacement_sigma = -2
    right_foot_displacement_sigma = -2
    hip_yaw_var_sigma = -2
    target_dof_pos_sigma = -0.1
    post_task = False

    class scales:
      # regularization reward
      regu_dof_acc = -2.5e-7
      regu_action_rate = -0.01
      regu_smoothness = -0.01
      regu_torques = -2.5e-6
      regu_joint_power = -2.5e-5
      regu_dof_vel = -1e-3
      regu_joint_tracking_error = -0.00025
      regu_dof_pos_limits = -100.0
      regu_dof_vel_limits = -1
      # style reward
      style_hip_yaw_deviation = -10
      style_hip_roll_deviation = -10
      style_left_foot_displacement = 2.5
      style_right_foot_displacement = 2.5
      style_knee_deviation = -0.25
      style_ground_parallel = 20
      style_feet_distance = -10
      style_style_ang_vel_xy = 1
      # post-task reward
      target_ang_vel_xy = 10
      target_lin_vel_xy = 10
      target_feet_height_var = 2.5
      target_target_orientation = 10
      target_target_base_height = 10

  class domain_rand:
    use_random = True

    randomize_actuation_offset = use_random
    actuation_offset_range = [-0.05, 0.05]
    randomize_motor_strength = use_random
    motor_strength_range = [0.9, 1.1]
    randomize_payload_mass = use_random
    payload_mass_range = [-2, 3]
    randomize_com_displacement = use_random
    com_displacement_range = [-0.03, 0.03]
    randomize_link_mass = use_random
    link_mass_range = [0.8, 1.2]
    randomize_friction = use_random
    friction_range = [0.1, 1]
    randomize_restitution = use_random
    restitution_range = [0.0, 1.0]
    randomize_kp = use_random
    kp_range = [0.85, 1.15]
    randomize_kd = use_random
    kd_range = [0.85, 1.15]
    randomize_initial_joint_pos = True
    initial_joint_pos_scale = [0.9, 1.1]
    initial_joint_pos_offset = [-0.1, 0.1]
    push_robots = True  # never applied: the push call is commented out in the release
    push_interval_s = 10
    max_push_vel_xy = 0.5
    delay = use_random
    max_delay_timesteps = 5

  class curriculum:
    pull_force = True
    force = 15
    dof_vel_limit = 300
    base_vel_limit = 20
    threshold_height = 0.37
    no_orientation = False

  class sim:
    dt = 0.005
    substeps = 1
    gravity = [0.0, 0.0, -9.81]
    up_axis = 1

    class physx:
      num_threads = 10
      solver_type = 1
      num_position_iterations = 8
      num_velocity_iterations = 1
      contact_offset = 0.01
      rest_offset = 0.0
      bounce_threshold_velocity = 0.5
      max_depenetration_velocity = 1.0
      max_gpu_contact_pairs = 2**23
      default_buffer_size_multiplier = 5
      contact_collection = 2


class PiCfgPPO(LeggedRobotCfgPPO):
  runner_class_name = "OnPolicyRunner"

  class policy:
    init_noise_std = 0.8
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256]

  class algorithm(LeggedRobotCfgPPO.algorithm):
    entropy_coef = 0.01
    value_smoothness_coef = 0.1
    smoothness_upper_bound = 1.0
    smoothness_lower_bound = 0.1

  class runner(LeggedRobotCfgPPO.runner):
    run_name = ""
    save_interval = 100
    experiment_name = "Pi_ground"
    algorithm_class_name = "PPO"
    init_at_random_ep_len = True
    max_iterations = 12000
