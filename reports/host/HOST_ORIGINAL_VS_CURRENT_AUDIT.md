# Audit HoST gốc và code hiện tại — 2026-10-13

## Phạm vi và bằng chứng

- HoST gốc: `/home/huy/ws/HoST`, commit `70bb580949a336a920833700e4b5dc3bf7fe87ce`.
- Repo hiện tại: `/home/huy/ws/minipi_getup_host`, HEAD `2050c04e068ae2228d11dc6f3756e50390333ac1`, có thay đổi chưa commit.
- Đường train được audit: `tools/train_host_staged.sh` → `host_staged_recovery.train` → `StagedRecoveryEnv` → `SafeRecoveryEnv` → `LeggedRobot_Pi`.
- So sánh trực tiếp source gốc, diff RL, AST bỏ import/comment, cấu hình có kế thừa; kiểm tra XML bằng MuJoCo và gọi asset guard. Không chạy lại training hay đánh giá chất lượng policy.
- `model_2600.pt` được train với XML cũ; XML đang mở đã sửa sau khi train. Không đồng nhất hai phiên bản này.
- Không áp kết luận này cho các task `getup` hoặc `ftsr`: chúng là các đường thực thi riêng.

## 1. Thuật toán HoST: phần giữ nguyên và phần thêm

| Thành phần | Kết quả so sánh trực tiếp |
|---|---|
| ActorCritic | AST giống hệt HoST gốc sau khi bỏ imports; chỉ thêm comment nguồn |
| RolloutStorage, GAE, chuẩn hóa advantage theo group | AST giống hệt sau khi bỏ imports |
| PPO core | Giữ công thức loss, clipping, smoothness, adaptive KL/LR; thêm thống kê và kiểm tra non-finite, std <= 0 |
| Network | Actor 258→512→256→128→12, ELU và tanh cuối; 4 critic 258→512→256→1 |
| Critic observation | Vẫn cùng 258 giá trị như actor; không thêm privileged obs |
| Reward groups | Vẫn task/regu/style/target; weights [2.5, 0.1, 1, 1] |
| PPO cấu hình | `PiCfgPPO` và `StagedRecoveryPPOCfg` không có khác biệt giá trị so với config gốc có kế thừa |

PPO vẫn clip 0.2, gamma 0.99, lambda 0.95, 5 epochs, 4 minibatches, LR 1e-3 adaptive, desired KL 0.01, entropy 0.01, grad norm 1.0, rollout 50 steps/env.

**Thay đổi hành vi learning ngoài PPO core:** safe/staged train cài `install_action_std_guard`; clamp std về [0.05, 1.0] sau mỗi update hoàn chỉnh và lúc cài guard. HoST gốc không có bước này. Vì vậy không thể gọi toàn bộ training hiện tại là hoàn toàn không đổi thuật toán.

Runner thêm TensorBoard, monitor, dump NaN, snapshot model/optimizer trước update; checkpoint có `it` và adaptive learning rate. Load ưu tiên `it` thay vì `iter` cũ và dùng device hiện tại thay vì hardcode cuda:0. Safe/staged checkpoint thêm variant, asset hash, seed, std bounds, curriculum state (force/action scale/success EMA); resume khác số env dùng mean của tensor đã lưu.

Giới hạn kế thừa: observation history không clear trực tiếp ở reset; transition filter theo norm obs toàn batch và timeout infos có thể stale vẫn còn. Recurrent storage vẫn tham chiếu `split_and_pad_trajectories` nhưng import đã bỏ: không ảnh hưởng actor feedforward hiện tại, sẽ lỗi nếu dùng recurrent branch.

## 2. Observation/action: layout giữ nguyên, động lực học đổi

- Một frame: [angular velocity body ×0.25 (3), gravity body (3), absolute q (12), qdot ×0.05 (12), previous action (12), action scale + U(-0.025,0.025) (1)].
- Sáu frame oldest-first = 258; clip obs ±100. Gravity là Rᵀ[0,0,-1]. Root frame là base_link.
- Không thêm COM hay sole coverage vào observation: chúng hiện là dữ liệu reward/monitor.
- Action clipping ±100; deterministic actor tanh trong [-1,1], nhưng action sampled lúc train có Gaussian noise và có thể vượt [-1,1].
- Giữ luật target = current q + delayed(action × scale); không phải default q + delta.
- Delay random 0/5/10/15/20 ms, mỗi episode một giá trị chung cho 12 khớp. Buffer update mỗi 5 ms và reset bằng 0.
- Giữ passive phase 30 policy steps, điều kiện action bật là counter >30. Action/frame mới bằng 0 nhưng damping và torque offset vẫn có thể tác động. Stance starts mới bỏ qua passive phase.

## 3. Robot và controller thay đổi

HoST gốc dùng `pi_12dof_release_v1.urdf`. Baseline port có XML chuyển từ URDF đó; staged train chuyển sang asset CL_PAI/HoST hybrid `cl_pai_host_torso_host_legs_hip120_legcoll.xml`. Đây không chỉ là đổi đuôi file: kinematics, inertials, collision và ROM khác baseline.

| Loại khớp | Kp gốc → hiện tại | Kd gốc → hiện tại | Effort gốc → hiện tại |
|---|---|---|---|
| hip_pitch | 30 → 60 | 0.2 → 2.4 | 20 → 16 |
| hip_roll | 15 → 40 | 0.2 → 0.8 | 20 → 16 |
| thigh | 15 → 20 | 0.2 → 0.4 | 20 → 16 |
| calf | 30 → 60 | 0.2 → 2.8 | 20 → 16 |
| ankle_pitch | 12 → 30 | 0.2 → 1.6 | 20 → 16 |
| ankle_roll | 5 → 10 | 0.2 → 0.3 | 20 → 6 |

Staged còn thêm torque-speed envelope: torque cùng chiều qdot bị giới hạn bởi T × clip((4-|qdot|)/1,0,1). Full torque tới 3 rad/s, giảm về 0 ở 4 rad/s; braking giữ torque clip thường.

Asset bổ sung shell collision giữa chân, chân–thân, bàn chân–hip cùng bên; shell massless. Hip roll mở rộng tới ~120° phía ngoài; calf cho phép gập ngược. Collision mask có chọn lọc, không phải mọi shell đều va chạm mặt đất hoặc mọi cặp cùng chân.

### XML vừa sửa trong phiên chat này

| Joint | Trước sửa | Hiện tại (rad) |
|---|---|---|
| r/l hip_pitch | [-1.25, 1.75] | [-2, 2] |
| r thigh | [-0.28, 0.30] | [-2, 2] |
| l thigh | [-0.30, 0.28] | [-2, 2] |
| r/l calf | [-2.5, 1.65] | [-2.5, 1.4] |

Các ROM này không phải ROM đã dùng để train model_2600. MuJoCo compile thành công chỉ xác nhận XML hợp lệ, không xác nhận mọi tư thế trong ROM mới tránh collision.

## 4. Mô phỏng thay đổi

| Tham số | HoST gốc | Port hiện tại |
|---|---|---|
| Engine | Isaac Gym / PhysX | MJLab / MuJoCo Warp |
| Controller / policy | 5 ms / 20 ms | Giữ 5 ms / 20 ms |
| Physics integration | 5 ms, PhysX substeps=1 | 2 × 2.5 ms mỗi controller update |
| Solver | TGS, position iterations 8, velocity iterations 1 | Newton, iterations 100, line search 50, tolerance 1e-8 |
| Integrator | PhysX | implicitfast |
| Contact | PhysX contact/restitution | MuJoCo soft contact, condim=3, pyramidal, impratio=1 |
| Friction | Static/dynamic ground 0.8/0.7, average combine | Robot priority=1; mu=(U(0.1,1)+0.75)/2 = [0.425,0.875] |
| Friction evaluation | HoST tùy DR config | Khi tắt random: mu=0.65 |
| Restitution | Random [0,1] có hiệu lực PhysX | Có sample nhưng không áp dụng tương đương |
| Body linear/angular damping | 0.01 | Không mô phỏng trực tiếp |
| Joint armature | 0.01 | 0.01, được thêm bằng Python |
| Passive joint damping/frictionloss | Khác API engine | XML compiled hiện tại là 0/0 |

MuJoCo compile scene kiểm chứng: floor priority=0, friction=[1,0.005,0.0001]; robot priority=1, friction trước random=[1,0.005,0.0001], sau random thay hệ số đầu bằng mu ở trên. Các trường physx trong config.json không phải option của simulator đang chạy.

Domain randomization phần lớn giữ range gốc: Kp/Kd factors [0.85,1.15], motor strength [0.9,1.1], offset ±5% torque limit, link mass [0.8,1.2], COM displacement. Logic inherited vẫn overwrite payload mass khi randomize_link_mass bật; không coi payload [-2,3] là thay đổi khối lượng thực sự cộng thêm trong cấu hình này.

## 5. Reset và curriculum khác HoST gốc

- HoST gốc reset chủ yếu supine, joint noise ±0.1 rad.
- Safe recovery thêm 5 lớp: supine/prone/left/right/random tilt; random yaw, vị trí và vận tốc đầu.
- Staged reset q quanh default với absolute half-span cố định: hip_pitch 0.75, hip_roll 0.4227, thigh 0.225, calf 0.575, ankle_pitch 0.45, ankle_roll 0.275 rad; clip cách ROM 0.02 rad. ROM mới không tự mở rộng reset distribution.
- Thêm 15% stance starts từ thư viện 64 standing poses; được điều khiển ngay, không pull force, loại khỏi penalty EMA/fallen metrics.
- Pull force gốc 15 N, mỗi curriculum trigger giảm 20 N. Safe/staged khởi đầu 40 N, mỗi trigger giảm 10 N.
- Action scale gốc 1→0.25, giảm 0.02. Safe thêm finetune xuống 0.10 bằng bước 0.005 khi đủ thành công; staged clamp floor 0.25 nên scale thực tế không xuống dưới 0.25.
- Thêm success EMA trên đứng có hai chân đỡ, cửa sổ 0.5 s, ít nhất 80% số mẫu phù hợp. Penalty tăng smoothstep khi EMA từ 0.60→0.85.
- Staged velocity termination được bật: threshold 12→5 rad/s theo curriculum, đo peak mỗi controller update. Drag termination có code nhưng hiện tắt.

**Chưa có curriculum “mông chạm đất → ngồi ổn định → đứng” đã trao đổi.** Staged hiện là soft shaping; stage monitor không bắt buộc ngồi, COM giữa, sole >90% rồi mới mở reward đứng.

## 6. Reward thay đổi lớn

Task vẫn nhân orientation và head-height, nhưng head-height/target-base-height đã nhân thêm credit theo foot support, COM và readiness. Vì vậy dù cấu trúc task còn giống, công thức reward hiệu dụng không còn nguyên bản.

Regu: action_rate và smoothness từ -0.01→-0.03; modulation theo success curriculum. Overspeed thêm rồi chuyển sang style.

Style thêm/chỉnh: ballistic ascent (-80), knee load (-45), rotation (-2), sole slip (-6), loaded sole coverage deficit (-10), sole tilt (-8), loaded foot translation (-4), standing anchor (-4), stance width (-4), unsupported ascent (-6), one-foot rise (-20), rise speed (-4), standing knee extension (-16), load imbalance (-6), leg contact (-6), joint overspeed (-2). Hip-roll deviation gốc -10 được tắt. Một số joint-specific shaping của safe recovery cũng tắt ở staged.

Target thêm: foot load (+8), sole coverage (+8), supported height (+8), feet under COM (+3), two-foot flat support (+6), COM support (+6), COM margin (+2), torso center (+4), stance return (+4), touchdown placement (+100); default posture/stable standing penalties cũng được thêm và có state/curriculum gates.

Scale constraint là raw scale, còn nhân dt=0.02; không so trực tiếp +100 touchdown với task product theo hệ số raw.

Coverage staged là score hình học: 10 điểm sole/foot, mean(exp(-(relu(z)/0.003)^2)) × load gate. Đây là proxy độ phẳng/gần đất có tải, không phải phép đo 90% diện tích tiếp xúc vật lý. Readiness dùng mix mean/min coverage (mean weight 0.3), nên không phải hard check cả hai sole >90%.

## 7. Các vấn đề cần xử lý trước train mới

1. **Asset hash guard đang chặn train.** Expected hash: `ea7d909d9a0516d18f84d43a9bbbc5a1d03b109e11e15e1793dd527fe439ebb0`. XML hiện tại: `d98003fd1414dc2cedfef5e1fbbd0f4b81a56886bdf8248728e28a6fd4221c04`. Đã gọi configure_asset và xác nhận RuntimeError. Audit không tự cập nhật guard hoặc làm checkpoint cũ giả tương thích.
2. **Đổi ROM chưa cập nhật toàn bộ tài liệu/reset library assumptions.** Reset spans vẫn cố định; thư viện stance sinh từ asset mới cần kiểm tra lại khả thi/collision trước sử dụng.
3. **Mục tiêu ngồi chưa triển khai.** Reward đứng và nonfoot-support còn tồn tại; cần một variant/config riêng với sitting success, allowed butt support, gating, reset curriculum.
4. **Checkpoint/ONNX model_2600 thuộc asset cũ.** Không mặc định coi policy đó được huấn luyện cho ROM mới.
5. **Audit không xác nhận khả năng ngồi/đứng sau sửa ROM.** Không rollout training; các nhận xét về tác động là phân tích source, không phải kết quả thực nghiệm mới.

## Các file chính

- Gốc: `HoST/rsl_rl/rsl_rl/{modules/actor_critic.py,algorithms/ppo.py,storage/rollout_storage.py,runners/on_policy_runner.py}`.
- Gốc env/config: `HoST/legged_gym/legged_gym/envs/pi/{pi_host_ground.py,pi_config_ground.py}` và `envs/base/legged_robot_config.py`.
- Hiện tại baseline: `src/minipi_getup/host/{env.py,config.py}` và `host_rl/`.
- Delta safe: `host_safe_recovery/{asset.py,config.py,env.py,optimization.py}`.
- Delta staged: `host_staged_recovery/{asset.py,config.py,env.py,geometry.py,stance_poses.py,train.py,REWARDS.md}`.
- Công cụ mới: export ONNX, environment activation/checks, recovery/side/drag analysis, training notification và báo cáo. Chúng không tự thay PPO; hành vi phụ thuộc entrypoint được gọi.
