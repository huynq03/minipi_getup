# Kiểm chứng collision cl_pai_v2

XML được giữ nguyên. SHA256: `67de30a8ca19d0e9fb059246adfb5b3bba7779fca0d5ad852da6d8493d17f444`.

## Kết luận

Shell chưa phù hợp để mặc định coi mọi tư thế ngồi hợp lệ sẽ đi qua được. Có bằng chứng shell hip–foot báo va chạm giả so với mesh. Có đường nâng thân với chân phẳng không penetration, nhưng chưa tìm được tư thế đúng yêu cầu mông chạm đất và COM giữa hai sole.

## Phương pháp và giới hạn

- MuJoCo CPU, static forward/contact, thêm plane chỉ trong model kiểm tra; không ghi plane vào XML.
- Penetration ở đây là khoảng cách contact âm quá 0.1 mm của collision geom đang bật; không phải kiểm tra toàn bộ triangle mesh.
- 100 bounded IK attempts: thân thẳng đứng, base z = 0.058/0.08/0.10/0.12/0.15 m, hai foot midpoint x = -0.18/-0.10/0/0.10/0.18 m, khoảng cách chân 0.16/0.24 m, hai nhánh khởi tạo gối.
- Candidate hợp lệ về chân: tilt <1 độ, sole centre cách ground <1 mm. Không có dynamics/load nên không khẳng định score coverage có load gate >90% hoặc khả năng giữ tĩnh bằng torque hiện tại.
- 2000 random ROM poses, seed=0, để tìm ví dụ va chạm; không dùng tỷ lệ đó làm tỷ lệ thất bại training.
- So sánh mesh bằng `mj_geomDistance` trên convex hull. Gap dương chứng minh mesh gốc tách nhau; overlap convex hull không đủ để kết luận triangle mesh xuyên nhau.
- Tư thế ngồi cụ thể chưa được cung cấp bằng joint angles. Đây là tìm kiếm hữu hạn với torso upright, không chứng minh không tồn tại nghiệm khi cho phép nghiêng thân hoặc hướng chân khác.

## 1. Ngồi có bị shell cản?

- 16/100 IK candidates có hai sole phẳng/gần ground; 9 candidate không có penetration quá 0.1 mm.
- Candidate sạch thấp nhất: base z=0.08 m; foot centres x=+0.18, y=±0.08 m; sole height sai số dưới 0.001 mm; tilt gần 0.
- Tuy nhiên đáy torso mesh còn cách ground **22.05 mm** và COM x lệch foot midpoint **106.83 mm**. Đây không phải tư thế mông chạm đất/COM giữa sole đã yêu cầu.
- Base z=0.058 m (đáy torso mesh gần sàn): không có candidate chân phẳng trong 20 attempts ở chiều cao này. Đây chỉ là kết quả tìm kiếm, không phải chứng minh bất khả thi.
- Candidate gập ngược, base z=0.08, foot centres x=-0.18, y=±0.08: hai sole phẳng nhưng hip–foot shell penetration **1.925 mm (R)** / **1.676 mm (L)**. Trong khi visual convex hull gap **20.829 mm** cả hai bên. Vì hull vẫn tách nhau, shell đang chặn quá sớm tại tư thế này.
- Các qpos cụ thể được lưu trong `audit.json`: `lowest_collision_free_candidate` và `blocked_candidate`.

## 2. Zero và chuyển tiếp

- Zero joints, robot trên không: không penetration ở collision geoms đang bật.
- Zero joints, sole tangent ground tại base z=0.345112 m: không penetration quá 0.1 mm.
- Nội suy thẳng joint angles/base height từ candidate thấp sạch lên zero: **0/101 frame self-penetration**, nhưng **99/101 frame penetration với ground**, sâu nhất **43.31 mm**. Không dùng nội suy này làm quỹ đạo chân trồng trên sàn.
- IK continuation giữ sole position và flat orientation, nâng base từ 0.08 lên 0.345112 m, cho base x đi theo: **0/61 frame penetration**, max IK residual **3.29e-8** (weighted units). Đây là đường hình học sạch được sample, không chứng minh cân bằng động hoặc đường ngồi thật.
- 226/2000 random ROM poses có penetration. ROM limits không phải giới hạn tránh self-collision; nhiều tổ hợp góc trong ROM vẫn đụng nhau là bình thường.

## 3. Shell có bao sát mesh?

- Thigh/shin/foot shells gần như chứa toàn bộ mesh vertices trong box AABB; hip shell phải thiếu tối đa khoảng **0.05 mm**. Điều này kiểm tra mức bao phủ, không chứng minh box sát mọi bề mặt.
- Hình hộp lấp luôn các vùng rỗng/lõm của mesh. Ví dụ hip–foot ở trên chứng minh việc fit AABB vẫn tạo false contact lớn.
- Torso mesh AABB local: x [-86.15,85.77], y [-50.20,50.20], z [-57.95,157.00] mm.
- Torso shell local: x [-82.5,82.5], y [-50,50], z [5,155] mm.
- Shell torso bỏ hở phần dưới hip tới **62.95 mm** so với mesh. Ground torso box cũng có cùng bounds. Vùng dưới thân này không được torso box mô phỏng khi muốn mông chạm ground.
- Không có shell/collision riêng cho hip_pitch body; visual mesh của body đó không tạo contact.
- Mask hiện tại chỉ bật foot–hip cùng bên, không bật shin–hip cùng bên qua shell. Các shell không tham gia ground contact; ground vẫn dùng geom cũ và sole capsules.

## Hướng sửa dựa trên bằng chứng

1. Sửa proxy hip–foot bằng shape bám bề mặt hơn hoặc decomposition, thay vì thu nhỏ toàn bộ box tùy ý: cần tránh bỏ sót va chạm thật.
2. Bổ sung collider phù hợp cho vùng mông/dưới hip nếu muốn train mông chạm ground; chọn theo hình học robot thật, không dùng riêng torso upper box hiện tại.
3. Chốt một seated target bằng qpos/pose và kiểm tra cả contact, COM, load balance và torque khả thi trước reward/curriculum ngồi.

Chưa sửa XML hoặc hash; chưa chạy training. Script tái lập: `source tools/activate_host_env.sh` rồi `python tools/audit_v2_collisions.py`. Hình `poses_side.png` chỉ là phép chiếu sagittal của mesh hull (xám) và shell box (cam), không thể hiện độ sâu y hoặc mô phỏng lực.
