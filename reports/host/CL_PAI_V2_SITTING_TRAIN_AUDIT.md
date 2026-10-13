# Audit training cl_pai_v2 cho mục tiêu ngồi trước đứng — 2026-10-13

## Đã thay đổi

- Staged pull force khởi đầu 40→30 N; vẫn giảm 10 N mỗi trigger và về 0. Resume giữ state force đã lưu.
- Thêm `target_torso_over_com=+6`: exp(-||torso_xy-COM_xy||²/0.06²) × clip((upright-0.65)/0.35,0,1) × (1-standing_gate) × actuated. Whole-body COM lấy từ subtree_com; torso là origin base_link. Không dùng khoảng cách 3D vì COM nằm trong thân và chiều cao tự nhiên khác origin.
- Reward mới hỗ trợ căn torso trước đứng nhưng không bắt buộc sole chạm đất hoặc mông chịu tải. Không thay thế COM-over-feet/support polygon.
- XML giữ nguyên bản không shell, hash `0ab0e6c4c799273e9fa17d3346ea4907ff9c6517e52763e737e56f10967c6558`. Asset guard và manifest hợp lệ.
- Feet-distance penalty hiện tại là linear, raw scale -400, 3D sole centres >0.4 m. 45 cm=-4 và 50 cm=-8 mỗi step 20 ms; không thêm hard termination.

## Kiểm chứng hình học sau bỏ shell

Đã chạy lại script audit CPU trên chính XML mới, thêm floor vào model kiểm tra trong RAM. Các kết quả trước bỏ shell được giữ riêng.

- Zero pose không có penetration >0.1 mm với enabled collision geoms.
- 100 upright IK attempts: 16 candidates sole gần ground và flat; 10 candidates không penetration. Không nghiệm đạt tiêu chí ở base z=0.058 m trong tập 20 attempts ở chiều cao đó; không chứng minh bất khả thi toàn cục.
- Candidate thấp sạch ở base z=0.08 m; đáy torso visual mesh còn cách ground 22.05 mm. Chưa đạt mông chạm ground; không coi đây là sitting target thành công.
- Đường IK nâng từ candidate thấp lên zero, giữ chân phẳng: 61 mẫu không penetration >0.1 mm. Đường nội suy thẳng góc/chiều cao vẫn xuyên sàn 99/101 mẫu.
- Không có shell false contacts vì shell đã bỏ. Không đồng nghĩa mesh được bảo vệ đầy đủ: visual mesh vẫn contype/conaffinity=0. Đây là kiểm tra collision proxies, không phải triangle-mesh penetration đầy đủ.
- Chưa kiểm chứng dynamics, lực đỡ từng chân, static torque equilibrium hoặc policy learning.

Kết quả: `reports/host/collision_v2/0ab0e6c4c799/audit.json`; hình: `poses_side.png` cùng thư mục. Cấu hình hiện tại: `reports/host/CL_PAI_V2_CURRENT_CONFIG.json`.

## Chưa hợp lý nếu mục tiêu bắt buộc ngồi trước

1. **Vùng mông/dưới hip chưa có collider phù hợp.** Base ground box local z=[0.005,0.155] m trong khi torso mesh xuống -0.05795 m. Chỉ tăng reward không làm mesh visual sinh contact. Mông trên mesh có thể xuyên ground mà base box không báo.
2. **Không đo riêng butt support.** Nonfoot sensor match hip/calf/ankle, không match base; chưa có sitting contact sensor hoặc điểm chuẩn mông. Không thể định nghĩa success ngồi đúng bằng coverage/height hiện tại.
3. **Standing task còn bật.** Task head-height vẫn thưởng tăng chiều cao; height credit floor=0.2 và foot-supported-height reward cho phép rise trước khi sitting đạt chuẩn. Reward mới không mở/đóng stage.
4. **15% stance starts bắt đầu đứng ngay.** Hữu ích cho chỉnh bước sau đứng, nhưng bỏ qua mục tiêu ngồi trước nếu dùng trong giai đoạn sitting-only.
5. **Coverage có load gate.** Score geometry × [1-exp(-relu(F/W-0.01)/0.05)]. Dù sole hoàn toàn sát ground, score >0.9 cần F/W>0.12513 trên mỗi foot. Mông đỡ phần lớn tải thì sole chỉ chạm nhẹ sẽ không đạt; cần tách flatness/near-ground khỏi foot load khi định nghĩa sitting success.
6. **Whole-body COM gần torso chưa đảm bảo COM được hai chân đỡ.** Sitting có mông là support thứ ba. Giai đoạn chuyển COM trước nhấc mông cần COM-over-feet reward và load transfer riêng.
7. **Penalty 40 cm rất mạnh.** Style -4 mỗi step ở 45 cm, trong khi task tối đa khoảng 1 và reward torso-COM tối đa +0.12. Multi-critic normalize advantages nên không suy trực tiếp thành dominance của update, nhưng vẫn có thể làm exploration thu chân quá sớm. Nên theo dõi separation distribution và thành công qua từng orientation; không tự giảm scale đã được user chọn.
8. **ROM mới vs chiến lược yaw.** XML thigh yaw ±2.45 nhưng HoST `style_hip_yaw_deviation=-10` vẫn phạt max |yaw|>1.4 hoặc min |yaw|>0.9. Đây là soft strategy preference hẹp hơn ROM, có thể cản sitting dùng yaw lớn; không phải XML bị đọc sai.
9. **Không shell khiến thigh/shin visual không được bảo vệ đầy đủ.** Thigh ground geom là box 1 mm, shin chỉ có knee housing. Reward leg-contact vẫn có, nhưng chỉ thấy geoms thật sự có contact. Không tự bật lại shell theo yêu cầu đã bỏ.
10. **30 N là assistance đáng kể:** robot nominal 6.939736 kg, khoảng 68.08 N; 30 N ~44% trọng lượng. Pull gate hiện tại bật khi actuated và torso upright (-g_z>0.8), không yêu cầu sitting/COM readiness, nên có thể nhấc robot thay vì học ngồi. Success chuyển stage nên đánh giá thêm force=0.
11. **Training entrypoint chặn dirty experiment code.** Các thay đổi chưa commit sẽ khiến real train dừng trừ khi dùng flag smoke-test `--allow-dirty`. Chưa commit hay khởi chạy training trong audit này.

## Cần thêm cho curriculum ngồi → đứng

- Đặt collider vùng mông phù hợp mesh/thực tế và ground-filtered butt sensor; không dùng lại shell AABB đã chứng minh chặn nhầm.
- Chốt một sitting target khả thi về geometry và torque. Tìm base height/pose từ IK+dynamics, không hardcode chiều cao ngồi dựa trên hình visual.
- Định nghĩa `sitting_success`: torso upright >0.9; từng sole geometric near-ground/flatness >0.9; cả hai có contact; butt contact; vận tốc nhỏ; feet separation <=0.4 m; giữ liên tục 0.5–1 s. COM target ở sitting dùng vùng đỡ có butt, không bắt toàn tải chuyển sang hai chân ngay.
- Sitting-only stage: bỏ standing starts, tắt task height/standing posture objectives, cho phép butt support; giữ mục tiêu chân phẳng, torso upright/COM, và movement regularization.
- Dataset/reset bank từ sitting successes cho stage đứng. Đổi task reward sang đứng; giảm butt support theo load-transfer tiến triển, duy trì sole flat và COM-over-feet.
- Chuyển stage theo EMA tỷ lệ sitting success không assistance, không chỉ iteration; sau đó mix sitting/fallen resets để học chuỗi hoàn chỉnh.
- Nếu một policy cần biết lệnh sitting/standing độc lập ở cùng một state, thêm stage/command vào obs hoặc dùng hai policy riêng. Thêm obs làm đổi network input và export; không tự thay 258-D trong audit này.

Chưa triển khai hard curriculum ngồi hoặc collider mới: cần xác định đúng geometry mông và sitting pose trước. Những sửa đã áp dụng chỉ là reward torso–COM và initial pull force theo yêu cầu.
