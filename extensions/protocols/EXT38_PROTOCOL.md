# EXT38: vòng xác nhận 7, quy tắc chuyển dựa trên chẩn đoán D (`rpt_auto_d`). Chốt ngày 2026-11-15, TRƯỚC khi train và chạy

**Cơ sở (EXT37, thăm dò):**
- **Chẩn đoán D** = tỷ lệ lạc quan của predictor trên hai bản perturbed của window calibration sạch, trừ tỷ lệ lạc quan khi sạch. Dùng cùng generator với RPT, seed + 53000 và seed + 54000.
- D > 0 ở toàn bộ run C-MAPSS và pin, và 95% run N-CMAPSS. D < 0 ở toàn bộ 80 run HDD (ba tập model).
- Quy tắc "dùng `rpt_L` khi và chỉ khi D > 0" giữ lợi ích trên C-MAPSS, N-CMAPSS và pin; giữ khoảng một nửa lợi ích trên vòng bi; không gây hại trên mọi tập HDD.
- Ngưỡng 0 là ngưỡng tự nhiên ("perturbation có tạo ra lỗi lạc quan không") và đã được ghi trước trong EXT37. Không có tham số nào được tune.

## Phương pháp
`rpt_auto_d` = `rpt_L` nếu D > 0, ngược lại `full`. D chỉ dùng unit calibration, không dùng gì từ test.

## Dữ liệu
- **Seed mới 152–161.** GRU và TCN cho C-MAPSS FD001–4, N-CMAPSS, vòng bi, pin; cấu hình như vòng 5 và 6; train trên GPU.
- **hdd4:** 4 model ổ cứng hạng 13–16 theo số ổ hỏng đủ điều kiện, chưa từng dùng, cùng cách dựng `prepare_hdd.py`. `configs/external_hdd4.json`, seed 152–161.

## Kiểm định xác nhận (α_7 = 0.000390625; tổng α sau vòng = 0.049609375)
- **Họ kiểm định:** `rpt_auto_d` − `full`; risk @90% matched, trung bình 6 lỗi; 5 họ (4 benchmark + hdd4). Wilcoxon hai phía, Holm trên 5.
- **Thành công:**
  - giảm có ý nghĩa ở ≥ 1 họ;
  - không họ nào tăng có ý nghĩa;
  - hiệu trung bình ≤ +0.3 ở mọi họ;
  - tỷ lệ bi quan tăng ≤ 1 điểm.
- **Phụ, đã chốt:** không kém trên hdd4. Bootstrap theo seed, cận trên của CI 90% cho `rpt_auto_d` − `full` < +0.3.
- **Mô tả:**
  - `rpt_auto` (R0) và `rpt_L`;
  - phân phối của D theo họ;
  - tỷ lệ run chuyển sang `rpt_L`.

## Kết quả, ngày 2026-11-15: XÁC NHẬN
C-MAPSS −0.64, N-CMAPSS −0.50 (có ý nghĩa ở α_7). Không kém trên hdd4. Chi tiết xem FINDINGS.md.
