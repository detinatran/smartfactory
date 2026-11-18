# EXT37: (a) chẩn đoán trực tiếp khi nào perturbation có ích; (b) đường cong học theo số unit selector. Chốt ngày 2026-11-15, TRƯỚC khi chạy. Thăm dò

**Lý do (review 11, W6 và W7):**
- RPT gây hại đúng ở HDD, họ có nhiều unit selector nhất (296). Đây có thể là dấu hiệu RPT là bộ điều chuẩn cho dữ liệu nhỏ.
- RPT-auto chọn dựa trên một argmin nhiễu. Một chẩn đoán dựa trên cơ chế sẽ có nguyên tắc hơn: các bản perturbed của dữ liệu calibration có thực sự làm predictor lạc quan hơn không.

## Dữ liệu
- Benchmark vòng 4: seed 92–101, 240 run.
- HDD gốc: seed 92–101 và 112–121, 40 run.
- hdd2: seed 102–111, 20 run.
- hdd3: seed 142–151, 20 run.

Tổng 320 run, không train lại predictor.

## Script `scripts/ext37.py`
Selector giống ext22.py. Với mỗi run:
- **Chẩn đoán D** = tỷ lệ lạc quan của predictor (trọng số engine) trên hai bản perturbed của window calibration sạch, trừ tỷ lệ lạc quan trên chính các window đó khi sạch. Bản perturbed dùng cùng generator với RPT, seed + 53000 và seed + 54000. D chỉ dùng nhãn calibration, không dùng gì từ test.
- **Risk test** @90% matched, trung bình 6 lỗi, cho `full` và `rpt_L`.
- **Đường cong học** (chỉ với HDD, hdd2, hdd3): lấy mẫu con n ∈ {15, 30, 60, 120} unit selector (rng seed + n). Train lại `full` và `rpt_L` chỉ trên các unit đó (S và các bản perturbed của chúng), rồi tính risk test như trên.

## Phân tích (thăm dò)
- **L1:** với mỗi tập HDD, `rpt_L` − `full` theo n (15, 30, 60, 120, tất cả). Wilcoxon theo cặp tại từng n; mô tả xu hướng.
  - Giả thuyết ghi trước: hiệu âm (RPT có ích) ở n nhỏ, tiến về dương khi n lớn.
- **D1:**
  - Tương quan Spearman giữa D và `rpt_L` − `full`, trên toàn bộ 320 run và trong từng họ.
  - Quy tắc "chỉ dùng RPT khi D > 0" được đánh giá mô tả theo họ: hiệu so với `full`.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-15
D < 0 ở toàn bộ 80 run HDD; D > 0 ở hầu hết run benchmark. Đường cong học: RPT có ích khi chỉ có 15 unit selector, gây hại khi có từ 120 unit. Đã mở vòng 7 (EXT38). Chi tiết xem FINDINGS.md.
