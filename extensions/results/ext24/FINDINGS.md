# EXT24: triển khai với ngưỡng cố định trên calibration (2026-11-15). Post hoc, thăm dò

Protocol `research/selective-rul/EXT24_PROTOCOL.md`. 260 run, seed 92–101, 0 lỗi. Kiểm tra nhất quán với EXT22: 780 ô,
chênh tối đa 0.0000 điểm.

## Mục tiêu risk (r* = 0.5 × risk khi nhận hết trên calibration sạch; r* trung bình: C-MAPSS 5.02%, N-CMAPSS 7.17%, vòng bi 12.35%, pin 0.84%, HDD 12.32%)
1. **Plug-in không đủ cho một mục tiêu risk.**
   - `rpt_auto` vi phạm ở 41–60% deployment trên 4 họ benchmark, với mọi tập calibration và mọi π, kể cả π = 0 (C0: 54/55/43/55%).
   - Nguyên nhân: với 3–51 unit calibration, ngưỡng plug-in chỉ là ước lượng điểm, nên sai khoảng một nửa số lần.
     Lỗi cảm biến không phải nguyên nhân chính.
   - Các điểm số có ít vi phạm hơn đạt được nhờ gần như không nhận gì. Ví dụ `ensemble_std` với CP chỉ có coverage 0.6–23%.
   - Kết quả này khớp với STEP3 và ltt_v1/v2. Đúng như diễn giải đã chốt trước, bài sẽ nói thẳng điều này.
2. **Risk thực tế ổn định theo tỷ lệ lỗi** với selector một phía, nhưng tăng với dispersion và Gaussian. Số liệu C-MAPSS, C0, π = 0 / 0.1 / 0.3 / 0.5:

   | Điểm số | Risk thực tế (%) |
   |---|---|
   | `rpt_auto` | 5.36 / 5.32 / 5.24 / 5.16 |
   | `full` | 5.19 / 5.21 / 5.24 / 5.28 |
   | `ensemble_std` | 5.95 / 6.14 / 6.52 / 6.91 |
   | `gauss_bias_aware` | 5.05 / 5.23 / 5.57 / 5.90 |
   | `err_reg` | 5.97 / 6.13 / 6.45 / 6.76 |

   Pin, `ensemble_std`: 2.55 → 3.60, trong khi `rpt_auto` 1.26 → 1.32.
3. **D1, `rpt_auto` − `full`** (CP, π = 0.3, Holm trên 10):
   - Coverage: C-MAPSS +7.7 (20/20, p_holm < 1e-4); vòng bi +8.9 (17/20, 0.003); pin +31.7 (17/20, 0.003);
     N-CMAPSS +1.3 (n.s.); HDD +0.4 (n.s.).
   - Phần vượt mục tiêu: C-MAPSS +0.55 điểm (p_holm = 0.004); các họ khác n.s.

   Ở cùng mục tiêu, RPT-auto giữ được nhiều coverage hơn, đổi lại phần vượt mục tiêu trên C-MAPSS lớn hơn một chút.
4. **D2, tập calibration cho `rpt_auto`** (π = 0.3, Holm trên 20):
   - CP − C0: phần vượt trên C-MAPSS −0.22 (p_holm = 0.016), coverage −2.7 (0.04); HDD coverage +5.7 (p_holm < 1e-4).
     Các họ khác n.s.
   - CS − C0: **không an toàn trên HDD**. Phần vượt +1.55 điểm và vi phạm ở 100% deployment. Các shift đã thấy (dropout, drift)
     làm lệch calibration của bộ đếm SMART.

## Mục tiêu coverage 90%
- Ngưỡng C0 của `rpt_auto` giữ coverage test gần 90% trên C-MAPSS (89.1–90.2%), vòng bi (89.6%) và HDD (87–90%) với mọi π.
- N-CMAPSS: 86.5–88.5%. Pin thiếu: 79–84%.
- CP đẩy coverage lên trên mục tiêu (pin 93–97%).

## Kết luận cho bài
- Kết quả âm: plug-in không giữ được mục tiêu risk, vì vi phạm khoảng 50% ngay cả khi dữ liệu sạch.
- Kết quả dương phụ: risk thực tế của selector một phía không trôi theo tỷ lệ lỗi, và RPT-auto giữ coverage cao hơn OSC ở cùng mục tiêu.
- Muốn có đảm bảo thì cần nhiều unit có nhãn hơn (ltt_v2).

# EXT24b: offset (dự đoán bảo thủ) so với từ chối + offset (2026-11-15). Thăm dò
260 run, 0 lỗi. Chính sách `joint90[m]` = ngưỡng 90% của m, sau đó chọn b nhỏ nhất để risk trên P ≤ r*.

## Calibration sạch C0, π = 0.3 (mô tả)
Mỗi ô ghi: risk % | tỷ số RMSE | b/δ.

| | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD |
|---|---|---|---|---|---|
| `offset_only` | 5.51 \| 1.16 \| 0.47 | 8.89 \| 1.15 \| 0.65 | 12.86 \| 1.17 \| 0.91 | 1.38 \| 1.28 \| 0.39 | 11.90 \| 1.39 \| 1.75 |
| `joint90[rpt_auto]` | 5.29 \| 1.03 \| 0.27 | 6.88 \| 0.83 \| 0.07 | 13.17 \| 1.12 \| 0.66 | 0.86 \| 1.01 \| 0.00 | 10.96 \| 1.18 \| 1.10 |

Với C0, chính sách joint **lấn át** offset trên 4 họ: risk thấp hơn và RMSE thấp hơn. Trên vòng bi, joint có risk cao hơn 0.3
nhưng RMSE thấp hơn. Kết quả này lặp lại STEP3.

## D4 đã chốt trước (CP, π = 0.3, Holm trên 10)
Hiệu = `joint90[rpt_auto]` − `offset_only`.

| Họ | Risk (điểm) | Tỷ số RMSE |
|---|---|---|
| C-MAPSS | +1.43 (p_holm < 1e-4) | −0.32 (20/20) |
| N-CMAPSS | +3.19 (< 1e-4) | −0.76 (20/20) |
| Vòng bi | +0.53 (n.s.) | −0.05 (15/20, p_holm = 0.003) |
| Pin | +1.10 (p_holm = 0.005) | −2.30 (20/20) |
| HDD | −0.08 (n.s.) | −0.20 (20/20) |

Với CP, `offset_only` phải dịch b rất lớn (0.85–2.49δ trên 4 họ), nên đạt risk thấp hơn. Cái giá là RMSE tăng 16–232% so với
predictor không dịch, trong khi joint chỉ thay đổi −6% đến +17%. **Đây là đánh đổi chứ không phải lấn át:** từ chối giữ độ chính xác
tốt hơn, còn offset đạt risk thấp hơn khi calibration bi quan.

## Mô tả: `joint90[rpt_auto]` − `joint90[full]` (CP, π = 0.3)
- risk cao hơn 0.65–0.87;
- RMSE thấp hơn (−0.03 đến −0.97);
- coverage cao hơn (+0.9 đến +4.6).

Hiệu này đến từ b được chọn nhỏ hơn. Đây là đánh đổi, không phải thắng rõ.
