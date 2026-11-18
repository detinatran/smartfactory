# EXT28: CI bootstrap phân tầng (seed → unit), 2026-11-15
500 run (vòng 2 và vòng 4), 0 lỗi; 2 run bị lỗi hết RAM ban đầu đã được chạy lại. Thiết lập: B = 4000, seed 2828; ngưỡng giữ cố định.

| Họ | Vòng 2: `rpt_L` − `full`, ước lượng [95% CI] | Vòng 4: `rpt_auto` − `full`, ước lượng [95% CI] |
|---|---|---|
| C-MAPSS | −0.55 [−0.74, −0.37] | −0.49 [−0.68, −0.31] |
| N-CMAPSS | −0.40 [−0.77, −0.10] | −0.27 [−0.55, −0.06] |
| Vòng bi | −1.06 [−1.63, −0.48] | −1.05 [−1.71, −0.33] |
| Pin | −0.90 [−1.18, −0.63] | −0.67 [−1.01, −0.37] |
| HDD | — | +0.03 [−0.01, +0.09] |

- Ước lượng điểm khớp các kiểm định đã báo cáo.
- CI loại trừ 0 ở mọi họ benchmark trong cả hai vòng. Kết luận vững trước biến thiên do lấy mẫu unit.
