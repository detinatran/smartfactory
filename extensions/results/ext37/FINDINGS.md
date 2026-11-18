# EXT37: chẩn đoán D và đường cong học theo số unit selector (2026-11-15). Thăm dò
320 run, 0 lỗi.

## Chẩn đoán D
D = tỷ lệ lạc quan trên bản perturbed − tỷ lệ lạc quan khi sạch, tính trên calibration. Đơn vị: điểm.

| Họ | D trung bình | Tỷ lệ run D > 0 | `rpt_L` − `full` | Quy tắc "dùng RPT khi D > 0" |
|---|---|---|---|---|
| C-MAPSS | +7.2 | 100% | −0.74 | −0.74 |
| N-CMAPSS | +16.8 | 95% | −0.35 | −0.32 |
| Vòng bi | −1.4 | 42% | −1.02 | −0.43 |
| Pin | +18.7 | 100% | −0.72 | −0.72 |
| HDD 92–101 | −2.3 | 0% | +0.29 | 0.00 |
| HDD 112–121 | −2.4 | 0% | +0.46 | 0.00 |
| hdd2 | −2.0 | 0% | +0.59 | 0.00 |
| hdd3 | −1.9 | 0% | +0.20 | 0.00 |

- Spearman giữa D và hiệu, trên toàn bộ run: ρ = −0.27, p = 1e-6.
- Trong từng họ, tương quan yếu hoặc không nhất quán (N-CMAPSS dương). Như vậy D chủ yếu phân biệt **giữa** các họ, không phân biệt các run trong cùng một họ.

## Đường cong học trên HDD
`rpt_L` − `full` (điểm), theo số unit selector:

| Tập | 15 | 30 | 60 | 120 | tất cả |
|---|---|---|---|---|---|
| HDD 92–101 | −0.58 (19/20) | −0.09 | +0.07 | +0.25 | +0.29 |
| HDD 112–121 | −0.99 (20/20) | −0.28 (17/20) | +0.10 | +0.37 | +0.46 |
| hdd2 | −1.02 (20/20) | −0.23 (15/20) | +0.13 | +0.41 | +0.59 |
| hdd3 | −0.61 (20/20) | −0.19 (16/20) | −0.01 | (96 unit) +0.20 | — |

**Kết luận:**
- RPT có ích khi dữ liệu selector ít; các họ benchmark chỉ có 5–39 unit selector.
- RPT gây hại khi dữ liệu nhiều và perturbation không tạo ra lỗi lạc quan (D < 0).
- Đã mở vòng 7 (EXT38) cho quy tắc D > 0.
