# EXT38: vòng xác nhận 7, `rpt_auto_d` (dùng RPT khi và chỉ khi D > 0), 2026-11-15: XÁC NHẬN
260 ensemble mới:
- seed 152–161, train trên GPU;
- 4 họ benchmark và hdd4 (4 model hạng 13–16, chưa từng dùng; 349 ổ, 109 ổ test).

0 lỗi.

## Chính: `rpt_auto_d` − `full` (Holm trên 5, α_7 = 0.000390625)
| Họ | Hiệu (điểm) | Cặp tốt hơn | p_Holm | Ý nghĩa ở α_7 |
|---|---|---|---|---|
| C-MAPSS | −0.64 | 20/20 | 1e-5 | có |
| N-CMAPSS | −0.50 | 20/20 | 1e-5 | có |
| Vòng bi | −0.24 | 12/20 | 0.16 | không |
| Pin | −0.51 | 17/20 | 0.0015 | không |
| hdd4 | +0.04 | 3 tốt hơn, 5 kém hơn | 0.48 | không |

- **Kết luận: XÁC NHẬN.**
  - Tỷ lệ bi quan tăng tối đa 0.48.
  - Không kém trên hdd4: CI 90% là [−0.03, +0.11], cận trên < 0.3.
- Tổng α đã tiêu = 0.049609375.

## Mô tả
- **D trung bình (điểm):** C-MAPSS +7.3, N-CMAPSS +13.6, vòng bi −1.7, pin +19.0, hdd4 −0.6.
- **Tỷ lệ chuyển của `rpt_auto_d`:** C-MAPSS, N-CMAPSS và pin 100%; vòng bi 42%; hdd4 40%.
- **`rpt_L` − `full`:** −0.64, −0.50, −0.81, −0.51; hdd4 −0.11. Trên hdd4 lần này RPT không gây hại.
- **`rpt_auto` (R0) − `full`:** −0.36, −0.47, −0.59, −0.51; hdd4 −0.06.
- **Điểm yếu còn lại:** `rpt_auto_d` mất phần lớn lợi ích trên vòng bi, vì D ở đây gần 0.
