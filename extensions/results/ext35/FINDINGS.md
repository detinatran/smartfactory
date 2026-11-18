# EXT35: vòng xác nhận 6, quy tắc chuyển margin 0.01 (2026-11-15): KHÔNG XÁC NHẬN
260 ensemble mới:
- seed 142–151, train trên GPU;
- 4 họ benchmark và hdd3 (4 model ổ cứng hạng 9–12, chưa từng dùng).

0 lỗi.

## Chính: `rpt_auto_m` − `full` (Holm trên 5, α_6 = 0.00078125)
| Họ | Hiệu | p_Holm |
|---|---|---|
| C-MAPSS | −0.11 | 0.065 |
| N-CMAPSS | 0.00 (không chuyển lần nào) | — |
| Vòng bi | −0.47 | 0.021 |
| Pin | −0.12 | 0.28 |
| hdd3 | 0.00 (không chuyển lần nào) | — |

- **Không họ nào đạt α_6, nên KHÔNG XÁC NHẬN.**
- Không kém trên hdd3: hiển nhiên, vì trùng hoàn toàn với `full`.

## Mô tả
- **Tỷ lệ chuyển của `rpt_auto_m`:** C-MAPSS 15%, N-CMAPSS 0%, vòng bi 42%, pin 40%, hdd3 0%.
- **Tỷ lệ chuyển của R0:** 64%, 70%, 72%, 95%, 35%.
- **`rpt_L` − `full`:** C-MAPSS −0.55, N-CMAPSS −0.30, vòng bi −1.15, pin −0.48, hdd3 **+0.20**. Kết quả âm trên HDD lặp lại lần thứ ba.
- **`rpt_auto` (R0) − `full`:** −0.34, −0.19, −0.80, −0.46, hdd3 +0.06.

## Kết luận
- Không có quy tắc chuyển dựa trên calibration nào vừa giữ được lợi ích vừa loại bỏ được thiệt hại. Đây là đánh đổi thật.
- R0 vẫn là khuyến nghị, kèm rủi ro nhỏ còn lại.
- Tổng α đã tiêu = 0.04921875.
