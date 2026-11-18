# EXT32: N-CMAPSS với engine test xoay vòng 5-fold trên toàn bộ 99 engine (2026-11-15). Post hoc
100 run (seed 132–141, GRU/TCN, 5 fold), train trên GPU, 0 lỗi.

## Risk (%), 6 lỗi
| Điểm số | Risk |
|---|---|
| `full` | 8.77 |
| `rpt_L` | 8.53 |
| `rpt_auto` | 8.59 |
| `gauss_bias_aware` | 8.64 |
| `err_reg` | 8.62 |
| `ensemble_std` | 15.90 |

## Kiểm định
- **`rpt_L` − `full`:** −0.25 điểm. 20/20 cặp; 10/10 seed, sign-flip p = 0.002.
- **Theo fold:** −0.23 đến −0.31. Âm ở mọi fold.
- **`rpt_auto` − `full`:** −0.19 (20/20). `rpt_auto` chọn `rpt_L` ở 75/100 run.
- **`rpt_L` so với hai bộ có hướng:** −0.11 so với `gauss_bias_aware` (16/20, p = 0.001); −0.09 so với `err_reg` (16/20, p = 0.006).

**Kết luận:** lợi ích nhỏ hơn so với khi dùng test chính thức (−0.40), nhưng nhất quán và tổng quát hoá sang engine mới.
