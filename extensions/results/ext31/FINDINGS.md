# EXT31: pin với cell test xoay vòng 5-fold trên toàn bộ 124 cell (2026-11-15). Post hoc
100 run (seed 122–131, GRU/TCN, 5 fold), train trên GPU, 0 lỗi.

## Risk (%)
| Điểm số | 6 lỗi | Sạch |
|---|---|---|
| `full` | 3.27 | 1.24 |
| `rpt_L` | 2.19 | 0.63 |
| `rpt_auto` | 2.19 | 0.63 |
| `gauss_bias_aware` | 3.12 | — |
| `err_reg` | 3.08 | — |
| `ensemble_std` | 5.70 | — |

## Kiểm định
- **`rpt_L` − `full`:** −1.09 điểm. 20/20 cặp, Wilcoxon p = 1.9e-6; 10/10 seed, sign-flip p = 0.002.
- **`rpt_auto` − `full`:** giống hệt, vì `rpt_auto` chọn `rpt_L` ở 100/100 run.
- **Theo fold:** −0.87, −1.11, −1.40, −1.58, −0.64. Âm ở mọi fold.
- **`rpt_L` so với hai bộ có hướng:** −0.93 so với `gauss_bias_aware`, −0.90 so với `err_reg`, cả hai 20/20 cặp.

**Kết luận:** lợi ích trên pin không phụ thuộc vào tập cell test cố định.
