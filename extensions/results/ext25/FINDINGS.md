# EXT25: risk trên window bị lỗi thay đổi, và baseline mới (2026-11-15). Post hoc, thăm dò
Protocol: `research/selective-rul/EXT25_PROTOCOL.md`. 260 run (seed 92–101, 5 họ), 0 lỗi. Nhất quán với EXT22: 4680 hàng, chênh 0.0000.

## Tác động của lỗi trên các window bị ảnh hưởng (A)
| Họ | A chiếm (%) | Tỷ lệ lạc quan của predictor, sạch → lỗi (%) | Hướng tác động |
|---|---|---|---|
| C-MAPSS | 47.1 | 17.6 → 20.2 | lạc quan hơn |
| N-CMAPSS | 46.2 | 18.8 → 28.0 | lạc quan hơn |
| Vòng bi | 43.5 | 19.6 → 18.9 | ít lạc quan hơn |
| Pin | 33.5 | 0.25 → 5.5 | lạc quan hơn |
| HDD | 22.9 | 78.9 → 72.7 | ít lạc quan hơn |

Tức là lỗi chỉ làm predictor lạc quan hơn ở 3 họ: C-MAPSS, N-CMAPSS và pin.

## F1 (Holm trên 20)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD |
|---|---|---|---|---|---|
| Δ`excess_A`, `rpt_L` − `full` | −1.22\* (18/20) | −0.57\* (15/20) | +0.06 | −1.90\* (20/20) | +1.15\* (kém) |
| Δ`excess_A`, `rpt_auto` − `full` | −0.75\* (16/20) | −0.40\* (12/20, p = 0.045) | +0.03 | −1.84\* (18/20) | +0.08 (n.s.) |
| `risk_A`, `rpt_auto` − `full` | −1.05\* | −0.85\* | −0.71 (p = 0.064) | −1.88\* | +0.08 |

**Kết luận theo diễn giải đã chốt:** phần giảm risk dư do lỗi có ý nghĩa cả **ngoài C-MAPSS** (N-CMAPSS, pin). Nhận định cũ
"phần riêng cho lỗi chỉ có ý nghĩa trên C-MAPSS" là hệ quả của việc pha loãng bởi các window không bị đụng tới. Bài đã được sửa
theo kết luận này.

## F2 (`risk_A`, Holm trên 15)
- `rpt_auto` tốt hơn `ensemble_std`, `gauss_bias_aware` và `err_reg` ở 4 họ benchmark.
- Trên HDD, `rpt_auto` kém `err_reg` (+1.85) và `gauss_bias_aware` (+1.30), cả hai có ý nghĩa.

## F3: baseline mới (`risk_matched`, Holm trên 15)
`rpt_auto` tốt hơn có ý nghĩa ở cả 5 họ:
- `eq90` (quantile regression của E): −0.35 đến −2.44;
- `pca_spe`: −1.78 đến −7.23;
- `pca_t2`: −1.79 đến −7.18.
