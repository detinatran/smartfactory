# EXT27: Patch-Transformer với 20 cặp (seed 72–91), 2026-11-15. Post hoc
240 run, 0 lỗi. Có sai lệch vận hành: 10 run được chạy lại, đã ghi trong protocol.

## Risk (%), 6 lỗi
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| `full` | 11.57 | 7.93 | 28.36 | 3.15 |
| `rpt_L` | 11.06 | 7.45 | 26.67 | 2.31 |
| `rpt_auto` | 11.15 | 7.53 | 27.02 | 2.31 |

## B1: `rpt_L` − … (Holm trên 12)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| `full` | −0.51\* (19/20) | −0.49\* (19/20) | −1.69\* (20/20) | −0.84\* (20/20) |
| `gauss_bias_aware` | −1.01\* | −0.15 (p = 0.072) | −1.59\* | −0.65\* |
| `err_reg` | −1.43\* | −0.15 (p = 0.072) | −1.33\* | −0.66\* |

## B2: `rpt_auto` − `full`
−0.42, −0.40, −1.34, −0.84; có ý nghĩa ở cả 4 họ.

## RMSE sạch của predictor
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Patch-Transformer | 19.98 | 0.161 | 0.307 | 0.122 |
| GRU | 15.91 | 0.167 | 0.268 | 0.144 |
| TCN | 15.86 | 0.162 | 0.283 | 0.131 |

Patch-Transformer không mạnh hơn ở mọi họ: kém trên C-MAPSS và vòng bi, tốt hơn trên pin. Lợi ích của RPT vẫn giữ nguyên dù chất
lượng predictor khác nhau.
