# EXT13: sáu baseline bổ sung (seed 62–71, 240 run, 0 lỗi)

Protocol: `selective-rul/EXT13_PROTOCOL.md`. Chi tiết: `summary_output.txt`, `familyEXT13.csv`.

OURS (`full_recon_rpt`) tốt hơn có ý nghĩa so với **cả 6** baseline, ở **24/24** cặp (baseline × dataset):
- Holm trên 24 phép thử, p ≤ 2e-4;
- thắng 18–20 trên 20 cặp seed;
- mức giảm: C-MAPSS −2.0 đến −3.8 điểm, N-CMAPSS −6.1 đến −7.9, vòng bi −1.4 đến −3.9, pin −1.2 đến −2.6.

| Baseline | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Laplace lớp cuối | 11.18 | 20.13 | 26.64 | 5.52 |
| Mahalanobis | 10.98 | 18.36 | 27.56 | 4.67 |
| kNN trên đặc trưng | 11.42 | 20.04 | 27.58 | 5.57 |
| Isolation Forest | 12.33 | 19.57 | 27.04 | 5.56 |
| Residual dư thừa giải tích | 10.52 | 18.94 | 29.18 | 5.45 |
| TTA | 10.70 | 20.09 | 27.72 | 6.06 |
| OURS | 8.52 | 12.28 | 25.26 | 3.47 |

Số trong bảng là risk (%) @90% matched coverage, trung bình 6 lỗi chưa thấy.

**Kết luận:** các bộ phát hiện OOD và bất thường (Mahalanobis, kNN, Isolation Forest, residual) đều không đủ. Phát hiện "đầu
vào lạ" không thay được việc ước lượng trực tiếp sai số lạc quan. Điều này nhất quán với kết quả của bộ phát hiện lỗi ở
EXT11.
