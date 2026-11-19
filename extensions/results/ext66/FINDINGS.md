# EXT66: ELPIS-J pretrain theo LeWorldModel (end-to-end, MSE + 0.05·SIGReg) so với EMA teacher (V13r); thăm dò; seed 162–171

- Đủ 300/300 run cho mỗi biến thể, không run lỗi, đủ 10/10 seed ở cả 3 họ.
- Kiểm tra chéo: V13r (chạy lại V13 bằng script mới) − V13 của EXT56, risk ELPIS-G: −0.01 (C-MAPSS), −0.01 (pin), +0.08 (vòng bi). Code mới không làm lệch kết quả cũ.

## Chỉ số chính: risk ELPIS-G dưới 6 lỗi, coverage 90% matched, V20 − V13r (seed level, Wilcoxon)
| Họ | V20 | V13r | Hiệu | Seed V20 tốt hơn | p |
|---|---|---|---|---|---|
| C-MAPSS | 6.54 | 7.44 | **−0.90** | 10/10 | 0.002 |
| Pin | 4.42 | 1.92 | **+2.49** | 0/10 | 0.002 |
| Vòng bi | 25.78 | 24.92 | **+0.86** | 3/10 | 0.23 |

## Phụ
| V20 − V13r | C-MAPSS | Pin | Vòng bi |
|---|---|---|---|
| Risk OSC | −1.09 | +2.02 | +0.72 |
| Tỷ lệ lạc quan khi nhận hết, dưới lỗi | −1.07 | +4.17 | +1.03 |
| Lỗi bi quan (ELPIS-G) | +1.30 | −0.11 | +0.03 |
| RMSE sạch | +0.02 (hòa) | +0.006 (kém, 9/10) | −0.004 (tốt hơn, 8/10) |
| Effective rank embedding (V20 / V13r) | 31.2 / 11.2 | 20.4 / 8.9 | 20.4 / 15.6 |
| Độ lệch chuẩn từng chiều (V20 / V13r) | 0.59 / 3.31 | 0.59 / 2.19 | 0.52 / 1.74 |
| Thống kê SIGReg (V20 / V13r) | 116 / 382 | 143 / 487 | 236 / 491 |

Bên trong V20, ELPIS-G − OSC: −0.79 (10/10), −0.60 (10/10), −0.56 (8/10, p = 0.008).

## Kết luận (theo tiêu chí đặt trước)
- **Không đạt "hứa hẹn"**: risk ELPIS-G chỉ giảm trên C-MAPSS; pin (+2.49) và vòng bi (+0.86) vượt ngưỡng +0.3. V13 giữ nguyên là ELPIS-J của bài; không kiểm lại trên seed mới.
- **EMA teacher của V13 làm sụp một phần biểu diễn**: effective rank chỉ 9–16, so với 20–31 khi có SIGReg; phân phối embedding xa Gaussian đẳng hướng hơn nhiều. SIGReg chữa được điều này như LeJEPA/LeWM dự đoán.
- **Nhưng biểu diễn ít sụp hơn không làm risk thấp hơn một cách nhất quán**: trên pin và vòng bi, predictor V20 lạc quan hơn khi có lỗi. Giả thuyết (chưa kiểm): biểu diễn đa dạng hơn phản ứng mạnh hơn với lỗi cảm biến, còn head RUL học trên dữ liệu sạch với encoder đóng băng ngoại suy sai.
- **ELPIS-G vẫn tốt hơn OSC trên predictor V20 ở cả 3 họ**: thêm bằng chứng rằng lợi ích của selector không phụ thuộc predictor.
- Chỉ thử λ = 0.05 (không dò), 10 seed phát triển; kết quả là thăm dò.
