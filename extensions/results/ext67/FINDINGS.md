# EXT67: cổng calibration chọn giữa ELPIS-J V13 (EMA) và V20 (SIGReg), kiểm lại trên seed 222–251; thăm dò, không tiêu α

- V20 chạy mới trên 900 run (C-MAPSS, pin, vòng bi; GRU/TCN; seed 222–251), không run lỗi. V13 lấy từ vòng 10 (EXT59). Đủ 30/30 seed ở cả 3 họ.
- Quy tắc cổng (giống ELPIS-JG, không chỉnh): dùng V20 nếu risk calibration của selector ELPIS-G chọn trên V20 thấp hơn trên V13. Chỉ dùng unit calibration.
- Seed 222–251 không được dùng để thiết kế V20 hay cổng (giả thuyết đến từ seed 162–171, EXT66).

## Chỉ số chính: risk ELPIS-G dưới 6 lỗi, coverage 90% matched, cổng − V13 (seed level)
| Họ | Cổng | V13 | Hiệu | Cận trên CI 90% | Seed tốt hơn | p | Kết luận |
|---|---|---|---|---|---|---|---|
| C-MAPSS | 7.32 | 8.35 | **−1.03** | −0.89 | 30/30 | < 1e-8 | tốt hơn |
| Pin | 1.93 | 1.90 | +0.03 | +0.08 | 14/30 | 0.41 | không kém |
| Vòng bi | 23.29 | 24.49 | **−1.20** | −0.66 | 21/30 | 0.002 | tốt hơn |

**Theo tiêu chí đặt trước: cổng CẢI THIỆN ELPIS-J** (tốt hơn ở 2 họ, không kém ở cả 3).

## Phụ
| | C-MAPSS | Pin | Vòng bi |
|---|---|---|---|
| V20 thay hẳn V13 (không cổng) | −1.21 | +1.89 | −0.37 (p = 0.45) |
| Lỗi bi quan, cổng − V13 | **+1.24** | +0.20 | **+1.67** |
| RMSE sạch, cổng − V13 | −0.06 (hòa) | hòa | −0.004 (21/30, p = 0.001) |
| ELPIS-G − OSC trên V20 | −0.71 (30/30) | −0.65 (30/30) | −0.75 (27/30) |
| Tỷ lệ deployment chọn V20 | 65% | 17% | 46% |

## Diễn giải
- Cổng giữ lợi ích của V20 trên C-MAPSS và vòng bi, và gần như từ chối V20 trên pin, nơi V20 thua +1.89.
- Cái giá: lỗi bi quan tăng 1.2–1.7 điểm trên C-MAPSS và vòng bi. RMSE không kém đi (vòng bi còn tốt hơn), nên đây không chỉ là dịch dự đoán về phía thận trọng, nhưng phải báo cáo cùng nhau.
- ELPIS-G vẫn tốt hơn OSC trên V20 ở cả 3 họ.
- Thăm dò (α đã hết); seed 222–251 đã dùng cho vòng 10 nhưng độc lập với việc thiết kế cổng.
