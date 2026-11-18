# EXT44: chẩn đoán D khi predictor đã được làm robust (thăm dò; robust72, seed 72–81, 240 run)

| Họ | D (điểm) | Tỷ lệ D > 0 | RPT − OSC | RPT-D − OSC | Độ chính xác quyết định của D |
|---|---|---|---|---|---|
| C-MAPSS | +5.9 | 99% | −0.45 | −0.45 | 76% |
| N-CMAPSS | +8.1 | 95% | −0.27 | −0.27 | 95% |
| Vòng bi | −0.36 | 46% | −1.30 | −0.55 | 43% |
| Pin | +10.9 | 100% | −0.89 | −0.89 | 100% |

- p = 0.002 theo seed ở mọi hiệu RPT − OSC và RPT-D − OSC.
- So với predictor thường (vòng 8): C-MAPSS +7.3, N-CMAPSS +15.5, pin +17.3.

## Kết luận
- Làm robust predictor làm D giảm, nhưng không đổi dấu ở bất kỳ họ benchmark nào. RPT vẫn có ích, RPT-D trùng với RPT.
- Không tạo được trường hợp âm mới trên benchmark. Hiệu lực của D như một cổng vẫn chỉ được kiểm chứng trên ổ cứng.
- Vòng bi giống hệt các vòng trước: D gần 0, chọn đúng chỉ 43%, RPT-D giữ được khoảng 40% lợi ích của RPT.
