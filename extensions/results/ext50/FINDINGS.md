# EXT50: predictor kiểu RAMP (masked JEPA theo horizon kết hợp nhánh symbolic có cổng), thăm dò

Phạm vi: seed 162–171, 300 run (C-MAPSS 80, pin 100, vòng bi 120). Mọi so sánh đều theo cặp seed với predictor hai giai đoạn mặc định của vòng 8. Risk tính dưới 6 loại lỗi, ở coverage 90%; hiệu tính bằng điểm.

| | C-MAPSS | Pin | Vòng bi |
|---|---|---|---|
| RMSE trên dữ liệu sạch: mặc định → RAMP | 15.55 → 14.38 (RAMP thấp hơn ở 10/10 seed) | 0.094 → 0.086 (10/10) | 0.280 → 0.279 (5/10) |
| Tỷ lệ lạc quan dưới lỗi (nhận hết): mặc định → RAMP | 10.80 → 9.83 | 7.50 → 6.30 | **26.33 → 29.86** |
| RAMP + RPT-D so với mặc định + RPT-D | **−0.82** (10/10, p = 0.002) | **−0.49** (8/10, p = 0.02) | **+3.30** (1/10, p = 0.006) |
| RAMP OSC so với mặc định OSC | −0.59 (9/10) | −0.43 (8/10, p = 0.06) | +3.80 (1/10) |
| Bên trong RAMP: RPT-D − OSC | −0.85 (10/10) | −0.98 (10/10) | −1.00 (9/10) |

## Kết luận
- **C-MAPSS và pin:** predictor RAMP chính xác hơn và ít lạc quan hơn khi có lỗi. Cả hệ thống tốt hơn mặc định.
- **Vòng bi:** predictor RAMP lạc quan hơn nhiều khi có lỗi, nên cả hệ thống kém hơn hẳn (+3.3 điểm). Lý do có thể là tập fit rất nhỏ (12–15 vòng bi mỗi fold) nên pretrain không đủ dữ liệu, hoặc đặc trưng symbolic không hợp với đặc trưng rung động. Cần EXT51 (bản không JEPA) để biết phần nào gây ra.
- **RPT/ELPIS vẫn có ích trên predictor RAMP** ở cả 3 họ (−0.85 đến −1.00 điểm). Cách huấn luyện selector bằng nhiễu được gắn nhãn theo lỗi không phụ thuộc vào loại predictor.
