# EXT59: vòng xác nhận 10 (vòng cuối của ngân sách α). XÁC NHẬN

Câu hỏi: ELPIS-G có tốt hơn OSC khi cả hai dùng chung predictor RAMP-V13 (pretrain JEPA, đóng băng encoder, hiệu chỉnh độ lệch) không?

- Dữ liệu: 30 seed mới (222–251), 900 ensemble, 900 run eval RAMP. Đủ 30/30 seed ở mọi họ.
- Kiểm định: Wilcoxon ở mức seed, Holm trên 3 họ, α_10 = 4.88e-5.
- Đơn vị: risk ở coverage 90% dưới 6 lỗi, tính bằng %; hiệu tính bằng điểm phần trăm.

| Họ | OSC | ELPIS-G | Hiệu | Seed ELPIS-G tốt hơn | p_Holm | Thay đổi tỷ lệ bi quan |
|---|---|---|---|---|---|---|
| C-MAPSS | 9.16 | 8.35 | −0.81 | 30/30 | 5.6e-9 | +0.47 |
| Pin | 3.04 | 1.90 | −1.14 | 30/30 | 5.6e-9 | +0.36 |
| Vòng bi | 25.52 | 24.49 | −1.03 | 29/30 | 5.6e-9 | +0.49 |

**CONFIRMED.** ELPIS-G chuyển sang RPT ở 100% deployment C-MAPSS và pin, và 70% deployment vòng bi.

## Ý nghĩa
Cách huấn luyện selector của ELPIS bổ sung cho một predictor mạnh hơn, đã được pretrain theo hướng chống lỗi. Trên vòng bi, lợi ích còn lớn hơn khi dùng predictor mặc định (−1.03 so với −0.62).

## Còn chờ
So sánh phụ: hệ thống RAMP-V13 so với hệ thống mặc định trên cùng seed. Bước eval predictor mặc định đang chạy.

## Kết quả phụ (mô tả): hệ thống RAMP-V13 so với hệ thống mặc định, cùng seed 222–251

Cả hai hệ thống đều dùng ELPIS-G; các con số là risk dưới lỗi.

| Họ | Hiệu risk | Seed RAMP tốt hơn | Lỗi bi quan | RMSE (RAMP so với mặc định) |
|---|---|---|---|---|
| C-MAPSS | −0.60 | 24/30 | −0.37 | 14.44 so với 15.85 |
| Pin | −0.45 | 21/30 | +0.16 | 0.085 so với 0.092 |
| Vòng bi | +1.99 | 15/30 (p = 0.15) | −5.89 | 0.265 so với 0.275 |

Kết quả khớp với các thí nghiệm phát triển trước đó.
