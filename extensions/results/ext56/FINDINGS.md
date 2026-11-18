# EXT52–EXT58: sửa ELPIS-J trên vòng bi (phát triển, thăm dò; seed 162–171)

## Các biến thể
Risk của RPT-D (lỗi lạc quan) dưới 6 lỗi, coverage 90%, so với predictor mặc định (23.76%).

| Biến thể | Hiệu (điểm) | Ghi chú |
|---|---|---|
| ELPIS-J | +3.30 | |
| V1: pretrain thêm dữ liệu | +3.62 | |
| V2: tắt symbolic | +3.25 | |
| V5: JEPA khử nhiễu | +3.70 | |
| V6: fine-tune trên dữ liệu bị hỏng | +3.61 | |
| V4: đóng băng + tắt symbolic | +1.93 | |
| V3: đóng băng encoder | +1.35 | |
| V14: hiệu chỉnh độ lệch trung bình | +1.79, không có ý nghĩa | |
| **V13: đóng băng + hiệu chỉnh độ lệch trung bình** | **+1.08, không có ý nghĩa (p = 0.62)** | lỗi bi quan 30.6%, mặc định 35.9% |
| V15: đóng băng + offset cuối đời | −14.94 | chỉ là offset bảo thủ: lỗi bi quan 55.9%, RMSE 0.321 |

## Chẩn đoán
Trên vòng bi, predictor ELPIS-J bị lệch về phía lạc quan ngay cả trên dữ liệu sạch (29.8% so với 26.5%). Lỗi lạc quan tập trung ở cuối đời vòng bi.

## So sánh công bằng ở cùng mức offset (EXT58)
Predictor mặc định được áp đúng offset cuối đời như V15.

| | Mặc định + offset | V15 (ELPIS-J + offset) | Hiệu V15 − mặc định |
|---|---|---|---|
| Lỗi lạc quan của RPT-D | 8.62% | 8.83% | +0.21 (n.s.) |
| Lỗi bi quan của RPT-D | 59.3% | 55.9% | −3.43 (8/10 seed, p = 0.02) |
| RMSE | 0.351 | 0.321 | −0.029 (9/10 seed) |

Ở cùng mức lỗi lạc quan, ELPIS-J mắc ít lỗi bi quan hơn và chính xác hơn. Không có offset thì V13 so với mặc định: lỗi lạc quan +1.1 (không có ý nghĩa), lỗi bi quan −5.3.

## V13 trên C-MAPSS và pin (risk RPT-D so với mặc định)
| Họ | V13 | ELPIS-J gốc |
|---|---|---|
| C-MAPSS | −0.42 (8/10 seed) | −0.82 |
| Pin | −0.42 (7/10 seed) | −0.49 |

## Kết luận
- Đóng băng encoder cộng hiệu chỉnh độ lệch (V13) xoá được phần thua có ý nghĩa trên vòng bi, và vẫn giữ lợi ích trên C-MAPSS và pin.
- Khi so ở cùng mức lỗi lạc quan (cùng offset), ELPIS-J tốt hơn predictor mặc định trên vòng bi: ít lỗi bi quan hơn, RMSE thấp hơn.
- Mọi kết quả trên đều là phát triển trên seed đã dùng. Muốn khẳng định cần một vòng xác nhận trên seed mới.

## Đánh giá lại cổng theo cỡ tập selector (post hoc), ngày 2026-11-17
Quy tắc: dùng ELPIS nếu tập selector có dưới 60 unit, ngược lại dùng OSC. Quy tắc này cho kết quả giống ELPIS-G ở mọi họ, trừ vòng bi. Trên vòng bi, nó tốt hơn ELPIS-G 0.80 điểm ở cả vòng 8 (27/30 seed) và vòng 9 (28/30 seed). Ngưỡng 60 được chọn từ EXT37 và EXT43 (seed 92–171), nên các seed của vòng 9 độc lập với việc chọn ngưỡng. Kết quả xem `size_gate_rescore.txt`.
