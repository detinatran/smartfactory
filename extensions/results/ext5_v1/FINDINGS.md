# EXT5: độ nhạy khi bỏ từng kênh (LOCO), kết quả ngày 2026-11-11

Protocol: `research/selective-rul/EXT5_PROTOCOL.md`, chốt trước khi chạy. Code: `scripts/ext5_loco.py`, `scripts/ext5_summary.py`.
- Quy mô: 240 run của bài ESANN, 10 080 hàng, 0 run lỗi. CPU trên van.
- Đối chiếu: `full` cho 9.37 / 12.49 / 23.29 / 4.47, trùng với EXT3 và EXT4.
- Mọi kết quả là exploratory.

## Cơ chế: LOCO có định vị được kênh lỗi không?
Tỷ lệ cửa sổ bị ảnh hưởng mà kênh có |dc| lớn nhất đúng là một kênh bị lỗi:

| Lỗi | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| bias | 0.80 | 0.86 | 0.46 | 0.95 |
| stuck | 0.59 | 0.37 | 0.08 | 0.33 |
| noise | 0.49 | 0.55 | 0.16 | 0.60 |
| spike | 0.45 | 0.50 | 0.14 | 0.54 |
| gain | 0.33 | 0.23 | 0.12 | 0.50 |
| intermittent | 0.11 | 0.08 | 0.03 | 0.18 |
| ngẫu nhiên | 0.21 | 0.15 | 0.09 | 0.42 |

- Với bias, LOCO định vị được kênh lỗi rất tốt.
- Với stuck, noise và spike, LOCO tốt hơn ngẫu nhiên.
- Với intermittent thì không định vị được, như dự kiến: kênh vốn đã thiếu dữ liệu, che thêm không đổi gì.
- Trên vòng bi, LOCO gần như không định vị được, trừ bias.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| Điểm | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| ensemble_std | 11.16 | 19.51 | 24.08 | 5.52 |
| loco_up (không học) | 10.25 | 18.22 | 23.50 | 5.37 |
| gauss_bias_aware | 10.24 | **12.19** | 23.49 | **3.56** |
| gba_loco | 9.98 | 12.21 | 23.31 | 3.77 |
| full (bài ESANN) | **9.37** | 12.49 | 23.29 | 4.47 |
| full_loco | **9.37** | 12.54 | **23.09** | 4.54 |

## Họ kiểm định chính: `full_loco` trừ `full` (Holm trên 4 kiểm định, 20 cặp)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Hiệu (điểm) | −0.00 | +0.05 | −0.20 | +0.07 |
| Số cặp thắng | 11/20 | 9/20 | 13/20 | 9/20 |
| p Holm | 1.00 | 0.74 | 0.57 | 1.00 |

**Tiêu chí "tốt hơn bài ESANN": KHÔNG ĐẠT.** Thêm LOCO vào bộ phân loại không thay đổi gì: hiệu số đều dưới 0.2 điểm và
không có kiểm định nào có ý nghĩa.

## Phân tích phụ (chưa hiệu chỉnh)
- **`loco_up` so với spread:** tốt hơn có ý nghĩa ở C-MAPSS (−0.91, 19/20 cặp), N-CMAPSS (−1.30, 17/20) và vòng bi (−0.58,
  p = 0.04). Ở pin thì ngang.
  - Đây là điểm không học, không cần nhãn, và tốt hơn spread trên 3 trong 4 họ dữ liệu.
  - Tuy vậy vẫn kém xa các điểm có bias đã học.
- **`gba_loco` so với `gauss_bias_aware`:** tốt hơn ở C-MAPSS (−0.26, 17/20, p = 0.0002), kém hơn nhẹ ở pin (+0.21, p = 0.08).
- **Ngưỡng frozen:**
  - `full_loco` hơi tốt hơn `full` ở C-MAPSS (8.95 so với 9.09) và vòng bi (21.18 so với 21.61).
  - Ở pin, `full_loco` có coverage cao hơn (76.1% so với 71.7%) nhưng risk cũng cao hơn (2.98% so với 2.64%).
  - `loco_up` ở pin: ngưỡng gần như suy biến, coverage 99.5%. Lý do là dc gần 0 trên dữ liệu sạch.
- **Tỷ lệ lỗi lạc quan do fault bị loại (ngưỡng frozen):** `full_loco` cao hơn `full` ở C-MAPSS (38.1% so với 34.8%) và vòng
  bi (29.5% so với 27.4%), thấp hơn ở pin (82.1% so với 88.0%).
- **Dữ liệu sạch:** LOCO không gây hại; hiệu số dưới 0.3 điểm.

## Kết luận
1. **Không thay phương pháp của bài ESANN.**
2. LOCO xác nhận cơ chế của bài:
   - lỗi đẩy dự đoán qua một vài kênh;
   - che kênh đó thì phần lớn dịch chuyển biến mất;
   - một điểm có dấu, không học, đã tốt hơn spread.
3. Bộ phân loại của bài **đã hấp thụ hết thông tin này**, qua thống kê từng kênh và thời gian đã chạy: thêm LOCO không
   cải thiện gì. Như vậy LOCO là bằng chứng cho cơ chế, chứ không phải một phương pháp mới tốt hơn.
4. Sau 5 vòng (EXT2–EXT5), kết luận nhất quán:
   - giới hạn hiện tại nằm ở bộ dự đoán điểm và ở shift giữa calibration và đánh giá (vòng bi), không phải ở cách chấm
     điểm;
   - muốn tiến thêm cần đổi bộ dự đoán, ví dụ train backbone có tăng cường bằng lỗi hoặc che kênh, hoặc dùng dự đoán đã
     sửa nhờ LOCO. Đó là một protocol mới và phải train lại.
