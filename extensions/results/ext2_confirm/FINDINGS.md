# EXT2 vòng xác nhận — kết quả (2026-11-10)

Protocol: `research/selective-rul/EXT2_CONFIRM_PROTOCOL.md` (phần A), chốt trước khi chạy.
- Quy mô: seed 47–51; C-MAPSS FD001–4 có 20 run, N-CMAPSS có 5 run.
- Backbone: GRU và FiLM-GRU. Tổng 2 800 hàng, 0 run lỗi.
- Phân tích: `scripts/ext2_confirm_tests.py` (`confirm_tests.csv`, `confirm_secondary.csv`, `confirm_h1.csv`) và
  `scripts/ext2_summary.py`.

## Họ kiểm định xác nhận (Holm trên 4 kiểm định)
Chỉ số: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy. Hiệu là FiLM trừ GRU, đơn vị điểm %.

| Kiểm định | Dữ liệu | n | FiLM | GRU | Hiệu | FiLM tốt hơn | p | p Holm |
|---|---|---|---|---|---|---|---|---|
| C1 gauss_bias_aware | C-MAPSS | 20 | 11.13 | 10.33 | +0.80 | 10/20 | 0.35 | 0.46 |
| C1 gauss_bias_aware | N-CMAPSS | 5 | 9.67 | 11.95 | −2.28 | 5/5 | 0.0625 | 0.25 |
| C2 full | C-MAPSS | 20 | 10.54 | 9.36 | +1.18 | 9/20 | 0.23 | 0.46 |
| C2 full | N-CMAPSS | 5 | 10.08 | 12.26 | −2.19 | 5/5 | 0.0625 | 0.25 |

- **C-MAPSS: KHÔNG xác nhận.**
  - Ở pilot FiLM tốt hơn 0.3–0.4 điểm; ở seed mới FiLM lại kém hơn 0.8–1.2 điểm (không có ý nghĩa thống kê).
  - RMSE sạch của FiLM cũng kém hơn: 0.142 so với 0.131 cap.
- **N-CMAPSS: cùng chiều 5/5 seed**, giảm khoảng 2.2 điểm.
  - Như đã ghi trước trong protocol, 5 cặp không thể đạt p < 0.05.
  - Phân tích phụ gộp pilot + xác nhận (10 seed, một nửa đã thấy): FiLM tốt hơn ở **10/10 seed** với cả gauss_bias_aware,
    full và conformal (−2.1 đến −2.3 điểm; p = 0.002, chưa hiệu chỉnh).

## Tái lập H1 (mô tả)
- **shift/spread của FiLM lớn hơn:** C-MAPSS 2.51 so với 1.83; N-CMAPSS 3.70 so với 2.10. **Tái lập trên cả hai.**
- **Spread của FiLM loại ít lỗi lạc quan do fault hơn:** N-CMAPSS 13.1% so với 15.7% (tái lập). C-MAPSS **ngược chiều**
  (25.1% so với 22.2%).
- Độ phủ của biên conformal: 86.7–89.0% dưới lỗi, so với 87.4–89.2% khi sạch. Chênh lệch dưới 1 điểm, khớp pilot.

## Kết luận
1. "FiLM + điểm có bias tốt nhất" **không tổng quát**: ổn định trên N-CMAPSS (10/10 seed) nhưng không trên C-MAPSS.
   Không được viết thành kết quả chung. Nếu đưa vào bài, chỉ có thể nói lợi ích phụ thuộc dữ liệu, kèm số liệu trên.
2. Phần vững: ensemble ngầm (FiLM) có các member dịch chuyển chung nhiều hơn (shift/spread lớn hơn khoảng 1.4–1.8 lần,
   trên 10 seed và 2 dữ liệu). Đây là bằng chứng thực nghiệm cho cơ chế common-shift.
3. Điểm có bias luôn tốt hơn họ variance, và biên conformal giữ độ phủ dưới lỗi, trên mọi seed và backbone đã chạy.
