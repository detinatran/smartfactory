# EXT19: trả lời review lần 3 (seed 72–81, 3 × 240 run, 0 lỗi; thăm dò)

Protocol: `selective-rul/EXT19_PROTOCOL.md`. Chi tiết: `ext19_main_output.txt`, `ext19_delta_output.txt`, `aurc_output.txt`.
Các score dùng chung khớp EXT12 chính xác.

## A. Lượng dữ liệu, hay nội dung nhiễu? H19a ĐÚNG
- `copy2` (2 bản sao sạch) và `tiny2` (2 bản nhiễu rất nhỏ) có risk ngang OSC, cả khi có lỗi lẫn khi sạch.
- `rpt_L` tốt hơn có ý nghĩa ở cả 4 dataset (Holm trên 8): so với `copy2` là −0.62 / −0.42 / −1.12 / −1.00; so với `tiny2` là
  −0.57 / −0.42 / −1.23 / −1.01.
- Kết luận: cả lợi ích trên dữ liệu sạch cũng đến từ nội dung nhiễu (độ nhạy của predictor với độ lệch đầu vào), không phải
  điều chuẩn hoá do có thêm dữ liệu.
- **Oracle ở cấu hình cuối:**
  - `oracle_f` (train trên 6 lỗi đánh giá) kém `rpt_L` ở mọi dataset: −0.15 / −0.09 / −1.04 / −0.16;
  - `oracle_rpt` (`rpt_L` + 6 lỗi) ≈ `rpt_L` (chênh tối đa 0.01).

## B. Pipeline phát hiện lỗi → tái tạo (mức hệ thống)
- So với sửa đầu vào + độ phân tán: `rpt_L` tốt hơn có ý nghĩa ở C-MAPSS, N-CMAPSS và pin; vòng bi không có ý nghĩa.
- So với sửa đầu vào + OSC: `rpt_L` tốt hơn ở C-MAPSS (−0.33), N-CMAPSS (−2.13) và pin (−0.81), đều có ý nghĩa; nhưng kém ở
  vòng bi (+2.16, p = 0.05 sau Holm).
- Sửa đầu vào + `rpt_L` cho risk vòng bi thấp nhất (18.8%), tức hai cách bổ sung cho nhau.

## C. Độ nhạy theo δ
- Với δ' = 0.5δ và 1.5δ, `rpt_L` tốt hơn `full` có ý nghĩa ở **8/8** phép thử (Holm trên 8):
  - 0.5δ: −0.48 / −0.17 / −1.06 / −0.77;
  - 1.5δ: −0.60 / −0.51 / −1.36 / −0.78.
- So với ensemble, Gaussian có bias và hồi quy sai số: 24/24 có ý nghĩa.

## D. AURC trên [0.5, 0.95]
- `rpt_L` có AURC thấp nhất ở mọi dataset (5.69 / 3.58 / 17.81 / 2.09).
- Tốt hơn có ý nghĩa ở 14/16 phép so sánh. Hai ngoại lệ là pin so với Gaussian có bias và so với hồi quy sai số.
