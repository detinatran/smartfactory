# EXT7: bộ chọn lọc ghép theo tác động của lỗi (stacked fault-impact selector). Chốt ngày 2026-11-11, TRƯỚC khi chạy

## Ý tưởng (lắp ghép)
Các vòng trước cho thấy mỗi thành phần mạnh ở một nơi:
- bộ phân loại kèm đặc trưng tái tạo (`full_recon`, EXT6 và EXT6-C) mạnh ở C-MAPSS, đã xác nhận bằng seed mới;
- Gaussian có bias (`gauss_bias_aware`, EXT3–EXT4) mạnh ở N-CMAPSS và pin;
- tác động có dấu `impact` = ŷ − ŷ_rep (EXT6) là điểm không học tốt nhất ở C-MAPSS và N-CMAPSS.

EXT7 ghép chúng lại. Mỗi run được fit một lần, không có bước chọn nào phụ thuộc vào tập đánh giá:
- `gba_recon`: Gaussian có bias, với b(x) và v(x) học trên đặc trưng của bài cộng 3 đặc trưng tái tạo.
- `stack`: hồi quy logistic (chuẩn hoá đầu vào, C = 1) trên [logit(`full_recon`), logit(`gba_recon`), `impact`].
  - Fit trên **unit calibration × {clean, dropout, drift}**, có trọng số theo unit.
  - Unit calibration tách biệt với unit selector (nơi fit các mô hình HGB) và với unit đánh giá.
  - Đây là stacked generalisation: trọng số ghép được học trên dữ liệu chưa dùng để fit các thành phần. Nhờ vậy `stack`
    thích nghi theo từng run, nhưng trơn hơn quy tắc "chọn một trong hai" của EXT4.
- `rank_recon`: trung bình hạng ECDF (trên calibration sạch) của `full_recon` và `gba_recon`. Không fit gì.

Mọi thành phần còn lại giữ như EXT6: bộ tái tạo, ngưỡng và siêu tham số HGB. Code: `scripts/ext7_stack.py`.

## Thiết kế: kiểm định chính trên SEED MỚI
- Phương pháp được định nghĩa đầy đủ ở trên **trước khi** chạy trên bất kỳ dữ liệu nào.
- Vì vậy kiểm định chính dùng các backbone **seed 52–56 chưa từng được đánh giá** cho bộ chọn lọc này:
  - C-MAPSS FD001–4 × GRU/TCN: đã train ở EXT6-C;
  - N-CMAPSS, vòng bi (6 điều kiện vận hành) và pin, × GRU/TCN: train mới, config giống hệt bài
    (`run_external.py --config configs/external.json`).
  - Tổng cộng 120 run.
- Lưu ý về C-MAPSS seed 52–56: EXT6-C đã xem kết quả của `full` và `full_recon` trên các run này, nhưng chưa xem `stack`.
- 240 run cũ (seed 42–51) chỉ là phân tích phụ.

## Họ kiểm định chính (Holm trên 4 kiểm định)
- So sánh `stack` trừ `full`: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy.
- Seed 52–56, mỗi dataset 10 cặp (backbone, seed). Gộp như bài: C-MAPSS lấy trung bình các subset; vòng bi gộp theo số
  unit của từng fold. Wilcoxon hai phía.
- **Tiêu chí "tốt hơn bài ESANN":**
  - không kém có ý nghĩa trên dataset nào;
  - tốt hơn có ý nghĩa trên ít nhất một dataset;
  - tỷ lệ lỗi bi quan ở coverage 90% không tăng quá 1 điểm trên dataset nào.
- Lưu ý về độ mạnh: với 10 cặp, p hai phía nhỏ nhất là 0.002, nên chỉ hiệu ứng nhất quán (khoảng ≥ 9/10 cặp) mới qua
  được Holm.

## Phân tích phụ (chưa hiệu chỉnh)
- Trên seed mới:
  - `rank_recon` trừ `full`;
  - `gba_recon` trừ `gauss_bias_aware`;
  - `full_recon` trừ `full` ở N-CMAPSS, vòng bi và pin (xác nhận thêm ngoài C-MAPSS).
- Cùng các so sánh trên 240 run cũ.
- Ngưỡng frozen: risk và coverage.
- Hệ số của stacker theo dataset.
- Risk trên dữ liệu sạch.
