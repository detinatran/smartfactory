# EXT4 — chọn điểm số theo calibration, chốt ngày 2026-11-10, TRƯỚC khi có kết quả

## Bối cảnh
- Sau EXT3 (`extension-results/2026-11-09/ext3_v1/FINDINGS.md`), người dùng yêu cầu chạy tiếp quy tắc chọn theo calibration.
- **Công khai:** quy tắc này được đề xuất sau khi đã thấy kết quả unc_v1 và EXT3. Ở đó `gauss_bias_aware` thắng `full` trên
  N-CMAPSS và pin, thua trên C-MAPSS. Vì vậy đây không phải phân tích độc lập với dữ liệu đã thấy. Điều quy tắc bảo đảm chỉ là
  nó không dùng nhãn hay đầu vào của tập đánh giá.

## Thiết lập
- Giống EXT3: 240 run của bài ESANN (GRU/TCN × seed 42–51 × 4 họ dữ liệu), CPU replay, cùng các mô hình HGB.
- Code: `scripts/ext4_select.py`.

## Quy tắc `select_cal` (theo từng run, chốt trước)
- Ứng viên: `full` (bộ phân loại của bài) và `gauss_bias_aware`.
- Tiêu chí: trung bình risk ở coverage matched 0.5, 0.6, 0.7, 0.8, 0.9 (một dạng AURC rút gọn).
  - Tính trên **unit calibration × {clean, dropout, drift}**, tức là chỉ dùng nhãn calibration và các shift đã thấy khi train.
- Chọn ứng viên có tiêu chí nhỏ hơn. Nếu bằng nhau thì giữ `full`.
- Điểm được chọn được dùng cho mọi scenario đánh giá của run đó, kể cả ngưỡng frozen của chính nó.

## Họ kiểm định chính (Holm trên 4 kiểm định)
- So sánh `select_cal` trừ `full`: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy.
- 4 dataset, mỗi dataset 20 cặp; gộp như bài. Wilcoxon hai phía.
- **Tiêu chí "tốt hơn bài ESANN":** không kém có ý nghĩa trên dataset nào (p_Holm ≥ 0.05 hoặc hiệu ≤ 0), và tốt hơn có ý
  nghĩa trên ít nhất một dataset.

## Phân tích phụ (chưa hiệu chỉnh)
- `select_cal_90`: cùng quy tắc nhưng tiêu chí là risk ở coverage 90%.
- `select_cal3`: thêm `rank_avg` làm ứng viên thứ ba.
- Ngưỡng frozen (risk và coverage) của `select_cal` so với `full`.
- Tỷ lệ run chọn từng ứng viên, theo dataset.

## Nếu thay vào bài ESANN
Chỉ cân nhắc thay nếu đạt tiêu chí trên. Khi đó vẫn phải ghi rõ:
- đây là phân tích post hoc;
- quy tắc được chọn sau khi đã thấy kết quả;
- phương pháp đăng ký trước là `full` với ngưỡng frozen.
