# EXT3 — conformal theo shift và điểm kết hợp, chốt ngày 2026-11-10, TRƯỚC khi có kết quả

Người dùng yêu cầu chạy hướng đã đề xuất sau EXT2 (xem `extension-results/2026-11-09/ext2_*/FINDINGS.md`), với mục tiêu
nếu được thì tốt hơn phương pháp của bài ESANN. Kết quả dùng cho bài mở rộng. Code: `scripts/ext3_conformal.py`.

## Thiết lập
Dùng đúng thiết lập của bài ESANN:
- toàn bộ run GRU và TCN, seed 42–51, trên C-MAPSS FD001–4, N-CMAPSS, vòng bi và pin (240 run);
- cùng checkpoint, split, cửa sổ và seed lỗi; CPU replay;
- không train lại, không dùng official test của C-MAPSS.

Mọi bộ chấm điểm dùng các thiết lập HGB cố định như bài.

## Điểm số
Định nghĩa chi tiết nằm trong docstring của script.
- `full`: bộ phân loại của bài, dùng làm mốc so sánh.
- `gauss_bias_aware`.
- `conformal`: biên một phía e ≤ b + q s, với q lấy trên calibration sạch.
- **Mới:**
  - `conformal_w`: weighted conformal dưới covariate shift (Tibshirani et al., 2019).
    - Pool calibration gồm unit calibration × {clean, dropout, drift}.
    - Tỷ số mật độ lấy từ một bộ phân loại miền giữa pool và các cửa sổ đánh giá chưa có nhãn của từng scenario. Nhãn không
      bao giờ được dùng; khi triển khai thực tế thì đầu vào cũng có sẵn như vậy.
    - Xác suất miền được kẹp trong [0.01, 0.99].
  - `mondrian`: conformal theo nhóm, chia 4 bin theo tứ phân vị của RUL dự đoán trên pool.
  - `rank_avg`: trung bình hai biến đổi CDF calibration của `full` và `gauss_bias_aware`.
    **Công khai:** ý tưởng này đến từ việc đã thấy kết quả của unc_v1, nơi hai điểm này thắng ở các dataset khác nhau.
    Đây là lựa chọn post hoc trên chính các run đó.
  - `laplace_ev`: Laplace lớp cuối, prior precision chọn bằng evidence của Laplace (lưới 1e-4 đến 1e4). Đây là bản chuẩn
    hơn `laplace` của EXT2.
- `ensemble_std`: tham chiếu.
- **Quy tắc "certify"** (chỉ cho các điểm conformal): nhận một cửa sổ nếu và chỉ nếu b + q s ≤ δ, tức là chặn trên của sai số
  đã nằm dưới ngưỡng lạc quan. Báo cáo risk và coverage của quy tắc này.

## Họ kiểm định chính (Holm trên 12 kiểm định)
- Phép so sánh: {`conformal_w`, `mondrian`, `rank_avg`} trừ `full`.
- Chỉ số: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy.
- Phạm vi: 4 dataset. Mỗi dataset có 20 cặp (backbone, seed); vòng bi gộp theo trọng số số unit, C-MAPSS lấy macro.
- Wilcoxon hai phía.
- Chỉ được nói "tốt hơn bài ESANN" cho một điểm nếu nó thắng sau Holm, và phải công khai là `rank_avg` mang tính post hoc.

## Phân tích phụ (mô tả, chưa hiệu chỉnh)
- Độ phủ của biên trên **toàn bộ cửa sổ và riêng cửa sổ bị lỗi**, cho `conformal`, `conformal_w` và `mondrian`; mục tiêu 90%.
  - Câu hỏi chính: trên vòng bi (shift giữa các điều kiện vận hành), `conformal_w` có kéo độ phủ từ khoảng 83% về gần 90%
    không? Cửa sổ bị lỗi có hụt độ phủ không?
  - Kích thước mẫu hiệu dụng của trọng số (`weight_ess`) được báo cáo để đánh giá độ ổn định.
- Risk và coverage của quy tắc certify.
- `laplace_ev` so với `ensemble_std` (4 kiểm định).
- Ngưỡng frozen: risk và coverage.

## Ngoài phạm vi
Chưa tune FiLM và Lipschitz. Việc đó cần train lại, và EXT2 không cho thấy lợi ích nhất quán; sẽ quyết sau vòng này.
