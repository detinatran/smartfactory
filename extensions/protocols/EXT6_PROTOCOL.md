# EXT6: sửa đầu vào nhờ dư thừa giữa các cảm biến, rồi mới chọn lọc ("repair, then select"). Chốt ngày 2026-11-11, TRƯỚC khi có kết quả

## Bối cảnh và ý tưởng
- Sau EXT2–EXT5, cách chấm điểm gần như đã bão hoà. Giới hạn nằm ở bộ dự đoán điểm.
- EXT6 là vòng đầu tiên thay đổi chính dự đoán, nhưng **không train lại backbone**. Ý tưởng lấy từ chẩn đoán lỗi kinh điển
  (analytic redundancy): các cảm biến tương quan với nhau, nên mỗi kênh có thể được dự đoán lại từ các kênh còn lại.
- **Mô hình tái tạo:**
  - Với mỗi kênh c, fit một Ridge (α = 1) dự đoán giá trị của kênh c tại mỗi bước thời gian từ các kênh còn lại (C-MAPSS
    dùng thêm 3 setting).
  - Dữ liệu fit là **unit fit, dữ liệu sạch, các bước không thiếu giá trị**. Đây đúng là dữ liệu backbone đã học, nên
    không dùng nhãn nào của tập selector, calibration hay đánh giá.
  - Lấy tối đa 200 000 bước, chọn ngẫu nhiên với seed của run.
  - Phần dư chuẩn hoá: z_tc = (x_tc − x̂_tc) / σ_c, trong đó σ_c là độ lệch chuẩn phần dư trên dữ liệu fit.
- **Phát hiện:**
  - Độ nghi ngờ của kênh c trong một cửa sổ là s_c = trung bình theo thời gian của |z_tc|.
  - Ngưỡng τ_c là phân vị 99.5% của s_c trên các cửa sổ fit sạch, nên tỷ lệ báo động nhầm khoảng 0.5% mỗi kênh. Ngưỡng
    được cố định theo dữ liệu, không chọn tay.
  - Kênh bị coi là nghi ngờ nếu s_c > τ_c, hoặc nếu có giá trị thiếu trong cửa sổ.
- **Sửa:**
  - Thay toàn bộ kênh nghi ngờ trong cửa sổ bằng giá trị tái tạo, và đặt cờ thiếu về 0.
  - Lặp 3 lần, để các kênh nghi ngờ được tái tạo từ giá trị đã sửa của nhau.
  - Cuối cùng tính lại dự đoán ŷ_rep bằng cùng ensemble.
- **Đặc trưng mới:**
  - `impact` = ŷ − ŷ_rep, có dấu: nếu thay kênh khả nghi thì dự đoán giảm bao nhiêu.
  - `n_susp`: số kênh nghi ngờ.
  - `max_ratio` = max_c s_c / τ_c.
- **Khác với các vòng trước:**
  - Bộ phát hiện lỗi ở bài ESANN chỉ trả lời "cửa sổ có bị lỗi không".
  - LOCO ở EXT5 che kênh bằng cách đánh dấu thiếu dữ liệu.
  - Ở đây kênh được thay bằng giá trị hợp lý trong phân phối. Cách này xử lý được cả intermittent, là lỗi mà LOCO bỏ sót.

## Thiết lập
- Giống EXT3–EXT5: 240 run của bài ESANN, CPU replay, cùng siêu tham số HGB. Code: `scripts/ext6_repair.py`.
- Tập đánh giá đã dùng ở các vòng trước, nên kết quả ở đây là **exploratory**. Nếu đạt tiêu chí, cần xác nhận bằng seed mới.

## Điểm số và bộ dự đoán
Trên bộ dự đoán gốc ŷ:
- `ensemble_std`, `full` (bài ESANN) và `gauss_bias_aware`: tham chiếu.
- `impact`: dùng thẳng làm điểm, không học.
- `full_recon`: bộ phân loại của bài, thêm 3 đặc trưng mới.

Trên bộ dự đoán đã sửa ŷ_rep:
- `repair_full`: bộ phân loại của bài, train lại với sai số của ŷ_rep. Đặc trưng là `selector_features` của cửa sổ đã sửa,
  cộng 3 đặc trưng mới.
- Risk của `repair_full` được tính trên ŷ_rep.

## Họ kiểm định chính (Holm trên 8 kiểm định)
Chỉ số: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy. Mỗi dataset 20 cặp (backbone, seed); Wilcoxon hai phía.
- A: `full_recon` trừ `full`, 4 dataset.
- B: `repair_full` (trên ŷ_rep) trừ `full` (trên ŷ), 4 dataset.

**Tiêu chí "tốt hơn bài ESANN"**, đánh giá riêng cho A và cho B:
- không kém có ý nghĩa trên dataset nào;
- tốt hơn có ý nghĩa trên ít nhất một dataset.

**Với B, thêm điều kiện chống "gian lận" bằng cách hạ dự đoán.** Ở coverage 90% và trung bình 6 lỗi, tỷ lệ lỗi bi quan
(ŷ − y < −δ) không được tăng quá 1 điểm trên dataset nào.

## Phân tích phụ (chưa hiệu chỉnh)
- Với ŷ_rep so với ŷ, ở coverage 100%:
  - RMSE (trên mục tiêu đã cap);
  - tỷ lệ lỗi lạc quan và lỗi bi quan;
  - tính cả khi sạch lẫn dưới từng loại lỗi.
- Phát hiện:
  - tỷ lệ báo động nhầm trên dữ liệu sạch;
  - độ nhạy theo từng loại lỗi, tức tỷ lệ kênh bị lỗi được đánh dấu nghi ngờ.
- `impact` so với `ensemble_std`.
- Ngưỡng frozen: risk và coverage của từng điểm.
