# EXT5: độ nhạy khi bỏ từng kênh (leave-one-channel-out, LOCO). Chốt ngày 2026-11-11, TRƯỚC khi có kết quả

## Bối cảnh và ý tưởng
- Sau EXT2–EXT4: mọi điểm dạng variance đều thất bại. Thứ còn thiếu là bias có điều kiện b(x). Nhưng b(x) học từ
  đặc trưng thống kê của cửa sổ, nên chỉ thấy lỗi một cách gián tiếp.
- Hướng mới, khác các vòng trước, nhắm thẳng vào cơ chế của bài:
  - lỗi cảm biến chạm vào **một vài kênh** và đẩy **mọi member** theo cùng một hướng;
  - nếu che (mask) đúng kênh lỗi, dự đoán sẽ quay về gần giá trị sạch.
- Vì vậy, với mỗi cửa sổ và mỗi kênh c, tính dc = ŷ − ŷ₋c. Ở đây ŷ₋c là dự đoán trung bình ensemble khi kênh c bị che.
- Che kênh = đặt giá trị kênh về 0 và cờ thiếu dữ liệu về 1. Đây đúng là cách mô hình đã thấy dữ liệu thiếu (shift
  "dropout" và các giá trị NaN), nên không cần train lại backbone.
- Đặc trưng LOCO, tính theo từng cửa sổ:
  - `loco_up` = max_c dc: có kênh nào một mình đẩy dự đoán **lên** không. Đây là chiều gây lỗi lạc quan.
  - `loco_down` = min_c dc.
  - `loco_abs` = max_c |dc|.
  - `loco_std` = độ lệch chuẩn của dc theo c.
- Khác biệt về nguyên lý so với spread:
  - spread bất biến với dịch chuyển chung (Prop. 3);
  - dc thì không, và còn có dấu, nên phân biệt được dịch lên với dịch xuống.

## Thiết lập
- Giống EXT3 và EXT4: 240 run của bài ESANN (GRU/TCN × seed 42–51 × 4 họ dữ liệu), CPU replay.
- Các mô hình HGB có cùng siêu tham số. Train trên unit selector × {clean, dropout, drift}.
- Code: `scripts/ext5_loco.py`.
- **Công khai:**
  - Tập đánh giá đã được dùng ở các vòng trước, nên kết quả ở đây là **exploratory**, không phải xác nhận.
  - Ý tưởng LOCO không đến từ việc xem kết quả của nó: nó chưa từng được tính trên các run này.
  - Nếu đạt tiêu chí bên dưới, bước tiếp theo bắt buộc là một vòng xác nhận với seed mới (train lại backbone, protocol
    riêng) trước khi viết thành kết quả.

## Điểm so sánh
- `full`: bộ phân loại của bài (tham chiếu).
- `full_loco`: cùng bộ phân loại, thêm 4 đặc trưng LOCO. **Đây là điểm chính.**
- `loco_up`: dùng thẳng `loco_up` làm điểm, không cần học.
- `gauss_bias_aware`: tham chiếu, giống EXT3.
- `gba_loco`: Gaussian có bias, với b(x) và v(x) học trên đặc trưng có LOCO.
- `ensemble_std`: tham chiếu.

## Họ kiểm định chính (Holm trên 4 kiểm định)
- So sánh `full_loco` trừ `full`: risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy.
- 4 dataset, mỗi dataset 20 cặp (backbone, seed); gộp như bài; Wilcoxon hai phía.
- **Tiêu chí "tốt hơn bài ESANN"** giống EXT4:
  - không kém có ý nghĩa trên dataset nào;
  - tốt hơn có ý nghĩa trên ít nhất một dataset.

## Phân tích phụ (chưa hiệu chỉnh)
- `gba_loco` trừ `gauss_bias_aware`; `loco_up` trừ `ensemble_std`.
- Ngưỡng frozen: risk và coverage của từng điểm.
- Tỷ lệ lỗi lạc quan **do fault gây ra** bị loại, ở ngưỡng frozen. Đó là các cửa sổ lạc quan khi có lỗi nhưng không lạc
  quan khi sạch.
- Định vị: trên các cửa sổ bị ảnh hưởng, tỷ lệ cửa sổ mà kênh có |dc| lớn nhất đúng là một kênh bị lỗi (so với mức
  ngẫu nhiên, 3/số kênh).
- Risk trên dữ liệu sạch, để kiểm tra LOCO có làm hại khi không có lỗi không.
