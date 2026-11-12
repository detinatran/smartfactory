# EXT7: bộ chọn lọc ghép theo tác động của lỗi (`stack`). Kết quả ngày 2026-11-11

Protocol: `research/selective-rul/EXT7_PROTOCOL.md`, chốt trước khi chạy. Code: `scripts/ext7_stack.py`, `scripts/ext7_summary.py`.
- Train 80 backbone seed mới 52–56 cho N-CMAPSS, vòng bi và pin (CPU trên van, 25 phút, 0 lỗi). C-MAPSS seed mới lấy từ EXT6-C.
- Đánh giá 359/360 run, 27 643 hàng. Toàn bộ output in ra nằm trong `summary_output.txt`.
- **1 run lỗi:** `bearings_tcn_femto_c1_seed43`, seed cũ, nên chỉ ảnh hưởng phân tích phụ.
  - Nguyên nhân: trên unit calibration không có cửa sổ lỗi lạc quan nào, nên stacker chỉ thấy một lớp và không fit được.
  - Khi gộp vòng bi cho cặp này, trọng số được chuẩn hoá lại trên 5 fold còn lại.
  - Dòng "FAILED: 9" trong log là đếm nhầm: 8 dòng còn lại là "failed=0" của log train.

## Họ kiểm định chính: seed mới 52–56, `stack` − `full`, risk @90% matched, Holm trên 4

| Dataset | Hiệu (điểm) | Số cặp thắng | p | p Holm | Lỗi bi quan tăng (điểm) |
|---|---|---|---|---|---|
| C-MAPSS | **−0.56** | 10/10 | 0.002 | **0.008** | +0.37 |
| N-CMAPSS | +0.09 | 4/10 | 0.19 | 0.39 | 0.00 |
| Vòng bi | **+0.85** | 2/10 | 0.11 | 0.32 | +0.03 |
| Pin | +0.01 | 7/10 | 0.38 | 0.39 | +0.10 |

**Theo tiêu chí đã chốt trước: ĐẠT.**
- Tốt hơn có ý nghĩa ở C-MAPSS.
- Không kém có ý nghĩa ở dataset nào.
- Lỗi bi quan tăng tối đa 0.37 điểm, dưới ngưỡng 1 điểm.

## Nhưng phải nói rõ: vòng bi kém đi một cách nhất quán
- Ở seed mới: +0.85 điểm, chỉ thắng 2/10 cặp. Không có ý nghĩa sau Holm, nhưng cùng chiều với seed cũ.
- Ở seed cũ: +0.95 điểm, thắng 4/20, p = 0.001.
- Dữ liệu sạch, seed mới: 25.97 so với 25.17.
- Ngưỡng frozen, seed mới: coverage giảm mạnh ở vòng bi (80.0 so với 86.6) và pin (71.6 so với 78.3).
- Vì vậy kết luận "không kém ở dataset nào" chỉ đúng theo nghĩa thống kê hình thức với 10 cặp. **Không thể** viết là
  "`stack` tốt hơn hoặc bằng ở mọi nơi".
- Giải thích có thể (chưa kiểm chứng):
  - Fold vòng bi có rất ít unit calibration (có fold không có cửa sổ lạc quan nào), nên stacker fit kém ổn định.
  - Thêm vào đó, `impact` vẫn có trọng số dương ở vòng bi (0.25), trong khi riêng `impact` lại tệ ở vòng bi (26.9 so với
    25.2).

## Lợi ích ở seed cũ không lặp lại ở N-CMAPSS và pin
| Dataset | Seed cũ (phân tích phụ) | Seed mới (kiểm định chính) |
|---|---|---|
| C-MAPSS | −0.41 (18/20) | −0.56 (10/10) |
| N-CMAPSS | −0.32 (17/20) | +0.09 (4/10) |
| Vòng bi | +0.95 (4/20) | +0.85 (2/10) |
| Pin | −0.39 (14/20) | +0.01 (7/10) |

Nếu chỉ nhìn seed cũ, `stack` trông như thắng 3/4 dataset. Seed mới cho thấy chỉ C-MAPSS là thật. Điều này một lần nữa cho
thấy vì sao cần kiểm định trên seed mới.

## Phân tích phụ (seed mới, chưa hiệu chỉnh)
- So sánh trên C-MAPSS:
  - `stack` (−0.56) tốt hơn `full_recon` (−0.31) và `rank_recon` (−0.24), so với `full`.
  - Risk lần lượt: 8.82 / 9.07 / 9.15 / 9.38 (`full`).
  - `stack` là điểm tốt nhất trên C-MAPSS từ trước đến nay. Tương đối, nó giảm khoảng 6% so với bài ESANN.
- `full_recon` − `full` ở N-CMAPSS: **+0.20, 1/10, p = 0.004**, tức kém hơn. Ở seed cũ hiệu này là +0.03 (hoà). Như vậy
  `full_recon` không chỉ "không giúp" ngoài C-MAPSS mà có thể hơi hại ở N-CMAPSS.
- `gba_recon` − `gauss_bias_aware`: C-MAPSS −0.46 (10/10), các dataset khác hoà.
- `rank_recon` (không fit gì): an toàn nhất, không dataset nào kém rõ, nhưng lợi ích nhỏ.
  - C-MAPSS −0.24 (9/10).
  - Pin −0.20 (7/10, không có ý nghĩa).
- Hệ số stacker (đầu vào đã chuẩn hoá):
  - Hai bộ phân loại (`full_recon`, `gba_recon`) chiếm phần lớn trọng số.
  - `impact` chỉ có trọng số đáng kể ở C-MAPSS (0.28) và vòng bi (0.25), gần 0 ở N-CMAPSS và pin.
- Ngưỡng frozen trên C-MAPSS: risk 7.97 so với 9.23, nhưng coverage 86.6 so với 89.6. Một phần lợi ích là nhờ từ chối nhiều
  hơn.

## Kết luận
1. **Đạt tiêu chí đăng ký trước.** `stack` giảm risk C-MAPSS 0.56 điểm, 10/10 cặp trên seed mới. Đây là mức lớn nhất trong
   các vòng mở rộng, gần gấp đôi `full_recon`.
2. **Lợi ích chỉ ở C-MAPSS.** N-CMAPSS và pin hoà. Vòng bi kém khoảng 0.9 điểm một cách nhất quán qua hai bộ seed, dù không
   có ý nghĩa sau Holm ở seed mới.
3. **Không nên thay phương pháp chính của bài ESANN bằng `stack`.** Một phương pháp tốt ở 1/4 họ dữ liệu và có xu hướng hại
   ở 1/4 khác thì không phải là cải tiến chung.
4. Cách viết trung thực cho bài mở rộng:
   - Trên C-MAPSS (cảm biến tương quan mạnh), ghép bộ phân loại tái tạo với Gaussian có bias giảm risk khoảng 6% tương
     đối. Kết quả xác nhận trên seed mới.
   - Trên vòng bi (ít unit calibration), stacking có xu hướng hại.
5. Hướng tiếp, nếu muốn: thêm cơ chế an toàn, ví dụ quay về `full` khi calibration quá nhỏ hoặc có quá ít lỗi lạc quan.
   Quy tắc đó phải được chốt trước và kiểm định trên một bộ seed mới nữa (57–61). Không được chỉnh trên seed 52–56 rồi tự
   công bố.
