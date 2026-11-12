# EXT10: huấn luyện bộ chọn lọc với nhiễu loạn ngẫu nhiên (RPT). XÁC NHẬN ngày 2026-11-12

Protocol: `research/selective-rul/EXT10_PROTOCOL.md` và quy tắc vòng lặp trong `EXT9_PROTOCOL.md`. Nhật ký:
`SEARCH_LOG.md`.
- Giai đoạn phát triển: seed 42–61, 480 run (`ext10_v1/`).
- Xác nhận: seed **62–71**, chưa từng dùng, 240 run, 0 lỗi.
- Phương pháp chính `full_recon_rpt` được chốt trước khi đánh giá seed mới.
- Kiểm định: `scripts/conf_test.py`. Output: `confirm_output.txt`.

## Kiểm định xác nhận: `full_recon_rpt` − `full`, 20 cặp mỗi dataset, Holm trên 4, α = 0.025

| Dataset | Hiệu (điểm) | Số cặp thắng | p Holm | Lỗi bi quan tăng |
|---|---|---|---|---|
| C-MAPSS | **−0.52** | 20/20 | 0.00001 | +0.35 |
| N-CMAPSS | **−0.14** | 17/20 | 0.0006 | 0.00 |
| Vòng bi | **−0.55** | 17/20 | 0.0003 | +0.32 |
| Pin | **−0.32** | 18/20 | 0.0032 | +0.40 |

**XÁC NHẬN: tốt hơn có ý nghĩa ở cả 4 dataset.** Không dataset nào kém, và lỗi bi quan tăng tối đa 0.40 điểm.
- Đây là phương pháp đầu tiên trong mọi vòng mở rộng vượt bài ESANN trên **cả 4** họ dữ liệu, kể cả vòng bi.
- Risk @90% (%): C-MAPSS 8.52 so với 9.05; N-CMAPSS 12.28 so với 12.42; vòng bi 25.26 so với 25.80; pin 3.47 so với 3.79.
- Tương đối, đó là giảm 5.8% / 1.1% / 2.1% / 8.4%.
- Dữ liệu sạch không bị hại: 8.28 / 7.67 / 25.43 / 2.24, so với 8.25 / 8.00 / 25.93 / 2.51.

## Theo từng loại lỗi (điểm, số cặp thắng trên 20)

| Lỗi | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| bias | −1.02 (20) | −0.17 (12) | −0.63 (13) | −1.44 (17) |
| noise | −0.78 (20) | −0.14 (17) | −0.57 (17) | −0.01 (12) |
| spike | −0.74 (19) | −0.16 (13) | −0.45 (16) | −0.04 (16) |
| intermittent | −0.31 (15) | −0.09 (11) | −0.50 (18) | −0.22 (12) |
| stuck | −0.20 (16) | −0.13 (16) | −0.60 (17) | −0.15 (10) |
| gain | −0.09 (12) | −0.16 (14) | −0.52 (16) | −0.05 (12) |
| clean | +0.03 (7) | −0.33 (18) | −0.50 (16) | −0.27 (13) |

- Lợi ích trải rộng trên mọi loại lỗi, không chỉ ở các lỗi giống họ nhiễu loạn ngẫu nhiên.
- **Lưu ý trung thực:** lợi ích lớn nhất là ở bias (C-MAPSS −1.02, pin −1.44), nơi họ hàm ngẫu nhiên chồng lấn nhiều nhất
  (thành phần hằng số). Ở pin, phần lớn lợi ích đến từ bias.
- Nếu bỏ bias, trung bình 5 lỗi còn lại vẫn âm ở cả 4 dataset: khoảng −0.42 / −0.14 / −0.53 / −0.09.

## Cần làm thêm trước khi viết bài (chưa chạy)
1. **Ablation "chỉ thêm một bản sao sạch":** cùng trọng số, nhưng không nhiễu loạn. Mục đích là tách tác động của nhiễu loạn
   khỏi việc đổi cân bằng dữ liệu.
2. **Ablation họ nhiễu loạn bỏ thành phần hằng số (m = 0):** kiểm tra lợi ích có phải chủ yếu do chồng lấn với bias.
3. **Độ nhạy theo biên độ A và số kênh k.**

Tất cả là phân tích phụ, chạy trên seed 42–61, không ảnh hưởng kết quả xác nhận.

## Mới ở đâu so với bài ESANN
Bài ESANN học bộ chọn lọc từ 2 loại lỗi cụ thể (dropout, drift). RPT học từ một **họ nhiễu loạn ngẫu nhiên tổng quát**, tức
domain randomization cho selective prediction dưới lỗi cảm biến chưa thấy. Kết hợp với đặc trưng tái tạo kênh từ EXT6.
