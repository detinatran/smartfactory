# EXT8: giai đoạn phát triển (seed 42–56). Kết quả ngày 2026-11-12

Protocol: `research/selective-rul/EXT8_PROTOCOL.md`. Code: `scripts/ext8_select.py`, `scripts/ext8_summary.py`.
- 360/360 run, 0 lỗi. Có 37 800 hàng. Toàn bộ output nằm trong `summary_output.txt`.
- Run vòng bi từng lỗi ở EXT7 nay chạy được nhờ cơ chế fallback. Fallback chỉ xảy ra ở 1% run vòng bi.
- Holdout seed 57–61: đã train 120/120 backbone, **chưa đánh giá**.
- Đây là phân tích **khám phá**: seed 52–56 đã được xem ở EXT7, và các p-value chưa hiệu chỉnh.

## Ứng viên trừ `full`: risk @90% matched, điểm, 30 cặp mỗi dataset

| Ứng viên | C-MAPSS | N-CMAPSS | Vòng bi | Pin | Tổng | Tệ nhất | An toàn? |
|---|---|---|---|---|---|---|---|
| **rank_recon** | −0.14 (24/30) | −0.20 (25/30) | −0.03 (16/30) | −0.30 (23/30) | **−0.67** | −0.03 | **có** |
| select_cv | −0.11 (19) | −0.19 (16) | −0.05 (16) | −0.32 (18) | −0.66 | −0.05 | có |
| full_recon | −0.27 (30) | +0.09 (11) | +0.05 (15) | 0.00 (14) | −0.13 | +0.09 | có |
| blend | −0.33 (30) | −0.19 (27) | +0.40 (11) | −0.23 (24) | −0.35 | +0.40 | không |
| stack_cv | −0.20 (26) | −0.13 (10) | +0.20 (11) | −0.18 (9) | −0.31 | +0.20 | không |
| stack_plus | −0.48 (29) | −0.26 (27) | +0.96 (6) | −0.27 (20) | −0.04 | +0.96 | không |
| stack | −0.46 (28) | −0.18 (21) | +0.90 (6) | −0.26 (21) | +0.01 | +0.90 | không |

Ngoặc là số cặp thắng trên 30. Lỗi bi quan tăng tối đa 0.32 điểm ở mọi ứng viên.

**Quy tắc chọn mặc định chọn `rank_recon`:**
- Đây là trung bình hạng ECDF của `full_recon` và `gba_recon`.
- Ứng viên này không fit thêm gì, đã định nghĩa từ EXT7.
- Nó không kém `full` ở dataset nào.

## Nhận xét
1. **Không biến thể nào sửa được vòng bi mà vẫn giữ lợi ích C-MAPSS của `stack`.**
   - `stack_plus` tốt nhất ở C-MAPSS (−0.48) nhưng vòng bi vẫn +0.96.
   - `blend` giảm thiệt hại vòng bi xuống còn +0.40, nhưng chưa đủ.
   - `stack_cv`: kiểm tra chéo trên calibration quá nhiễu để biết khi nào nên dùng `stack`.
2. **`rank_recon` và `select_cv` an toàn nhưng lợi ích nhỏ, và giảm khi chuyển sang seed mới:**

   | Ứng viên | Bộ seed | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
   |---|---|---|---|---|---|
   | rank_recon | cũ, 20 cặp | −0.10 | −0.31 | −0.06 | −0.35 |
   | rank_recon | mới, 10 cặp | −0.24 | +0.01 | +0.04 | −0.20 |
   | select_cv | cũ, 20 cặp | −0.11 | −0.27 | −0.07 | −0.50 |
   | select_cv | mới, 10 cặp | −0.10 | −0.01 | −0.01 | +0.05 |

   Lợi ích ở N-CMAPSS chỉ có ở seed cũ. Lợi ích của `select_cv` gần như biến mất ở seed mới.
3. **Dự đoán cho bước holdout của `rank_recon`:**
   - Khả năng cao chỉ C-MAPSS (và có thể pin) tốt hơn có ý nghĩa.
   - Các dataset khác nhiều khả năng hoà.
   - Mức cải thiện nhỏ: khoảng 0.1–0.3 điểm.
4. Trên dữ liệu sạch, `rank_recon` không gây hại: C-MAPSS 8.73 so với 8.72; N-CMAPSS, vòng bi và pin đều bằng hoặc tốt hơn.

## Bước tiếp
- Chạy đúng **một** phương pháp đã chốt trên holdout 57–61 với tiêu chí trong protocol.
  - So với `full`, Holm trên 4 dataset.
  - Phải tốt hơn có ý nghĩa ở ít nhất một dataset, và không kém có ý nghĩa ở dataset nào.
  - Lỗi bi quan tăng không quá 1 điểm.
  - Hiệu trung bình không dương quá 0.3 điểm ở dataset nào.
- Mặc định chạy `rank_recon`.
