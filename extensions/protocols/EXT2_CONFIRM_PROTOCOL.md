# EXT2 xác nhận + mở rộng — chốt ngày 2026-11-10, TRƯỚC khi có kết quả

Người dùng yêu cầu chạy cả hai bước sau pilot `ext2_v1` (FINDINGS: `extension-results/2026-11-09/ext2_v1/FINDINGS.md`).
Code giữ nguyên `scripts/ext2_pilot.py`, chỉ thêm tuỳ chọn `--backbones`. Mọi cấu hình backbone và điểm số giống hệt
EXT2_PILOT_PROTOCOL.md và không thay đổi gì sau khi xem kết quả pilot. Chạy CPU trên van.

## A. Vòng xác nhận (`ext2_confirm`)
- Seed mới 47–51, chưa dùng trong pilot.
  - C-MAPSS FD001–4: 20 run.
  - N-CMAPSS: 5 run.
- Backbone: `gru` và `film_gru`. Lipschitz không xác nhận vì pilot không cho thấy lợi ích.
- Các giả thuyết được rút ra từ pilot, nên đây là kiểm định **xác nhận**. Họ 4 kiểm định chính được hiệu chỉnh Holm:
  - **C1.** risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy: FiLM + `gauss_bias_aware` < GRU + `gauss_bias_aware`,
    kiểm trên C-MAPSS và N-CMAPSS (2 kiểm định).
  - **C2.** cũng so sánh đó với `full` (bộ phân loại một phía), trên C-MAPSS và N-CMAPSS (2 kiểm định).
- Cặp so sánh: C-MAPSS dùng (seed, subset), 20 cặp; N-CMAPSS dùng seed, 5 cặp.
  - Với N-CMAPSS, 5 cặp chỉ cho p nhỏ nhất 0.0625, nên không thể đạt mức ý nghĩa 0.05.
  - Vì vậy ghi trước: N-CMAPSS chỉ được báo cáo theo hướng (bao nhiêu trên 5 seed cùng chiều), kèm cặp gộp pilot + xác nhận
    (10 seed) như một phân tích phụ, ghi rõ là đã thấy một nửa dữ liệu.
  - Wilcoxon hai phía.
- **Tái lập H1:** shift/spread của FiLM > GRU, và tỷ lệ loại lỗi lạc quan do fault (`reject_new`) của spread FiLM < GRU.
  Phân tích mô tả.
- Không chọn lại tổ hợp. Nếu C1/C2 không qua, báo cáo đúng như vậy.

## B. Mở rộng (`ext2_ext`)
- Seed 42–46 như pilot.
  - Vòng bi: 6 điều kiện để riêng × 5 seed = 30 run.
  - Pin: 5 run.
- Cả 3 backbone, 8 điểm, giống pilot.
- Kiểm tra lại H1–H4 của pilot trên hai họ dữ liệu này. Mô tả, chưa hiệu chỉnh.
- Vòng bi gộp theo trọng số số unit của từng fold, như bài.
