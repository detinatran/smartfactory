# EXT8: stacking có cơ chế an toàn và bộ seed giữ riêng. Chốt ngày 2026-11-11, TRƯỚC khi chạy

## Bối cảnh
EXT7: `stack` giảm risk C-MAPSS 0.56 điểm, 10/10 cặp seed mới. Nhưng ở vòng bi nó kém đi khoảng 0.9 điểm, cùng chiều ở cả
hai bộ seed. Ngoài ra có 1 run lỗi vì calibration chỉ có một lớp. EXT8 thử các biến thể để giữ lợi ích C-MAPSS mà không hại
vòng bi.

## Ứng viên
Code: `scripts/ext8_select.py`. Code giống hệt `ext7_stack.py`, chỉ thêm 4 ứng viên. Mọi stacker gặp dữ liệu chỉ một lớp
đều quay về `full`.

| Tên | Định nghĩa |
|---|---|
| `stack_plus` | Logistic trên [logit `full`, logit `gauss_bias_aware`, logit `full_recon`, logit `gba_recon`, `impact`], regularise mạnh hơn (C = 0.1). Vì có `full` trong đầu vào, nó có thể học lại `full`. |
| `blend` | Trung bình hạng ECDF (trên calibration sạch) của `full` và `stack`. Trộn cố định 50/50, không fit gì. |
| `stack_cv` | Chia unit calibration làm 2 nửa và tính risk out-of-fold của `stack` (90% matched, trung bình 3 shift đã thấy). Dùng `stack` nếu risk này thấp hơn `full`, ngược lại dùng `full`. |
| `select_cv` | Mỗi run chọn điểm có risk calibration thấp nhất trong {`full`, `full_recon`, `gauss_bias_aware`, `gba_recon`, `stack` (OOF)}. Bằng nhau thì chọn `full`. |

## Thiết kế hai giai đoạn
1. **Phát triển (EXT8, chạy ngay):** chạy mọi ứng viên trên seed 42–56.
   - Đó là 360 run, 30 cặp mỗi dataset: 240 run cũ, 40 run C-MAPSS của EXT6-C và 80 run external của EXT7.
   - Đây là phân tích khám phá. Seed 52–56 đã được xem ở EXT7, nên kết quả giai đoạn này **không** dùng để công bố "tốt
     hơn".
2. **Bộ giữ riêng (holdout):** train ngay backbone seed **57–61** cho cả 4 họ dữ liệu, tổng 120 run. Config giống hệt bài.
   - **Không đánh giá bất kỳ bộ chọn lọc nào** trên các run này cho đến khi chốt phương pháp.
   - Sau khi chốt, chạy đúng một phương pháp đó trên 57–61, với tiêu chí giống EXT7:
     - so với `full`, Holm trên 4 dataset;
     - tốt hơn có ý nghĩa ở ít nhất một dataset;
     - không kém có ý nghĩa ở dataset nào;
     - lỗi bi quan tăng không quá 1 điểm.
   - Bổ sung so với EXT7: hiệu trung bình **không được dương quá 0.3 điểm** ở dataset nào, có ý nghĩa thống kê hay không.

## Quy tắc chọn mặc định sau giai đoạn phát triển
- Chỉ xét ứng viên **an toàn**: hiệu so với `full` ≤ +0.10 điểm ở mọi dataset (30 cặp), và lỗi bi quan tăng ≤ 1 điểm.
- Trong các ứng viên an toàn, chọn ứng viên có **tổng** hiệu (cộng 4 dataset) âm nhất.
- Nếu không có ứng viên an toàn nào giảm được tổng hiệu, kết luận là giữ `full`.
- Người dùng có thể đổi lựa chọn sau khi xem kết quả phát triển, miễn là chốt **trước khi** chạy holdout.

## Ghi chép thêm
Lựa chọn của `stack_cv` và `select_cv` theo dataset, số run phải fallback, và risk calibration của từng ứng viên.

## Bổ sung ngày 2026-11-12: chốt TRƯỚC khi đánh giá holdout
Kết quả giai đoạn phát triển: `ext8_v1/FINDINGS.md`. Quy tắc mặc định chọn `rank_recon`. Người dùng yêu cầu chạy **mọi**
ứng viên trên holdout 57–61. Để không chọn người thắng sau khi xem kết quả, chốt như sau:
- **Kiểm định chính:** `rank_recon` − `full`, với tiêu chí đầy đủ ở trên (Holm trên 4 dataset, cộng các điều kiện an toàn).
  Chỉ kết quả này được gọi là "xác nhận".
- **Phụ:** tất cả 8 ứng viên trừ `full`: `stack`, `stack_plus`, `blend`, `stack_cv`, `select_cv`, `full_recon`,
  `rank_recon`, `gba_recon`.
  - Hiệu chỉnh Holm trên toàn bộ 32 kiểm định (8 × 4).
  - Một ứng viên khác `rank_recon` chỉ được coi là thay thế nếu đạt mọi tiêu chí với p đã hiệu chỉnh theo 32 kiểm định.
- Code: `scripts/ext8_select.py` không đổi, chạy trên `runs/ext8_hold_cmapss` và `runs/ext8_hold_ext`.
  Kiểm định: `scripts/ext8_holdout.py`.
