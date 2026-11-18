# EXT22: chọn tự động giữa OSC và RPT theo từng run (`rpt_auto`). Chốt ngày 2026-11-14, TRƯỚC khi chạy

**Lý do:** trên HDD (EXT21b, kiểm định ngoài), `rpt_L` kém `full`. Trên 4 họ dữ liệu phát triển thì `rpt_L` tốt hơn. Một
phương pháp có nguyên tắc nên tự phát hiện khi nào perturbation training không giúp, mà không dùng dữ liệu test.

## Phương pháp `rpt_auto` (`scripts/ext22.py`)
- Với mỗi run, tính **risk trên calibration** của `full` và `rpt_L`:
  - dữ liệu: unit calibration (rời với selector, fit và evaluation) dưới các shift đã thấy S = {clean, dropout, drift};
  - metric: risk @90% matched, trung bình trên 3 shift.
  Không dùng dữ liệu nhiễu, nên việc chọn không thiên vị RPT.
- `rpt_auto` = `rpt_L` nếu risk calibration của nó < risk của `full`; nếu không thì `full` (hoà thì chọn `full`).
- Biến thể phụ `auto3`: chọn trong {`full`, `rpt_L`, `err_reg_rptL`} theo cùng quy tắc. Chỉ thăm dò.

## Giai đoạn phát triển (không tiêu α)
- Dữ liệu: seed 42–61 cho 4 họ dữ liệu, cùng HDD seed 42–51 (`runs/hdd_ext21b`).
- **Quy tắc chuyển sang xác nhận:**
  1. `rpt_auto` không kém `full` quá +0.10 ở bất kỳ họ nào, kể cả HDD;
  2. tổng hiệu so với `full` trên 5 họ ≤ −0.50 điểm;
  3. lỗi bi quan tăng ≤ 1 điểm.

  Nếu không thoả, không mở vòng 4.

## Xác nhận vòng 4 (nếu mở)
- Seed **92–101** cho cả 5 họ dữ liệu (240 + 20 backbone train mới).
- **Họ kiểm định:** `rpt_auto` so với {`full`, `gauss_bias_aware`, `err_reg`} × 5 họ, tức 15 phép thử. Holm ở mức
  **α_4 = 0.003125**. Tổng α sau vòng này là 0.046875 ≤ 0.05.
- **Thành công:** quy tắc EXT9 so với `full` trên 5 họ.

## Kết quả phát triển, ngày 2026-11-15: MỞ vòng 4
480 run (4 họ, seed 42–61) + 20 run HDD (seed 42–51), 0 lỗi.

| | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD |
|---|---|---|---|---|---|
| `full` | 9.25 | 11.83 | 23.16 | 4.35 | 15.44 |
| `rpt_L` | 8.66 | 11.36 | 22.30 | 3.73 | 15.80 |
| `rpt_auto` | 8.93 | 11.42 | 22.51 | 3.73 | 15.51 |
| `rpt_auto` − `full` | −0.32 | −0.41 | −0.64 | −0.62 | +0.07 |
| Số run chọn `rpt_L` | 83/160 | 31/40 | 175/240 | 40/40 | 3/20 |

- Tổng hiệu −1.92 (≤ −0.50); không họ nào > +0.10; lỗi bi quan tăng tối đa +0.55. **Đạt quy tắc, nên mở vòng 4.**
- `auto3` (thăm dò): tổng −2.37.
- **Kiểm định vòng 4:** `scripts/ext22.py` trên `runs/conf92_cmapss` và `runs/conf92_ext` (gồm cả HDD seed 92–101).
  `rpt_auto` so với {`full`, `gauss_bias_aware`, `err_reg`} × 5 họ, Holm trên 15, α_4 = 0.003125.

## Kết quả vòng 4, ngày 2026-11-15: XÁC NHẬN
Seed 92–101, 260 run (gồm HDD), 0 lỗi. Holm trên 15, α_4 = 0.003125.

| `rpt_auto` − … | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD |
|---|---|---|---|---|---|
| `full` | −0.49 (20/20, 3e-5) | −0.27 (15/20, 0.0046) | −1.05 (17/20, 0.0024) | −0.67 (17/20, 0.0023) | +0.03 (2/20, 0.47) |
| `gauss_bias_aware` | −0.85 (20/20) | −0.17 (n.s.) | −1.68 (19/20) | −0.44 (n.s. ở α_4) | +0.13 (n.s.) |
| `err_reg` | −1.29 (20/20) | −0.22 (n.s.) | −1.33 (18/20) | −0.27 (n.s.) | +0.16 (p = 0.0032, n.s. ở α_4) |

- **Quy tắc EXT9 so với `full`: ĐẠT.** Tốt hơn có ý nghĩa ở 3 họ, không kém ở họ nào; hiệu lớn nhất +0.03 (HDD); lỗi bi
  quan tăng tối đa +0.72.
- Số run chọn `rpt_L`: C-MAPSS 50/80, N-CMAPSS 15/20, vòng bi 88/120, pin 18/20, HDD 4/20.
- Tổng α đã tiêu: 0.025 + 0.0125 + 0.00625 + 0.003125 = 0.046875.
