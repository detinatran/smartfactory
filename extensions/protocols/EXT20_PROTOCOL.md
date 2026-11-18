# EXT20: tái lập độc lập `rpt_L` trên seed mới 82–91 (vòng xác nhận 3). Chốt ngày 2026-11-14, TRƯỚC khi đánh giá

- EXT18 không mở vòng 3, nên 240 backbone đã train cho seed 82–91 chưa từng được đánh giá. EXT20 dùng chúng để **tái lập**
  phương pháp cuối `rpt_L` (không đổi gì).
- Code: `scripts/ext12_main_v2.py` (giống hệt vòng 2, không có stress), chạy trên `runs\conf82_cmapss` và `runs\conf82_ext`.
- **Họ kiểm định:** `rpt_L` so với {`full`, `gauss_bias_aware`, `err_reg`} × 4 dataset, tức 12 phép thử. Wilcoxon hai phía,
  Holm ở mức **α_3 = 0.00625**. Tổng α sau vòng này là 0.04375.
- **Thành công:** quy tắc EXT9 so với `full`, giống vòng 2. Báo cáo thêm số dataset mà `rpt_L` tốt hơn cả hai bộ ước lượng
  có hướng.
- Lệnh: `python scripts/ext12_summary.py confirm all_ext20.csv.gz rpt_L`. Mức α trong script là 0.0125, nên mức 0.00625 được
  áp dụng thủ công lên cột `p_holm` khi đọc kết quả.

## Kết quả ngày 2026-11-14: XÁC NHẬN (tái lập)
240 run, 0 lỗi. Holm trên 12, α_3 = 0.00625.

| `rpt_L` − … | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| `full` | −0.70 (20/20, 2e-5) | −0.34 (20/20, 2e-5) | −1.56 (19/20, 3e-5) | −0.73 (19/20, 3e-5) |
| `gauss_bias_aware` | −1.24 (20/20, 2e-5) | −0.27 (16/20, 0.002) | −1.73 (18/20, 6e-5) | −0.41 (15/20, 0.1, n.s.) |
| `err_reg` | −1.82 (20/20, 2e-5) | −0.28 (16/20, 0.002) | −1.63 (18/20, 7e-5) | −0.45 (15/20, 0.1, n.s.) |

- Quy tắc EXT9 so với `full`: đạt (lỗi bi quan tăng tối đa +0.74).
- Tốt hơn có ý nghĩa so với cả hai bộ có hướng ở 3/4 dataset (không đạt ở pin).
- Tổng α đã tiêu: 0.025 + 0.0125 + 0.00625 = 0.04375.
