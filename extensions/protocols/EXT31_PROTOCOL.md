# EXT31: pin với unit test xoay vòng (5-fold theo cell trên toàn bộ 124 cell). Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do (review 10, W3):** trên pin, 40 cell test (batch 3) cố định qua mọi seed. Vì vậy p-value đo nhiễu do seed huấn luyện, chứ
không đo khả năng tổng quát hoá sang cell mới. EXT31 xoay vòng tập test để mọi cell đều được dùng làm test.

## Thiết kế
- **Dataset `battery_cv`:** cùng dữ liệu và đặc trưng như `battery`.
  - Fold = crc32(tên cell) mod 5, tức cv0–cv4, cố định qua các seed.
  - Với fold k: test là các cell thuộc fold k. Các cell còn lại chia validation / selector / calibration / fit với tỷ lệ 0.10 / 0.20 / 0.30 / phần còn lại, giống `battery`.
- **Huấn luyện:** GRU và TCN, `configs/external_battery_cv.json` (giống `external.json`, window 30, stride 10).
  - Seed mới 122–131, train trên GPU.
  - 5 fold × 10 seed × 2 backbone = 100 run.
- **Đánh giá:** `scripts/ext22.py`, không sửa gì: `full`, `rpt_L`, `rpt_auto`, `gauss_bias_aware`, `err_reg`, `ensemble_std`.

## Kiểm định (thăm dò; nằm ngoài ngân sách α; α = 0.05, Holm trên 2)
- **Đơn vị:** cặp (backbone, seed). Risk @90% matched, trung bình 6 lỗi. Lấy trung bình 5 fold theo trọng số số cell test, nên mỗi cặp đã phủ toàn bộ 124 cell.
- **Chính:** `rpt_L` − `full` và `rpt_auto` − `full`.
  - Wilcoxon hai phía trên 20 cặp.
  - Kèm sign-flip chính xác ở mức seed (10 seed, trung bình hai backbone).
- **Mô tả:**
  - `rpt_L` so với `gauss_bias_aware` và `err_reg`;
  - hiệu theo từng fold;
  - risk sạch.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-15
`rpt_L` − `full` = −1.09 (20/20 cặp, 10/10 seed), âm ở mọi fold. Chi tiết xem FINDINGS.md.
