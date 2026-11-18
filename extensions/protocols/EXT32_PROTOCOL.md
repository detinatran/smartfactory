# EXT32: N-CMAPSS với engine test xoay vòng (5-fold trên toàn bộ 99 engine). Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do (review 10, W3):** test của N-CMAPSS là các engine test chính thức, cố định qua mọi seed. EXT32 làm như EXT31 cho
N-CMAPSS để kiểm tra lợi ích có tổng quát hoá sang engine mới không.

## Thiết kế
- **Dataset `ncmapss_cv`:** cùng dữ liệu và đặc trưng như `ncmapss`.
  - Fold = crc32(unit_id) mod 5, tức cv0–cv4.
  - Với fold k: test là các engine thuộc fold k. Các engine còn lại chia validation / selector / calibration / fit với tỷ lệ 0.15 / 0.20 / 0.15 / phần còn lại, giống `ncmapss`.
- **Huấn luyện:** GRU và TCN, `configs/external_ncmapss_cv.json` (giống `external.json`, window 10, stride 1).
  - Seed mới 132–141, train trên GPU.
  - 5 fold × 10 seed × 2 backbone = 100 run.
- **Đánh giá:** `scripts/ext22.py`, không sửa gì.

## Kiểm định (thăm dò; nằm ngoài ngân sách α; α = 0.05, Holm trên 2)
- **Đơn vị:** cặp (backbone, seed); lấy trung bình 5 fold theo trọng số số engine test.
- **Chính:** `rpt_L` − `full` và `rpt_auto` − `full`.
  - Wilcoxon trên 20 cặp.
  - Kèm sign-flip chính xác ở mức seed.
- **Mô tả:** so với hai bộ có hướng; hiệu theo từng fold.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-15
`rpt_L` − `full` = −0.25 (20/20 cặp, 10/10 seed), âm ở mọi fold. Chi tiết xem FINDINGS.md.
