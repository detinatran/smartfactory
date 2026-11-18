# EXT27: backbone thứ ba (Patch-Transformer) với đủ 20 cặp. Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do (review 7, W4):** kết quả với Patch-Transformer hiện chỉ có 5 cặp mỗi họ (seed 72–76), nên chỉ mang tính mô tả: Wilcoxon
với 5 cặp có p tối thiểu là 0.0625. EXT27 bổ sung seed 77–91 để đủ 20 cặp mỗi họ (seed 72–91, một backbone).

## Huấn luyện
- **Backbone:** Patch-Transformer, cấu hình giống hệt các run seed 72–76:
  - C-MAPSS: `run_matrix.py`, `configs/ext_capped.json`, head point, FD001–FD004;
  - vòng bi và pin: `run_external.py`, `configs/external.json`;
  - N-CMAPSS: `configs/external_pt_ncmapss.json` (patch 4, stride 2; sai lệch đã ghi ở EXT12).
- **Seed mới 77–91.** Run ghi vào `runs/conf77t_cmapss` và `runs/conf77t_ext`. Tổng 15 × 12 = 180 ensemble.

## Đánh giá
`scripts/ext22.py`, không sửa, trên seed 72–91 (`conf72t_*` và `conf77t_*`), cho các điểm số `full`, `rpt_L`, `rpt_auto`,
`gauss_bias_aware`, `err_reg`, `ensemble_std`.

## Kiểm định (thăm dò, post hoc; không tiêu α xác nhận)
- **B1:** `rpt_L` so với {`full`, `gauss_bias_aware`, `err_reg`} × 4 họ = 12 phép thử, Holm trên 12, α = 0.05.
  Risk @90% matched, trung bình 6 lỗi; subset C-MAPSS trung bình trong cặp; fold vòng bi theo trọng số unit.
- **B2:** `rpt_auto` so với `full` × 4 họ, Holm trên 4.
- **Mô tả:**
  - RMSE của predictor trên dữ liệu sạch (so với GRU/TCN);
  - risk dư do lỗi;
  - lỗi bi quan.
- Mọi kết quả đều được báo cáo. Phần backbone trong bài sẽ được thay bằng kết quả 20 cặp.

## Sai lệch vận hành, ghi ngày 2026-11-15, TRƯỚC khi xem kết quả
- **Lỗi:** `run_matrix.py` không cho dùng chung thư mục batch, nên 8 job C-MAPSS ban đầu dừng ngay (FileExistsError).
  - Các job này được chạy lại với thư mục batch riêng `pt_FDxxx_gK`.
  - Hai job FD001/FD002 seed 77–81 khởi động từ phiên ssh bị dừng khi phiên đóng, nên được chạy lại lần nữa trong `pt_FDxxx_g1b`.
  - Các thư mục dở dang không có `complete.json` nên không được đánh giá.
- **Phạm vi:** cấu hình, seed và phương pháp không đổi. 10 run này được đánh giá bổ sung bằng cùng `ext22.py` vào cùng thư mục kết quả.

## Kết quả, ngày 2026-11-15
`rpt_L` tốt hơn `full` có ý nghĩa ở cả 4 họ, và tốt hơn hai bộ có hướng ở 6/8 phép so sánh (N-CMAPSS: p = 0.072).
`rpt_auto` tốt hơn `full` ở cả 4 họ. Chi tiết xem FINDINGS.md.
