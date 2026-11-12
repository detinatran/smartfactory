# EXT6-C: xác nhận `full_recon` bằng seed mới. Chốt ngày 2026-11-11, TRƯỚC khi train

## Bối cảnh
- EXT6 (`extension-results/2026-11-09/ext6_v1/FINDINGS.md`): `full_recon` tốt hơn `full` trên C-MAPSS, hiệu −0.25 điểm,
  20/20 cặp, p_Holm < 1e-4. Ba dataset còn lại hoà.
- Tập đánh giá đó đã được dùng qua 5 vòng. Vì vậy protocol EXT6 đã ghi trước rằng nếu đạt tiêu chí thì bắt buộc xác nhận
  bằng seed mới.

## Thiết lập
- **Backbone mới:** C-MAPSS FD001–FD004 × {GRU, TCN} × seed 52, 53, 54, 55, 56, tức 40 run.
  - Train bằng `scripts/run_matrix.py --config configs/ext_capped.json --heads point --device cpu`.
  - Config giống hệt các run của bài; chỉ seed khác.
- **Seed mới kéo theo split mới:** các unit fit, selector, calibration và đánh giá được chia lại theo seed. Unit đánh giá
  vẫn lấy từ tập development của C-MAPSS. Không dùng tập test chính thức.
- **Phân tích:** `scripts/ext6_repair.py` **không sửa một dòng nào**, cùng bộ tái tạo, ngưỡng và siêu tham số HGB.
- Lỗi chưa thấy, cách gộp và chỉ số: giống EXT6. Chỉ số là risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy; các
  subset được lấy trung bình trong mỗi cặp (backbone, seed).

## Giả thuyết xác nhận (một kiểm định duy nhất)
- H: `full_recon` có risk thấp hơn `full` trên C-MAPSS.
- Có 10 cặp (backbone, seed). Wilcoxon signed-rank **một phía**, α = 0.05.
  - Với 10 cặp, p một phía nhỏ nhất là 1/1024. Muốn có p < 0.05 cần khoảng ít nhất 8/10 cặp cùng chiều.
- **Xác nhận** khi: p < 0.05 **và** hiệu trung bình < 0.

## Mô tả thêm (không kiểm định)
- Hiệu theo từng subset FD001–FD004.
- Ngưỡng frozen: risk và coverage.
- Risk trên dữ liệu sạch.
- `repair_full` và `impact`, chỉ để mô tả.
- Kiểm tra lại: risk của `full` trên seed mới có cùng tầm với seed 42–51 không.

## Diễn giải đã chốt trước
- **Nếu xác nhận:** được phép viết trong bài mở rộng rằng `full_recon` cải thiện nhỏ trên C-MAPSS, kèm số liệu của cả hai
  vòng. Có thể cân nhắc đưa vào bài ESANN như một phân tích có xác nhận bằng seed mới. Quyết định đó thuộc về tác giả.
- **Nếu không xác nhận:** kết quả EXT6 trên C-MAPSS được coi là dương tính giả do thử nhiều lần, và bài giữ nguyên.
