# Nhật ký tìm phương pháp (bắt đầu từ EXT9, theo EXT9_PROTOCOL.md)

Các vòng trước khi có quy tắc này đều khám phá hoặc xác nhận riêng lẻ. Không vòng nào vượt `full` trên cả 4 dataset.
- EXT2–EXT5;
- EXT6 và EXT6-C (`full_recon`: xác nhận ở C-MAPSS);
- EXT7 (`stack`: đạt tiêu chí cũ, nhưng hại vòng bi);
- EXT8 cùng holdout seed 57–61 (`rank_recon`: không xác nhận).

| Vòng | Phương pháp chính | Seed xác nhận | α | Kết quả |
|---|---|---|---|---|
| — (EXT9) | cổng dư thừa (`rg_*`) | không dùng | 0 | Giả thuyết bị bác bỏ ngay ở dữ liệu train (ρ ngược chiều), không xác nhận gì |
| 1 (EXT10) | `full_recon_rpt` (selector học thêm trên nhiễu loạn ngẫu nhiên) | 62–71 | 0.025 | **XÁC NHẬN**: tốt hơn có ý nghĩa ở cả 4 dataset (p Holm ≤ 0.0032). Dừng vòng lặp theo quy tắc 6 |

Tổng số vòng xác nhận đã chạy theo quy tắc này: **1** (α đã tiêu: 0.025). EXT9 bị bác bỏ trước bước xác nhận.
