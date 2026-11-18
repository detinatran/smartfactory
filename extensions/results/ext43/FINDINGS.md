# EXT43: đường cong học theo số unit selector trên các họ benchmark (thăm dò, các run vòng 8, seed 162–171)

Hiệu `rpt_L` − `full` (điểm), lấy trung bình trên 10 seed. Risk đo ở coverage 90% (matched), trung bình 6 loại lỗi. Lấy mẫu con số unit selector xuống 1/4, 1/2, hoặc dùng toàn bộ.

| Họ | 1/4 | 1/2 | Toàn bộ | Ghi chú |
|---|---|---|---|---|
| C-MAPSS | −0.80 (7 unit) | −0.78 (14) | −0.61 (27) | 10/10 seed ở mọi ô |
| N-CMAPSS | −1.62 (4) | −0.63 (8) | −0.32 (16) | 10/10 seed ở mọi ô |
| Pin | −2.37 (5) | −1.79 (10) | −0.93 (20) | 10/10 seed ở mọi ô |
| Vòng bi | −1.32 (2) | −3.05 (3) | −1.11 (5) | không đơn điệu, 9/10 seed |

- **So với giả thuyết:** đúng trên C-MAPSS, N-CMAPSS và pin: lợi ích của RPT tăng khi số unit selector giảm. Vòng bi không đơn điệu, nhưng lợi ích vẫn lớn ở mọi mức.
- **Ghép với EXT37 (HDD):** lợi ích ở 15 unit là −0.80, ở 30 unit −0.20, ở 60 unit +0.07, ở 120 unit +0.34. Tất cả nằm trên cùng một đường dốc lên theo số unit.
- **Hình:** `paper-acm-rpt/v3/figures/learning_curve.pdf`, tạo bằng `scripts/ext43_figure.py`.
- **Bài:** bản đầy đủ có thêm Hình `fig:lc` và đoạn văn trong mục HDD.
