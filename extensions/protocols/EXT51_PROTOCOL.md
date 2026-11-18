# EXT51: đối chứng cho EXT50. Giữ nguyên kiến trúc ELPIS-J nhưng bỏ pretrain JEPA (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Câu hỏi:** lợi ích của predictor ELPIS-J trong EXT50 đến từ pretrain JEPA ở Stage I, hay chỉ đến từ kiến trúc (nhánh symbolic có cổng, điều kiện chất lượng quan sát, head fusion)?

## Thiết kế
- `scripts/ext50.py` với `EXT50_E1=0`: bỏ Stage I, các module đều khởi tạo ngẫu nhiên.
- Stage II giữ nguyên: có giám sát, loss MSE, early stopping.
- Cùng seed, cùng split, cùng các họ như EXT50: C-MAPSS, `battery_cv`, vòng bi; seed 162–171.

## Phân tích (đặt trước)
So EXT50 (có JEPA) với EXT51 (không JEPA), mức seed, Wilcoxon:
- RMSE trên dữ liệu sạch;
- tỷ lệ lạc quan dưới lỗi;
- risk của `full` và `rpt_auto_d` dưới lỗi.

Đọc kết quả:
- Nếu hai bản ngang nhau thì lợi ích đến từ kiến trúc, không phải JEPA.
- Nếu EXT50 tốt hơn thì pretrain JEPA có đóng góp riêng.

## Kết quả, ngày 2026-11-17 (thăm dò)
C-MAPSS: lợi ích chủ yếu đến từ kiến trúc. Pin: lợi ích đến từ JEPA. Vòng bi: cả kiến trúc lẫn JEPA đều gây hại. Chi tiết xem FINDINGS.md.
