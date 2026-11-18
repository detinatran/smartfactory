# EXT43: đường cong học theo số unit selector trên các họ benchmark (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Lý do (review lần 13, điểm yếu 1):** bài đang định khung RPT là bộ điều chuẩn cho dữ liệu selector ít. Trên HDD (EXT37), RPT có ích khi có 15 unit selector và gây hại khi có từ 120 unit trở lên. Các họ benchmark chỉ có 5–39 unit selector. Câu hỏi: nếu giảm số unit selector trên benchmark xuống, lợi ích của RPT có tăng không?

## Dữ liệu
Dùng lại các run vòng 8, seed 162–171: C-MAPSS FD001–4, `ncmapss_cv`, vòng bi, `battery_cv`; GRU và TCN. Tổng 400 run. Không train lại predictor.

## Script `scripts/ext43.py` (dựa trên `ext37.py`)
Với mỗi run:
- Lấy mẫu con n = max(2, round(f·N)) unit selector, với f ∈ {1/4, 1/2} và N là số unit selector của run (rng seed + n).
- Train lại `full` và `rpt_L` chỉ trên các unit đó.
- Đo risk test ở coverage 90% (matched), trung bình 6 loại lỗi theo EVAL[1:], giống EXT37 (bản đầu ghi nhầm là 5).
- Ghi thêm D và kết quả với toàn bộ unit.

## Phân tích (đặt trước)
- **L2:** hiệu `rpt_L` − `full` theo f (1/4, 1/2, 1) cho từng họ. Trung bình theo seed; Wilcoxon theo seed tại từng f.
- **Giả thuyết:** hiệu càng âm khi f càng nhỏ, tức lợi ích của RPT tăng khi unit selector ít đi.
- **Gộp với EXT37:** vẽ một hình hiệu theo số unit selector tuyệt đối, cho mọi họ.
- Báo cáo mọi kết quả, kể cả khi trái giả thuyết.

## Kết quả, ngày 2026-11-16 (thăm dò)
Đúng giả thuyết trên 3/4 họ: lợi ích của RPT tăng khi số unit selector giảm. Vòng bi không đơn điệu. Chi tiết xem FINDINGS.md.
