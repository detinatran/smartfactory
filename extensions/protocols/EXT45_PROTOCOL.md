# EXT45: predictor bảo thủ, thử cổng D trên các trường hợp âm ngay trong benchmark (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Lý do:** EXT44 không làm D đổi dấu trên benchmark. Trong thực tế, người ta hay dịch dự đoán của predictor xuống cho bảo thủ hơn (offset). Khi đó lỗi lạc quan hiếm đi, và perturbation có thể không còn tạo ra lỗi lạc quan, tức D ≤ 0.

## Thiết lập
- Dùng lại các run vòng 8, seed 162–166: C-MAPSS, `ncmapss_cv`, vòng bi, `battery_cv`, `hdd`; GRU và TCN.
- Predictor bảo thủ: mọi dự đoán của ensemble trừ đi k·δ, với k ∈ {1, 2}.
- Offset áp dụng ở mọi bước: nhãn huấn luyện selector, chẩn đoán D, calibration và đánh giá.
- Code: `scripts/ext45.py`, là `ext38.py` cộng thêm offset.

## Phân tích (đặt trước, thăm dò)
Với mỗi k:
1. Phân phối D theo họ; tỷ lệ D > 0.
2. Hiệu `rpt_L` − `full` và `rpt_auto_d` − `full`, trung bình theo seed.
3. Độ chính xác quyết định của D theo deployment.
4. Câu hỏi chính: ở những deployment có D ≤ 0, RPT có còn giúp không? Hiệu `rpt_L` − `full` được tách theo dấu của D.

**Diễn giải đặt trước:** nếu ở các deployment D ≤ 0 mà RPT không giúp hoặc gây hại, còn ở các deployment D > 0 thì RPT giúp, thì cổng D được ủng hộ trên benchmark.

## Kết quả, ngày 2026-11-16 (thăm dò)
D vẫn dương trên benchmark ở cả k = 1 và k = 2. Ở D ≤ 0, RPT gây hại chỉ trên HDD (tập selector lớn); trên vòng bi (tập nhỏ), RPT vẫn có ích. Chi tiết xem FINDINGS.md.
