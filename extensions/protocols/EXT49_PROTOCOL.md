# EXT49: tách hai yếu tố "cỡ tập selector" và "dấu của D" trên ổ cứng bằng generator có hướng (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Lý do (review lần 4, W4):**
- Trên benchmark, RPT chỉ gây hại khi có nhiều unit selector. Nhưng vùng có nhiều unit selector chỉ có ở ổ cứng, mà ở đó D < 0.
- Vì vậy chưa biết tác hại đến từ cỡ dữ liệu hay từ việc nhiễu không tạo ra lỗi lạc quan.

## Thiết kế
**Generator có hướng (`rpt_opt`):**
- Dùng đúng nhiễu của RPT.
- Với mỗi kênh, đặt dấu nhiễu theo hướng làm tăng RUL dự đoán. Hướng này ước lượng bằng sai phân hữu hạn +4 trên window selector sạch.
- Nhiễu có hướng như vậy được thiết kế để gây lạc quan, tức D_opt > 0.
- D_opt dùng cùng generator này trên window calibration sạch.

**Dữ liệu:** các run `hdd` của vòng 8 (seed 162–191, GRU và TCN, 60 run).

**So sánh:** `rpt_opt`, `rpt_L` và `full`, với 15 unit, 60 unit và toàn bộ 296 unit selector. Risk ở coverage 90% matched, trung bình 6 lỗi.

## Diễn giải đặt trước
- Nếu D_opt > 0 mà `rpt_opt` vẫn kém `full` khi dùng toàn bộ unit: tác hại do cỡ dữ liệu. Ủng hộ cách hiểu "bộ điều chuẩn".
- Nếu `rpt_opt` tốt hơn hoặc bằng `full` khi dùng toàn bộ unit: tác hại ở ổ cứng chủ yếu vì D < 0. Ủng hộ cổng D.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-16 (thăm dò)
D_opt > 0 ở mọi run, nhưng RPT có hướng vẫn gây hại khi dùng toàn bộ unit (+0.42, 0/30 seed) và có ích khi chỉ có 15 unit (−0.65). Tác hại đến từ cỡ dữ liệu, không phải từ dấu của D. Chi tiết xem FINDINGS.md.
