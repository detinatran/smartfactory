# EXT39: kiểm tra cơ chế bằng đặc trưng độ nhạy của predictor (THĂM DÒ, không tiêu α). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Câu hỏi (review lần 12, điểm yếu 4):** RPT có lợi vì selector học được độ nhạy của predictor f với nhiễu trên từng kênh không? Nếu đưa thẳng độ nhạy đó vào làm đặc trưng cho OSC, không thêm bản nhiễu nào, có đạt mức của RPT không?

## Selector mới (tính trong `scripts/ext39.py`; mọi thứ khác giống EXT38)
**Đặc trưng độ nhạy:** với mỗi kênh c và mỗi dấu s ∈ {+4, −4} (đơn vị đã chuẩn hoá), cộng s vào kênh c trên toàn window, rồi ghi Δ = f̄(x + s·e_c) − f̄(x). Thêm hai đặc trưng tổng hợp: max Δ và mean |Δ|. Tổng cộng 2·n_ch + 2 đặc trưng.

Hai selector mới:
- **`sens`**: OSC (HGB, cùng siêu tham số, cùng dữ liệu S gồm clean/dropout/drift) dùng thêm đặc trưng độ nhạy. KHÔNG có bản nhiễu.
- **`rpt_sens`**: `rpt_L` dùng thêm đặc trưng độ nhạy.

## Dữ liệu
Dùng lại các run vòng 7 (seed 152–161: C-MAPSS FD001–4, N-CMAPSS, vòng bi, pin, hdd4; GRU và TCN). Không train lại predictor. Đây là phân tích cơ chế trên seed đã dùng, nên chỉ mang tính thăm dò.

## Phân tích (đặt trước)
Metric: risk @90% matched, trung bình 6 lỗi, gộp theo cặp (model, seed) như EXT38. Wilcoxon hai phía, Holm trên 5 họ cho mỗi so sánh. Báo cáo ở mức 0.05, ghi rõ là thăm dò.

- **M1:** `sens` − `full`. Riêng độ nhạy có giúp không?
- **M2:** `rpt_L` − `sens`. Diễn giải đặt trước:
  - |hiệu trung bình| ≤ 0.15 điểm ở ≥ 3/4 họ benchmark, hoặc `sens` tốt hơn: lợi ích của RPT chủ yếu đến từ độ nhạy cục bộ, coi như đã xác định được cơ chế.
  - `rpt_L` tốt hơn có ý nghĩa ở ≥ 2 họ: bản nhiễu mang thông tin vượt ra ngoài độ nhạy tuyến tính theo từng kênh (nhiễu nhiều kênh, biên độ lớn, có onset).
- **M3:** `rpt_sens` − `rpt_L`. Hai cách có bổ sung cho nhau không?

**Mô tả thêm:**
- Lỗi lạc quan bị loại có tương quan với max Δ trên window bị lỗi không.
- Tỷ lệ khoảng cách tới oracle được thu hẹp (`ext39_oracle_gap.py`).

## Không làm
- Không thay đổi phương pháp được khuyến nghị (RPT-D) dựa trên kết quả này.
- Nếu `sens` tốt hơn, chỉ ghi nhận là thăm dò. Muốn khẳng định thì cần một vòng xác nhận mới trên seed mới, ở α_8 = 0.000195.

## Kết quả, ngày 2026-11-16 (thăm dò)
M1: sens không giúp (−0.11 đến +0.20). M2: rpt_L tốt hơn sens 0.53–0.74 điểm trên cả 4 họ benchmark (p_Holm ≤ 0.034). M3: không bổ sung cho nhau. Kết luận: RPT không thể quy về đặc trưng độ nhạy cục bộ. Chi tiết xem FINDINGS.md.
