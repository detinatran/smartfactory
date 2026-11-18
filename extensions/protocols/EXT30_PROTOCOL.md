# EXT30: chứng nhận ngưỡng hữu hạn mẫu (LTT) trên HDD, nơi có đủ unit calibration. Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do:**
- Review 8 (W8) và review 9 đề nghị làm ngưỡng có đảm bảo.
- ltt_v1/v2 không chứng nhận được gì với 3–51 unit calibration; ước tính cần khoảng 10²–10³ unit.
- HDD có 222 unit calibration mỗi run (hdd2 có 120), nên lần đầu tiên chứng nhận có thể khả thi.
- Phân tích thăm dò, không tiêu α.

## Dữ liệu
60 run HDD:
- seed 92–101 (`conf92_ext`, 4 model gốc);
- seed 102–111 (`hdd2_ext26`, 4 model chưa dùng);
- seed 112–121 (`conf112g_ext`, 4 model gốc).

Không train lại.

## Phương pháp (giữ nguyên ltt_v2)
- **Điểm số:** `full`, `rpt_auto`, `gauss_bias_aware`, `err_reg`, `ensemble_std`. Predictor, selector và `rpt_auto` giống ext22.py.
- **Lưới mức coverage:** 0.05, 0.10, …, 1.00. Ngưỡng tại mỗi mức = quantile có trọng số unit của điểm số calibration (sạch).
- **Loss theo unit:** L_u = b_u − r* · a_u, trong đó a_u là phần window được nhận của unit u, b_u là phần được nhận và lạc quan.
- **Fixed-sequence:** tăng dần mức coverage, dừng ở lần không chứng nhận đầu tiên, với δ = 0.1. Nếu mức 0.05 không đạt thì từ chối tất cả.
- **Biên:** Hoeffding, Bernstein (Maurer–Pontil), WSR betting và plug-in (đối chứng). Hàm lấy từ `scripts/ltt_v2.py`.
- **Mục tiêu:** r* = f × tỷ lệ lạc quan khi nhận hết trên calibration sạch, f ∈ {1.0, 0.85, 0.7}.

## Đánh giá trên test
- Kịch bản: clean (exchangeable với calibration, nên có đảm bảo) và 6 lỗi (không exchangeable, nên không có đảm bảo).
- Chỉ số: mức được chứng nhận, coverage test, risk test, vi phạm (risk > r*).

## Báo cáo (mô tả)
- **Tỷ lệ chứng nhận được:** mức ≥ 0.05, theo biên × f × điểm số.
- **Vi phạm trên test sạch:** tỷ lệ run vi phạm. Đảm bảo yêu cầu ≤ δ = 0.1 cho xác suất vi phạm; ước lượng qua 20 run mỗi bộ dữ liệu.
- Vi phạm và coverage dưới lỗi.
- **So sánh thăm dò:** coverage được chứng nhận của `rpt_auto` so với `full` (Wilcoxon theo cặp).

## Kết quả, ngày 2026-11-15
WSR chứng nhận f = 1.0 ở coverage 0.95 trong 100% run có 222 unit, không vi phạm. f = 0.85 chỉ đạt 10%, f = 0.7 đạt 0%. hdd2 (120 unit) không chứng nhận được gì. Chi tiết xem FINDINGS.md.
