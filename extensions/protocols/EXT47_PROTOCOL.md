# EXT47: RPT-D với predictor C-MAPSS mạnh hơn (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi train

**Lý do:** reviewer chê predictor yếu (RMSE trên mọi window khoảng 15.7 cycle) và lo rằng lợi ích sẽ teo lại với một backbone mạnh hơn.

## Thiết lập
- Config `configs/ext_strong.json`: giống `ext_capped.json`, chỉ đổi hidden 128 (trước 64), 3 lớp (trước 2), tối đa 100 epoch (trước 40), patience 15 (trước 8).
- GRU và TCN; C-MAPSS FD001–4; seed 162–171 (giống vòng 8, nên so được theo cặp với predictor mặc định).
- Train trên GPU, tối đa 8 tiến trình cùng lúc.

## Phân tích (đặt trước)
1. RMSE của predictor trên mọi window đánh giá, mạnh so với mặc định (cùng seed).
2. Risk dưới 6 lỗi ở coverage 90% (matched): `rpt_auto_d` − `full`, `rpt_L` − `full`, và so với `gauss_bias_aware`, `err_reg`. Trung bình theo seed; Wilcoxon theo seed.
3. So với kết quả của predictor mặc định trên cùng seed (vòng 8): lợi ích có giữ được không.

Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-16 (thăm dò)
RMSE 15.55 → 15.39. Lợi ích RPT-D − OSC không đổi: −0.61 → −0.62, 10/10 seed. Chi tiết xem FINDINGS.md.
