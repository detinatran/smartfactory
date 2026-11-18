# EXT44: chẩn đoán D khi predictor đã được làm robust (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Lý do (review lần 3, điểm yếu 2):**
- Cổng D mới chỉ được thử trên một họ "âm" là ổ cứng.
- Nếu predictor đã được train trên chính các perturbation này (predictor robust, các run `robust72`, seed 72–81), thì perturbation có thể không còn làm predictor lạc quan. D có thể nhỏ lại, thậm chí âm, ngay trên benchmark.

## Dữ liệu
240 run có sẵn trong `runs/robust72`: predictor robust; C-MAPSS FD001–4, N-CMAPSS, vòng bi, pin; GRU và TCN; seed 72–81. Không train lại.

## Chạy
`scripts/ext38.py` giữ nguyên, tạo ra D, `full`, `rpt_L` và `rpt_auto_d`.

## Phân tích (đặt trước, thăm dò)
1. **Phân phối D theo họ:** so sánh với predictor thường (EXT37 và các vòng 7–8).
2. **Hiệu `rpt_L` − `full` và `rpt_auto_d` − `full` theo họ:** trung bình theo seed, Wilcoxon theo seed.
3. **Độ chính xác quyết định của D theo deployment:** dấu của D có khớp với dấu lợi ích của `rpt_L` không.

**Diễn giải đặt trước:**
- Nếu D vẫn dương và RPT vẫn có ích: cổng D nhất quán, nhưng chưa được thử trên trường hợp âm.
- Nếu D âm ở một phần deployment: mục 3 cho biết D có bỏ RPT đúng chỗ không.

## Kết quả, ngày 2026-11-16 (thăm dò)
D giảm nhưng vẫn dương trên C-MAPSS, N-CMAPSS và pin. RPT vẫn có ích; RPT-D trùng với RPT. Không tạo được trường hợp âm mới. Chi tiết xem FINDINGS.md.
