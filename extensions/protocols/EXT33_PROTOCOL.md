# EXT33: lỗi bias có chủ đích (kênh predictor nhạy nhất, theo chiều lạc quan). Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do (review 9, review 10 W1):**
- Các lỗi đã dùng hầu như không làm predictor lạc quan trên vòng bi và HDD.
- Lỗi masking chỉ được phát hiện trên C-MAPSS.

EXT33 kiểm tra độ bền với lỗi trên một loại lỗi được thiết kế để chắc chắn đẩy dự đoán về phía lạc quan, ở mọi họ dữ liệu.

## Lỗi `tbias` (xác định riêng cho từng run, không dùng nhãn test)
1. Trên các window calibration sạch, tính độ nhạy của predictor theo từng kênh: Δ_c = trung bình [f(x + 1·e_c) − f(x)], trong đó e_c cộng +1 (đơn vị đã chuẩn hoá) vào kênh c trên toàn window.
2. Chọn 3 kênh có |Δ_c| lớn nhất. Bias +3σ được cộng theo dấu sign(Δ_c), tức theo chiều làm dự đoán RUL tăng.
3. Khoảng thời gian và onset giống lỗi `bias` của thư viện: cùng seed theo unit, C-MAPSS dùng chu kỳ [onset, onset + 60), các họ khác dùng 30% tuổi thọ.

## Dữ liệu và đánh giá
- **Run:** vòng 5, seed 112–121 (`conf112g_*`), 5 họ, 260 run. Không train lại.
- **Script:** `scripts/ext33.py`. Predictor, selector và `rpt_auto` giống ext22/ext29.
- **Kịch bản:** clean và `tbias`.
- **Chỉ số:** như EXT25/EXT29, gồm tỷ lệ window bị ảnh hưởng, tỷ lệ lạc quan của predictor trên window bị ảnh hưởng (sạch so với lỗi), `excess_A` và `risk_matched`.

## Kiểm định (thăm dò; nằm ngoài ngân sách α)
- **T1:** Δ`excess_A` của `rpt_auto` − `full` và `rpt_L` − `full`, trên 5 họ. Holm trên 10, α = 0.05.
- **T2:** `risk_matched` dưới `tbias`: `rpt_auto` − {`full`, `gauss_bias_aware`, `err_reg`}. Holm trên 15.
- **Mô tả:** mức tăng tỷ lệ lạc quan do lỗi trên mỗi họ, dùng để xác nhận lỗi thật sự gây lạc quan.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-15
`rpt_auto` giảm risk dư trên C-MAPSS (−1.14) và vòng bi (−1.03). Chi tiết xem FINDINGS.md.
