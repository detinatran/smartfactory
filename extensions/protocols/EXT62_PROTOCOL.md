# EXT62: ensemble gộp (mặc định + RAMP-V13, tức V17 của EXT61) trên seed mới 222–251, cả 3 họ (THĂM DÒ, không tiêu α; ngân sách α đã hết). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Lý do:** ở EXT61, V17 là biến thể RAMP duy nhất trên vòng bi không kém predictor mặc định ở chỉ số nào: risk lạc quan hòa (+0.14), lỗi bi quan ít hơn, RMSE tốt hơn. Tuy vậy, V17 được chọn sau nhiều biến thể thử trên cùng seed 162–171, nên phải lặp lại trên seed khác. Các seed 222–251 chưa từng được dùng để chọn biến thể RAMP; chúng chỉ được dùng ở vòng 10 với V13.

**Câu hỏi:** một predictor gộp có bao giờ kém predictor mặc định không? Cụ thể là trên cả C-MAPSS, pin và vòng bi, khi giữ ELPIS-G làm selector.

## Hệ thống (cố định)
- **Predictor:** `scripts/ext50.py` với `RAMP_FREEZE=1`, `RAMP_BIASCAL=1`, `RAMP_HYBRID=1`. Gồm 5 thành viên mặc định của run và 5 thành viên RAMP-V13.
- **Selector:** `full` (OSC) và `rpt_auto_d` (ELPIS-G), giống EXT59.
- **Run:** `conf222_cmapss` và `conf222_ext` (vòng 10), không train lại ensemble mặc định.
- **So sánh:**
  - hệ thống mặc định của EXT59 (`all_ext59_default.csv.gz`);
  - hệ thống RAMP-V13 của EXT59 (`all_ext59_ramp.csv.gz`).
  Cả hai đều trên cùng seed.

## Phân tích (đặt trước; đơn vị là seed, cách gộp giống EXT59)
- **Chính (không kém):** với mỗi họ, lấy hiệu risk của ELPIS-G trên predictor gộp trừ ELPIS-G trên predictor mặc định (coverage 90%, trung bình 6 lỗi).
  - Đạt nếu cận trên của CI 90% bootstrap theo seed (20 000 lần) ≤ +0.3 điểm ở cả 3 họ.
  - Báo cáo kèm Wilcoxon hai phía, Holm trên 3.
- **Phụ:**
  - tỷ lệ bi quan ở cùng coverage (gộp so với mặc định);
  - RMSE trên dữ liệu sạch;
  - ELPIS-G − OSC trên predictor gộp (Wilcoxon theo seed);
  - predictor gộp so với RAMP-V13.

## Diễn giải đặt trước
- Đạt không kém ở cả 3 họ: báo cáo predictor gộp là cách dùng RAMP an toàn. Nó không kém ở họ nào, nhưng chỉ là thăm dò.
- Không đạt ở họ nào: báo cáo đúng như vậy.
- Trong mọi trường hợp, KHÔNG đổi các phát biểu xác nhận của bài.

## Vận hành
Chạy 6 shard trên GPU. Nếu có run lỗi thì chạy lại; CSV đã có sẽ được bỏ qua.

## Kết quả, ngày 2026-11-17 (thăm dò): KHÔNG đạt tiêu chí không kém trên vòng bi; tốt hơn trên C-MAPSS và pin
Đủ 900/900 run, 30/30 seed ở mọi họ, không có run lỗi.

Risk của ELPIS-G dưới lỗi, predictor gộp so với predictor mặc định:

| Họ | Hiệu | Seed gộp tốt hơn | p_Holm | Cận trên CI 90% | Lỗi bi quan | RMSE (gộp so với mặc định) |
|---|---|---|---|---|---|---|
| C-MAPSS | −0.66 | 28/30 | < 1e-4 | −0.55 | −0.72 (30/30) | 14.58 so với 15.85 |
| Pin | −0.67 | 30/30 | < 1e-4 | −0.57 | −0.85 (29/30) | 0.083 so với 0.092 |
| Vòng bi | +0.35 | 15/30 | 0.61 | +1.21, **không đạt** biên 0.3 | −3.54 (26/30) | 0.258 so với 0.275 |

- **So với RAMP-V13:**
  - vòng bi −1.63 (22/30, p_Holm = 0.026);
  - pin −0.22 (p_Holm = 0.010);
  - C-MAPSS −0.06, không có ý nghĩa.
- **ELPIS-G − OSC trên predictor gộp:**
  - C-MAPSS −0.65 (30/30);
  - pin −1.08 (30/30);
  - vòng bi −0.96 (28/30);
  - p ≤ 3.5e-8.

**Diễn giải (đúng như đặt trước):** không đạt tiêu chí không kém ở cả 3 họ, vì vòng bi không đạt.
- Trên vòng bi, risk lạc quan hòa về điểm (+0.35, không có ý nghĩa), nhưng CI rộng, và lỗi bi quan ít hơn 3.5 điểm.
- Về chi phí (post hoc): với chi phí r·lạc quan + bi quan, predictor gộp rẻ hơn mặc định trên vòng bi khi r < khoảng 10.
- Predictor gộp tốt hơn RAMP-V13 trên vòng bi và pin, còn C-MAPSS hòa, và tốt hơn predictor mặc định trên C-MAPSS và pin ở cả hai loại lỗi.
- Không đổi phát biểu xác nhận nào của bài.
