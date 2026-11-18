# EXT28: khoảng tin cậy bootstrap phân tầng (cặp → unit) cho các hiệu chính. Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do (review 7, W7):**
- 20 cặp (backbone, seed) không độc lập, vì hai backbone của cùng một seed dùng chung cách chia unit;
- Wilcoxon theo cặp bỏ qua việc lấy mẫu unit.

EXT28 bổ sung khoảng tin cậy 95% bằng bootstrap phân tầng, tính đến cả hai nguồn biến thiên. Đây là phân tích bổ sung; các kiểm
định xác nhận đã có giữ nguyên.

## Dữ liệu
- Vòng 2 (seed 72–81, `conf72_*`): hiệu `rpt_L` − `full`.
- Vòng 4 (seed 92–101, `conf92_*`, gồm HDD): hiệu `rpt_auto` − `full`.
- Script `scripts/ext28_units.py`. Với mỗi run, kịch bản và điểm số, script ghi theo từng unit test:
  - a_u = trọng số engine được nhận;
  - o_u = trọng số được nhận và lạc quan.

  Ngưỡng là ngưỡng khớp 90% trên toàn bộ test, đúng như thước đo chính. Predictor, selector và `rpt_auto` giống `ext22.py`.

## Bootstrap (B = 4000, seed 2828)
1. Lấy mẫu có hoàn lại các **seed** (10 seed). Cả hai backbone của một seed đi cùng nhau, vì chúng dùng chung cách chia unit.
2. Trong mỗi run đã chọn, lấy mẫu có hoàn lại các **unit test**. Cùng một tập unit được dùng cho mọi điểm số và mọi kịch bản của run đó,
   để giữ tính ghép cặp.
3. Risk của run và kịch bản = Σo / Σa trên các unit đã lấy. Sau đó gộp như thước đo chính:
   - trung bình 6 lỗi;
   - subset C-MAPSS trung bình trong cặp;
   - fold vòng bi theo trọng số unit.

   Hiệu = trung bình qua các cặp đã lấy.
4. Khoảng tin cậy percentile 2.5–97.5%. Báo cáo kèm tỷ lệ bootstrap có hiệu ≥ 0.

**Hạn chế đã biết:** ngưỡng không được fit lại trên mỗi mẫu bootstrap (xấp xỉ có điều kiện theo ngưỡng).

## Diễn giải
- Nếu khoảng tin cậy loại trừ 0 ở các họ mà Wilcoxon đã có ý nghĩa, bài sẽ ghi rằng kết luận vững trước biến thiên do lấy mẫu unit.
- Nếu không, bài sẽ nêu rõ những họ nào kết luận phụ thuộc vào cách tính.

## Kết quả, ngày 2026-11-15
CI loại trừ 0 ở cả 4 họ benchmark, ở cả vòng 2 và vòng 4. HDD: [−0.01, +0.09]. Chi tiết xem `extension-results/2026-11-09/ext28/FINDINGS.md`.
