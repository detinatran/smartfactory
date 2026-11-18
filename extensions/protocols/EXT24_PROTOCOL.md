# EXT24: triển khai với ngưỡng cố định trước, khi tỷ lệ lỗi chưa biết. Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do:** reviewer hỏi ngưỡng triển khai được chọn thế nào khi không có nhãn của miền đích. Chứng nhận hữu hạn mẫu (LTT,
conformal risk control) đã được thử ở ltt_v1/v2: **0% chứng nhận** với 10–51 unit calibration, cần khoảng 10²–10³ unit.
Vì vậy hướng chứng nhận không được mở lại (tránh tune theo kết quả). EXT24 đo cách chọn ngưỡng plug-in thực dụng:
- ngưỡng cố định trên unit calibration;
- áp dụng cho tập test có tỷ lệ lỗi π chưa biết.

Đây là **phân tích post hoc / thăm dò**:
- không tiêu α xác nhận (ngân sách vòng 1–4 đã tiêu 0.046875);
- mọi kiểm định đều ở mức danh nghĩa 0.05, Holm trong họ, và được ghi rõ là thăm dò.

## Dữ liệu
- Run vòng 4: seed 92–101, `runs/conf92_cmapss` và `runs/conf92_ext` (5 họ, gồm HDD), tổng 260 run.
- Không train lại.
- Predictor, selector và quy tắc chọn `rpt_auto` giống hệt `scripts/ext22.py` (cùng seed, cùng dữ liệu). Kiểm tra nhất quán: `risk_matched` của `full`, `rpt_L` và `rpt_auto` dưới 6 lỗi phải khớp EXT22 (chênh ≤ 0.01 điểm).

## Điểm số
`ensemble_std`, `gauss_bias_aware`, `err_reg`, `full` (OSC), `rpt_L` (RPT), `rpt_auto`.

## Tập calibration (unit calibration; rời với fit, selector và test)
- **C0:** calibration sạch.
- **CS:** clean, dropout, drift (các shift đã thấy), trọng số bằng nhau.
- **CP:** hỗn hợp gồm (1 − π_a) calibration sạch và π_a bản perturbed của chính các window calibration đó.
  - Bản perturbed dùng cùng generator với RPT (smooth cosine, A ∈ LogU(1, 8), k ≤ round(0.5n)), nhưng seed mới là
    seed + 52000, nên không trùng các bản dùng để train selector.
  - π_a = 0.3 cố định. Đây là tỷ lệ lỗi giả định, không chỉnh theo kết quả.

Trong mỗi phần, window có trọng số engine (mỗi unit có tổng trọng số bằng nhau), chuẩn hoá về tổng 1, rồi nhân với trọng số của phần.

## Tập test
- Hỗn hợp gồm (1 − π) test sạch và π chia đều cho 6 lỗi chưa thấy.
- π ∈ {0, 0.1, 0.3, 0.5}. Khi π ≠ 0.3, giả định của CP bị sai lệch, và điều này được báo cáo.
- Coverage, tỷ lệ lỗi lạc quan (E > δ) và tỷ lệ lỗi bi quan được tính theo trọng số hỗn hợp:
  - mỗi phần dùng trọng số engine chuẩn hoá;
  - risk = Σ (nhận và lạc quan) / Σ nhận.

## Quy tắc ngưỡng (đều cố định trên calibration rồi áp dụng nguyên cho test)
1. **Mục tiêu coverage 90%:** ngưỡng = quantile 0.9 có trọng số của điểm số trên tập calibration (C0, CS hoặc CP).
   C0 chính là ngưỡng "frozen" đã báo cáo.
2. **Mục tiêu risk** (chính của EXT24):
   - r* = 0.5 × tỷ lệ lạc quan khi nhận hết trên **C0**, dùng chung cho mọi quy tắc trong cùng run.
   - Trên lưới coverage 0.30, 0.35, …, 1.00, chọn mức lớn nhất mà risk trên tập calibration ≤ r*. Ngưỡng tại mỗi mức là
     quantile có trọng số trên tập đó.
   - Nếu không mức nào đạt thì từ chối tất cả (coverage 0, không tính là vi phạm).
   - Trên test: **vi phạm** = risk test > r*; ghi kèm coverage test.

## Phân tích (thăm dò)
- **Bảng mô tả,** cho mỗi họ × điểm số × tập calibration × π:
  - tỷ lệ run vi phạm;
  - coverage test trung bình;
  - risk test trung bình.

  Mỗi deployment là một run; subset C-MAPSS và fold vòng bi được tính như từng deployment riêng.
- **D1** (điểm số; chỉ ở calibration CP, mục tiêu risk, π = 0.3): `rpt_auto` so với `full`.
  - Ở mức cặp (backbone, seed), tính hai đại lượng:
    - (a) phần vượt mục tiêu (risk − r*)⁺, trung bình trong cặp, với vòng bi trọng số theo fold như thường lệ;
    - (b) coverage.
  - Wilcoxon hai phía; Holm trên 5 họ × 2 đại lượng = 10 phép thử.
- **D2** (tập calibration; cho `rpt_auto`, mục tiêu risk, π = 0.3): CP so với C0 và CS so với C0. Cùng hai đại lượng và cách
  thống kê; Holm trên 5 × 2 × 2 = 20.
- **D3** (mô tả): tỷ lệ vi phạm của mọi điểm số khi π = 0.1 và 0.5 (giả định π_a sai lệch).

## Diễn giải đã chốt
- Nếu CP giảm vi phạm so với C0 mà coverage giảm, đó là đánh đổi; bài sẽ báo cáo cả hai đại lượng.
- Nếu vi phạm vẫn > 20% ở mọi quy tắc, bài sẽ nói thẳng rằng quy tắc ngưỡng plug-in không đủ cho một mục tiêu risk khi có lỗi chưa thấy.
- Không có kết quả nào trong EXT24 được trình bày như đảm bảo.

## EXT24b: dự đoán bảo thủ (offset) so với từ chối. Bổ sung ngày 2026-11-15, TRƯỚC khi có bất kỳ kết quả EXT24 nào
**Lý do:** reviewer hỏi vì sao phải từ chối thay vì chỉ dịch dự đoán xuống (predictor bảo thủ). STEP3 đã trả lời câu này cho
OSC trên các run cũ; EXT24b lặp lại cho phương pháp cuối, trên cùng run vòng 4 (seed 92–101, 260 run).
Script: `scripts/ext24b_offset.py`, có predictor, điểm số, tập calibration, r* và hỗn hợp test giống EXT24.

**Chính sách,** chọn trên tập calibration P ∈ {C0, CP}, rồi áp nguyên cho test:
- `offset_only`: nhận hết. Chọn b nhỏ nhất trong lưới 0, 0.05δ, …, 3δ sao cho risk trên P của dự đoán f − b ≤ r*.
  Nếu không có b nào đạt thì dùng b = 3δ.
- `joint90[m]`: ngưỡng là quantile 0.9 của m trên P; trong các window được nhận, chọn b nhỏ nhất theo cùng quy tắc.
  m ∈ {`ensemble_std`, `gauss_bias_aware`, `err_reg`, `full`, `rpt_auto`}.

**Chỉ số test,** với π ∈ {0, 0.3, 0.5}:
- coverage;
- risk lạc quan và tỷ lệ bi quan của f − b;
- tỷ số RMSE = RMSE(f − b trên các window được nhận) / RMSE(f, nhận hết, b = 0), tính so với mục tiêu đã cap;
- b được chọn;
- vi phạm (risk > r*).

**D4 (thăm dò):** ở P = CP, π = 0.3, so `joint90[rpt_auto]` với `offset_only` trên hai đại lượng là risk và tỷ số RMSE.
Wilcoxon theo cặp; Holm trên 5 họ × 2 = 10. Cũng báo cáo `joint90[rpt_auto]` so với `joint90[full]` (mô tả).

## Kết quả EXT24, ngày 2026-11-15
Xem `research/extension-results/2026-11-09/ext24/FINDINGS.md`.
- Mục tiêu risk: vi phạm 41–60% ở 4 họ benchmark với mọi tập calibration, kể cả π = 0. Theo diễn giải đã chốt, đây là kết quả âm
  về plug-in.
- D1: ở cùng mục tiêu, `rpt_auto` có coverage cao hơn `full` (C-MAPSS +7.7, vòng bi +8.9, pin +31.7, có ý nghĩa), đổi lại phần vượt
  mục tiêu trên C-MAPSS +0.55.
- D2: CP giảm nhẹ phần vượt trên C-MAPSS (−0.22) và tốn 2.7 điểm coverage. CS không an toàn trên HDD (vi phạm 100%).

## Kết quả EXT24b, ngày 2026-11-15
- D4 (CP, π = 0.3):
  - `joint90[rpt_auto]` có RMSE thấp hơn `offset_only` có ý nghĩa ở cả 5 họ.
  - Risk cao hơn ở 3 họ (+1.1 đến +3.2 điểm), vì offset_only dịch tới 0.85–2.49δ.
  - Kết luận: đánh đổi.
- Với C0 (mô tả), joint lấn át offset trên 4/5 họ.

Chi tiết xem FINDINGS.md.
