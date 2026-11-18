# EXT36: đo hai thành phần của cận chuyển giao (Mệnh đề 2). Chốt ngày 2026-11-15, TRƯỚC khi chạy. Thăm dò

**Lý do (review 9 W7, review 10 W8):** bài viết rằng RPT "chủ yếu giảm sai số ước lượng, không phải tăng độ phủ", nhưng nhận
định này chưa được đo trực tiếp. EXT36 đo cả hai thành phần trong không gian đặc trưng φ của selector.

## Dữ liệu
- Run vòng 4: seed 92–101, 4 họ benchmark, 240 run.
- Script `scripts/ext36_bound.py`. Selector `full` và `rpt_L` giống ext22.py.

## Đại lượng đo (mỗi run)
- **Sai số ước lượng trên P.** Brier score của η̂ so với nhãn 1{f(x) − y > δ} trên các window calibration dưới S (clean, dropout, drift). Đây là các unit không dùng để train selector; Brier score bằng ‖η̂ − η‖² cộng phần nhiễu không giảm được. Đo cho `full` và `rpt_L`.
- **Sai số trên Q.** Brier score trên các window test dưới 6 lỗi.
- **Độ phủ.**
  - Train một HGB phân biệt miền: đặc trưng φ của dữ liệu train selector (P_OSC = S; P_RPT = S ∪ hai bản perturbed) so với φ của window test có lỗi (Q). Dùng 2-fold cross-fit theo unit.
  - Báo cáo AUC.
  - Ước lượng χ²(Q‖P) = E_P[w²] − 1, với w = (p / (1 − p)) · (n_P / n_Q), p là xác suất dự đoán của classifier, clip ở 100.

## Giả thuyết ghi trước (thăm dò, Wilcoxon theo cặp, Holm trong từng họ giả thuyết trên 4 họ dữ liệu)
- **H-est:** Brier_P(`rpt_L`) < Brier_P(`full`).
- **H-Q:** Brier_Q(`rpt_L`) < Brier_Q(`full`).
- **H-cov:** χ²(Q‖P_RPT) < χ²(Q‖P_OSC) và AUC(P_RPT, Q) < AUC(P_OSC, Q).

## Diễn giải đã chốt
- Nếu H-est và H-Q đúng nhưng hiệu của H-cov nhỏ hoặc không có ý nghĩa: bài giữ nhận định "chủ yếu giảm sai số ước lượng", có số liệu kèm theo.
- Nếu H-cov cũng mạnh: bài sửa thành "cả hai thành phần".

## Kết quả, ngày 2026-11-15
Brier score giảm trên P ở 3 họ và trên Q ở 3 họ. Bằng chứng về độ phủ không nhất quán. Chi tiết xem FINDINGS.md.
