# EXT25: risk trên các window thật sự bị lỗi thay đổi, và các baseline còn thiếu. Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do:** review 7, W1 và W5.

**W1.** Lỗi được inject theo unit, trong một khoảng thời gian:
- C-MAPSS: các chu kỳ [onset, onset + 60), với onset ∈ [40, 80];
- các họ ngoài: tương tự theo `external.py`.

Vì vậy nhiều window của một unit bị lỗi thực ra không bị thay đổi. Risk tính trên mọi window bị pha loãng, và "risk dư do lỗi" bị đánh giá thấp. EXT25 đo risk chỉ trên các window có đầu vào khác với bản sạch.

**W5.** Reviewer đề nghị thêm các baseline:
- quantile regression trực tiếp của sai số có dấu;
- kiểm định cảm biến bằng PCA (SPE/Q và Hotelling T²), là chuẩn trong giám sát quá trình.

Đây là **phân tích post hoc / thăm dò** trên seed vòng 4 (92–101, 5 họ, tổng 260 run):
- không tiêu α;
- Holm trong từng họ kiểm định ở mức 0.05.

## Script: `scripts/ext25_affected.py`
Predictor, selector và `rpt_auto` giống hệt `ext22.py`. Kiểm tra nhất quán: `risk_matched` của `full`, `rpt_L` và `rpt_auto` phải khớp EXT22 (chênh ≤ 0.01 điểm).

**Baseline mới**, đều fit trên dữ liệu và đặc trưng của OSC hoặc dữ liệu fit sạch, cấu hình cố định trước:
- `eq90`: HGB quantile regression (loss quantile, α = 0.9) của E trên đặc trưng f, dữ liệu S. Điểm số là q̂₀.₉(E | x).
- `pca_spe`, `pca_t2`:
  - PCA trên từng bản ghi (vector n_ch kênh cảm biến, NaN → 0) của các window fit sạch;
  - số thành phần chiếm 95% phương sai;
  - điểm số của window là max theo các bản ghi trong window của SPE (Q) hoặc của T².

## Window bị ảnh hưởng
- Với mỗi lỗi f, window i là **bị ảnh hưởng** nếu x_f[i] khác x_clean[i] ở bất kỳ phần tử nào (NaN được coi là bằng nhau).
- Window của hai kịch bản phải trùng nhau theo (unit, chu kỳ); script assert điều này.
- Trọng số là trọng số engine trên toàn tập test, giới hạn vào tập A.

## Chỉ số (mỗi điểm số, mỗi lỗi)
- `risk_matched`: như trước, tính trên mọi window, coverage khớp 90%.
- Với cùng ngưỡng 90% trên mọi window:
  - `risk_A`: tỷ lệ lạc quan trong các window bị ảnh hưởng được nhận;
  - `cov_A`: phần window bị ảnh hưởng được nhận.
- `risk_A_clean`: risk trên chính các window đó trong kịch bản sạch, dùng ngưỡng khớp 90% của kịch bản sạch.
- `excess_A` = `risk_A` − `risk_A_clean`.
- Mô tả tác động của lỗi: tỷ lệ lạc quan của predictor khi nhận hết trên A, có lỗi và sạch.

## Phân tích (thăm dò)
Gộp như thường lệ: trung bình qua 6 lỗi; subset C-MAPSS trung bình trong cặp; fold vòng bi theo trọng số unit; Wilcoxon theo cặp hai phía.
- **F1:** `rpt_auto` và `rpt_L`, mỗi cái so với `full`, trên `risk_A` và `excess_A`. Tổng 2 × 2 × 5 = 20 phép thử, Holm trên 20.
- **F2:** `rpt_auto` so với {`ensemble_std`, `gauss_bias_aware`, `err_reg`} trên `risk_A`. Holm trên 15.
- **F3, baseline mới:** `rpt_auto` so với {`eq90`, `pca_spe`, `pca_t2`} trên `risk_matched`. Holm trên 15.
- **Mô tả:** tỷ lệ window bị ảnh hưởng, tác động của lỗi lên predictor trên A, và `cov_A`. `cov_A` thấp hơn nghĩa là selector từ chối nhiều window bị lỗi hơn.

## Diễn giải đã chốt
- Nếu `excess_A` của `rpt_auto` thấp hơn `full` có ý nghĩa ở các họ ngoài C-MAPSS, bài sẽ sửa câu "phần riêng cho lỗi chỉ có ý nghĩa trên C-MAPSS".
- Nếu không, bài giữ nguyên nhận định và nói rõ rằng kết quả không phải do pha loãng.

## Kết quả, ngày 2026-11-15
Xem `extension-results/2026-11-09/ext25/FINDINGS.md`.
- `excess_A` của `rpt_auto` thấp hơn `full` có ý nghĩa trên C-MAPSS, N-CMAPSS và pin, tức là cả ngoài C-MAPSS. Theo diễn giải đã chốt, nhận định trong bài đã được sửa.
- Baseline mới: `rpt_auto` tốt hơn có ý nghĩa ở cả 5 họ.
