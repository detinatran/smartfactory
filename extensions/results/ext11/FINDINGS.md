# EXT11: ablation, đổi bộ phân loại, so sánh SOTA (seed 62–71, 240 run, 0 lỗi)

- Protocol: `selective-rul/EXT11_PROTOCOL.md`, chốt trước khi chạy.
- Code: `scripts/ext11_ablate.py`, `scripts/ext11_sota.py`, `scripts/ext11_summary.py`.
- Số liệu đầy đủ: `summary_output.txt`, `family{A,B,C}_*.csv`.

**Quy ước:**
- Risk = tỷ lệ lỗi lạc quan (%) @90% matched coverage, trung bình 6 lỗi chưa thấy; 20 cặp mỗi dataset.
- Hiệu tính bằng điểm phần trăm, kèm (số cặp thắng / 20, p sau Holm).
- EXT11 chạy **sau** góp ý của leader, trên các seed đã dùng một lần cho kiểm định xác nhận EXT10.
- Phương pháp chính không đổi.

**Kiểm tra nhất quán:** `full`, `full_recon`, `full_rpt` và `full_recon_rpt` tính lại khớp EXT10 chính xác (sai lệch 0.0).

## A. So với SOTA (họ chính, Holm trên 44 phép thử, α = 0.05)

| Baseline | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Ensemble spread | −1.95 (20, 8e-5) | −7.31 (20, 8e-5) | −1.80 (18, 4e-4) | −1.77 (20, 8e-5) |
| MC dropout | −2.27 (20, 8e-5) | −7.78 (20, 8e-5) | −1.39 (17, 0.007) | −2.18 (20, 8e-5) |
| Deep ensemble NLL | −2.04 (20, 8e-5) | −6.52 (20, 8e-5) | −3.12 (20, 8e-5) | −1.06 (20, 8e-5) |
| Evidential (Amini 2020) | −2.31 (20, 8e-5) | −6.59 (20, 8e-5) | −3.17 (19, 8e-5) | −1.24 (19, 2e-4) |
| SelectiveNet | −3.40 (20, 8e-5) | −7.25 (20, 8e-5) | −3.66 (20, 8e-5) | −1.73 (20, 8e-5) |
| CQR lower | −1.10 (20, 8e-5) | −4.94 (20, 8e-5) | −0.54 (16, n.s.) | −1.33 (19, 4e-4) |
| Phân loại đối xứng \|E\|>δ | −3.37 (20, 8e-5) | −5.05 (20, 8e-5) | −3.33 (19, 8e-5) | −1.15 (19, 3e-4) |
| Bộ phát hiện lỗi | −2.08 (20, 8e-5) | −6.39 (20, 8e-5) | −2.47 (19, 8e-5) | −2.12 (20, 8e-5) |
| Hồi quy sai số (`err_reg`) | −1.50 (20, 8e-5) | −0.01 (n.s.) | −0.17 (n.s.) | +0.05 (n.s.) |
| Gaussian có bias | −1.03 (20, 8e-5) | −0.02 (n.s.) | −0.56 (n.s.) | 0.00 (n.s.) |
| Augmentation nhiễu trắng trên k kênh (`ours_white`) | −0.03 (n.s.) | +0.01 (n.s.) | −0.21 (n.s.) | −0.11 (n.s.) |

**Kết luận:**
- OURS tốt hơn có ý nghĩa so với mọi baseline định lượng bất định và selective-prediction (8/8), ở 31/32 cặp
  (baseline × dataset). Ngoại lệ là CQR ở vòng bi.
- So với các bộ ước lượng **có hướng** (hồi quy sai số, Gaussian có bias), OURS chỉ tốt hơn có ý nghĩa ở C-MAPSS.
- Với ngưỡng frozen ở pin, hai baseline này tốt hơn OURS: risk 1.55 so với 2.73 ở cùng coverage khoảng 78%.
- OURS **không** tốt hơn augmentation nhiễu trắng trên k kênh có cùng phân phối k, A và onset.

## B. Ablation (thăm dò, Holm trên 44 phép thử; số là biến thể − OURS)
- **H-B1 đúng:** bản sao sạch (`ours_copy`) kém OURS ở cả 4 dataset, +0.26 / +0.28 / +0.40 / +0.40. Kết quả có ý nghĩa ở
  C-MAPSS (p = 0.009), N-CMAPSS (2e-4) và pin (0.015). Lợi ích đến từ nhiễu loạn, không phải từ việc thêm dữ liệu hay
  đổi trọng số.
- **Thành phần hằng (m = 0) không cần:** `ours_nobias` cho −0.05 / +0.01 / +0.09 / +0.02, đều không có ý nghĩa. Lợi ích
  ở lỗi bias vẫn còn khi bỏ m = 0 (C-MAPSS −1.15, pin −1.45 so với `full`). Lo ngại "chồng lấn với bias" vì vậy không
  giải thích được lợi ích.
- **H-B2:** trung bình 5 lỗi không phải bias, so với `full_recon`, `ours_nobias` cho −0.26 / −0.29 / −0.29 / −0.13.
- **Không có onset:** không ảnh hưởng.
- **Biên độ là yếu tố quan trọng:**
  - A ∈ [0.25, 1] kém hơn: +0.21 / +0.07 / +0.47 / +0.27, có ý nghĩa ở C-MAPSS và pin.
  - Augmentation nhiễu kinh điển nhỏ trên mọi kênh (`ours_gauss_all`) kém hơn: +0.23 / +0.07 / +0.12 / +0.34, có ý nghĩa
    ở C-MAPSS và pin.
  - A ∈ [2, 8] hơi tốt hơn (−0.15 / −0.05 / −0.38 / −0.04), không có ý nghĩa.
- **Số kênh:** k = 1 hơi kém ở N-CMAPSS (+0.08, p = 0.02); k tới 50% số kênh hơi tốt hơn, không có ý nghĩa.
- **Hai bản nhiễu:** −0.15 / −0.05 / +0.03 / −0.09. Có ý nghĩa ở C-MAPSS (p = 0.009).
- **Dạng hàm:** nhiễu trắng (`ours_white`) ngang hàm trơn, không khác biệt có ý nghĩa ở dataset nào.
- **Bỏ dropout và drift:** `ours_noseen` (chỉ clean + rand) ngang OURS. RPT thay được các shift thiết kế tay.
- **Oracle** (train thẳng trên 6 lỗi đánh giá): chỉ tốt hơn OURS −0.16 / −0.08 ở C-MAPSS và N-CMAPSS; kém hơn ở vòng bi
  (+0.16) và pin (+0.03). RPT gần như đạt cận trên.
- **Ma trận 2×2 (so với `full`):**

  | | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
  |---|---|---|---|---|
  | + tái tạo | −0.24 | +0.14 | −0.11 | +0.04 |
  | + RPT | −0.25 | −0.20 | −0.57 | −0.41 |
  | + cả hai | −0.52 | −0.14 | −0.55 | −0.32 |

**Kết luận B:**
- Thành phần cốt lõi là **nhiễu loạn biên độ lớn trên một số ít kênh**, áp lên dữ liệu huấn luyện selector.
- Dạng hàm trơn, thành phần hằng và onset đều không cần thiết.
- Đặc trưng tái tạo chỉ có ích ở C-MAPSS.
- Bài báo nên trình bày lại phương pháp theo hướng này. Hiện mô tả "random smooth functions" là quá cụ thể.

## C. Đổi bộ phân loại (thăm dò, Holm trên 32 phép thử)

| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| OSC: LR − HGB | +1.10 (6e-5) | +2.45 (6e-5) | +0.87 (0.001) | +0.15 (n.s.) |
| OSC: RF − HGB | +0.70 (6e-5) | +1.94 (6e-5) | +0.44 (n.s.) | +0.37 (n.s.) |
| OSC: MLP − HGB | +1.65 (6e-5) | +2.54 (6e-5) | +1.23 (0.003) | +0.81 (0.01) |
| OSC: HGB mặc định − HGB | +0.06 | 0.00 | +0.01 | +0.19 (đều n.s.) |
| RPT với LR | −0.40 (8e-5) | −1.45 (6e-5) | −1.33 (3e-4) | −0.36 (0.007) |
| RPT với RF | −0.72 (6e-5) | −0.92 (8e-5) | −0.16 (n.s.) | −0.17 (n.s.) |
| RPT với MLP | −0.73 (6e-5) | −1.29 (6e-5) | −0.43 (n.s.) | −0.60 (0.001) |
| RPT với HGB mặc định | −0.62 (6e-5) | −0.17 (0.008) | −0.30 (n.s.) | −0.53 (0.01) |

**Kết luận C:**
- HGB tốt hơn LR, RF và MLP. Không có bộ nào tốt hơn HGB ở dataset nào.
- Kết quả không nhạy với siêu tham số của HGB.
- Lợi ích của RPT xuất hiện với cả 4 bộ phân loại, ở 16/16 ô, trong đó 11 ô có ý nghĩa.

## Còn thiếu hoặc lưu ý
- Chưa so với phương pháp RUL chịu lỗi cảm biến chuyên biệt (ví dụ Wang 2025); chưa có dữ liệu lỗi thực tế.
- Mọi kết quả B và C là thăm dò, chạy trên seed đã dùng một lần.
- Muốn đổi phương pháp chính sang biến thể đơn giản hơn (ví dụ `full_rpt` hoặc nhiễu trắng) thì phải xác nhận ở seed
  72–81 với α_2 = 0.0125.
