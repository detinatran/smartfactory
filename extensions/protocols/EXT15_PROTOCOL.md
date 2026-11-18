# EXT15: trả lời review mô phỏng (weak reject 4/10). Chốt ngày 2026-11-13, TRƯỚC khi chạy

Các điểm review được xác nhận là đúng sau khi kiểm tra (phân tích lại ở `extension-results/2026-11-09/ext15/reanalysis_output.txt`):
- **W1:** lợi ích phần lớn không đặc thù cho lỗi. Risk dư do lỗi chỉ giảm rõ ở C-MAPSS (−0.50, 20/20) và với lỗi bias ở pin.
- **W4:** đơn vị kiểm định (cặp) không độc lập. Ở mức seed, kết luận vẫn giữ: 40/40 CI không chứa 0.
- **W5:** chỉ ưu thế khi chi phí lỗi lạc quan ≥ 2 lần lỗi bi quan.
- **W2:** có thể chỉ là augmentation hoặc outlier exposure.
- **W3:** họ nhiễu có thể trùng với lỗi đánh giá.

EXT15 bổ sung bằng chứng cho W2 và W3, cùng kiểm tra tiền đề của Mệnh đề 1. Dữ liệu: seed 72–81 (đã dùng cho xác nhận vòng 2),
GRU/TCN. **Mọi kết quả là thăm dò; phương pháp cuối không đổi.**

## 1. Novelty: nhãn theo sai số của predictor, hay chỉ là augmentation hoặc outlier exposure? (`scripts/ext15_more.py`)
Cùng dữ liệu nhiễu với `rpt_L` (2 bản, A ∈ [1, 8], k ≤ 0.5n, cùng seed), cùng đặc trưng `f` và HGB:

| Tên | Định nghĩa |
|---|---|
| `oe_label` | outlier exposure (Hendrycks 2019): mọi cửa sổ nhiễu được gán nhãn 1 ("từ chối"), S giữ nhãn sai số |
| `det_rpt` | bộ phát hiện nhiễu: nhãn = "cửa sổ bị nhiễu", trên S + nhiễu |
| `symmetric_rpt` | bộ phân loại đối xứng \|E\| > δ, train trên S + nhiễu (nhãn theo sai số thật) |
| `err_reg_rptL` | hồi quy sai số có dấu, train trên S + nhiễu cấu hình L |
| `gba_rptL` | Gaussian có bias, train trên S + nhiễu cấu hình L |

**Kiểm định:** `rpt_L` so với từng biến thể, Holm trên 5 × 4 = 20 phép thử. Giả thuyết chính là **H15a**: `rpt_L` tốt hơn
`oe_label`, tức việc gán nhãn bằng sai số của predictor là cần thiết.

## 2. Lỗi khác về cấu trúc: lỗi trên mọi kênh
Bộ sinh chỉ nhiễu tối đa 50% số kênh. Thêm 4 kịch bản lỗi trên **mọi kênh**:
- `bias@1#99` (lệch đồng pha 1σ);
- `stuck#99`;
- `gain@1.25#99`;
- `noise@1#99`.

Số kênh bị kẹp ở mức tối đa của dataset; với C-MAPSS, sửa `data.corrupt` để kẹp tương tự như phía external. Kiểm định
`rpt_L` so với `ensemble_std`, `full`, `gauss_bias_aware` và `err_reg` trên trung bình 4 kịch bản, Holm trên 16.

## 3. Đo độ trùng giữa bộ sinh và từng lỗi
- Trên unit selector, so sánh đặc trưng `f` của cửa sổ nhiễu (cấu hình L, bản 1) với đặc trưng `f` của cửa sổ thật sự bị
  lỗi (chỉ các cửa sổ có thay đổi), cho từng lỗi trong 6 lỗi cũ, 4 lỗi held-out và 4 lỗi mọi kênh.
- Dùng HGB, AUC kiểm định chéo 2 fold theo unit. AUC ≈ 0.5 nghĩa là trùng nhiều, AUC ≈ 1 nghĩa là tách biệt.
- Mô tả, kèm tương quan giữa AUC và lợi ích của `rpt_L` so với `full` theo từng lỗi.

## 4. Tiền đề của Mệnh đề 1: lỗi làm lệch μ nhiều hơn σ
- Trên cửa sổ đánh giá bị lỗi, tính Δμ = trung bình ensemble khi có lỗi − khi sạch, và Δσ = độ lệch chuẩn ensemble khi có
  lỗi − khi sạch (đơn vị RUL đã chuẩn hoá theo cap).
- Báo cáo trung vị của |Δμ| / |Δσ|, và tỷ lệ cửa sổ lạc quan mới phát sinh mà độ phân tán vẫn dưới trung vị.
- Mô tả.

## Kết quả ngày 2026-11-13
Xem `extension-results/2026-11-09/ext15/FINDINGS.md`.
- H15a đúng: gán nhãn kiểu outlier exposure kém hơn 1.6–5.2 điểm ở mọi dataset.
- Lỗi trên mọi kênh: `rpt_L` tốt hơn mọi baseline ở mọi dataset.
- Độ trùng không giải thích được lợi ích.
- Tiền đề Δμ ≫ Δσ đúng.
