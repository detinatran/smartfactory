# EXT19: trả lời review lần 3. Chốt ngày 2026-11-14, TRƯỚC khi chạy

Dữ liệu: seed 72–81 (GRU/TCN, 240 run). **Thăm dò; phương pháp cuối (`rpt_L`) không đổi.** Code: `scripts/ext19.py`, dựa trên
`ext12_main.py` v2. `full`, `rpt_L`, `gauss_bias_aware`, `err_reg` và `ensemble_std` phải khớp EXT12.

## A. Đối chứng cân bằng lượng dữ liệu và oracle ở cấu hình cuối
Câu hỏi: lợi ích là do nhiều dữ liệu hơn (điều chuẩn hoá), hay do phủ được vùng lỗi?

| Tên | Định nghĩa (đặc trưng `f`, HGB như `rpt_L`) |
|---|---|
| `copy2` | S + 2 bản sao **sạch** của cửa sổ selector (cùng lượng dữ liệu với `rpt_L`) |
| `tiny2` | S + 2 bản nhiễu rất nhỏ (A ~ LogUniform(0.01, 0.05), cùng k, onset và seed với `rpt_L`) |
| `oracle_f` | S + 6 lỗi đánh giá trên unit selector |
| `oracle_rpt` | S + 2 bản nhiễu L + 6 lỗi đánh giá |

- **Kiểm định:** `rpt_L` so với `copy2` và `tiny2`, Holm trên 8.
- **Giả thuyết chốt trước, H19a:** `rpt_L` tốt hơn cả `copy2` lẫn `tiny2` ở ≥ 3/4 dataset. Nếu không đúng, lợi ích chủ yếu đến
  từ lượng dữ liệu.
- Hai oracle chỉ để mô tả.

## B. Pipeline phát hiện lỗi → tái tạo → dự đoán (FDI + imputation)
- Dùng bộ tái tạo Ridge của EXT6: kênh nào nghi lỗi (vượt ngưỡng 99.5%) thì được thay bằng giá trị tái tạo, rồi chạy lại
  ensemble trên đầu vào đã sửa.
- So sánh **ở mức hệ thống**, mỗi hệ thống dùng dự đoán riêng của nó:
  - `repair_spread`: predictor đã sửa + độ phân tán ensemble;
  - `repair_osc`: predictor đã sửa + OSC train trên dữ liệu đã sửa;
  - `repair_rpt`: predictor đã sửa + `rpt_L` train trên dữ liệu đã sửa.
- **Kiểm định:** `rpt_L` (predictor gốc) so với `repair_spread` và `repair_osc`, Holm trên 8.

## C. Độ nhạy theo δ
- Chạy lại các score cốt lõi (`ensemble_std`, `full`, `gauss_bias_aware`, `err_reg`, `rpt_L`) với δ' = 0.5·δ và 1.5·δ. Nhãn
  huấn luyện và metric đều dùng δ'.
- **Kiểm định:** `rpt_L` so với `full` ở mỗi δ', Holm trên 8.

## D. AURC (không chạy thêm)
- Diện tích dưới đường risk–coverage trên [0.5, 0.95] theo quy tắc hình thang, từ các cột quét coverage của `ext12_conf`.
- So sánh `rpt_L` với các baseline, Wilcoxon, Holm trên số phép so sánh.
