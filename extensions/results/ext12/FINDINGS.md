# EXT12: chọn và xác nhận phương pháp cuối, stress test, predictor bền, backbone thứ ba

- Protocol: `selective-rul/EXT12_PROTOCOL.md`, chốt trước; sai lệch được ghi ngay trong protocol.
- Số liệu: `select_output.txt`, `conf/confirm_output.txt`, `conf/stress_output.txt`, `robust/robust_output.txt` (nằm cạnh
  file dữ liệu tương ứng) và `transf/backbone_output.txt`.
- Mọi lần chạy: 0 lỗi. Các score dùng chung khớp EXT10 chính xác.

## D. Chọn phương pháp (seed 42–61, 480 run)
Quy tắc đã chốt chọn **`rpt_L`**: đặc trưng OSC (không tái tạo), S + 2 bản nhiễu hàm trơn, A ~ LogUniform(1, 8),
k ≤ round(0.5·n). Hiệu so với baseline tốt nhất của từng dataset: −0.59 / −0.23 / −0.86 / −0.11. Áp RPT cho các bộ ước lượng
có hướng không an toàn, vì hại C-MAPSS.

## C. Xác nhận vòng 2 (seed 72–81, 240 backbone mới, Holm trên 12, α_2 = 0.0125): **XÁC NHẬN**

| `rpt_L` − … | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| `full` (OSC) | −0.55 (19/20, 4e-5) | −0.40 (18/20, 7e-5) | −1.06 (19/20, 4e-5) | −0.90 (19/20, 4e-5) |
| `gauss_bias_aware` | −1.15 (20/20, 2e-5) | −0.25 (17/20, 0.001) | −1.48 (19/20, 8e-5) | −0.54 (16/20, 0.004) |
| `err_reg` | −1.79 (20/20, 2e-5) | −0.25 (17/20, 3e-4) | −1.03 (17/20, 5e-4) | −0.55 (17/20, 0.004) |

- **Risk** của `rpt_L` so với OSC: 8.97 / 9.81 / 21.98 / 3.39 so với 9.52 / 10.21 / 23.04 / 4.29. Tương đối là −5.8% /
  −3.9% / −4.6% / −21%.
- **Lỗi bi quan** tăng tối đa +0.65 điểm (vòng bi).
- **Dữ liệu sạch:** risk giảm hoặc giữ nguyên ở mọi dataset (8.78 / 6.44 / 22.16 / 2.08 so với 8.83 / 6.77 / 23.22 / 2.84).
- **Theo từng lỗi:** `rpt_L` tốt hơn `full` ở **cả 24 cặp** lỗi × dataset, từ −0.06 đến −3.17 điểm.
- **Tiêu chí phụ:** `rpt_L` tốt hơn có ý nghĩa so với **cả hai** bộ ước lượng có hướng ở **cả 4** dataset. Đây là điểm yếu
  lớn nhất của R1, nay đã được giải quyết.

## S. Stress test (seed 72–81)
- **S1, 4 lỗi mới thiết kế sau khi chốt phương pháp** (clip, lag, quant, invert), Holm trên 20: `rpt_L` tốt hơn có ý nghĩa
  so với ensemble, OSC, Gaussian có bias và hồi quy sai số ở **cả 4 dataset**.
  - So với OSC: −0.47 / −0.33 / −1.05 / −1.00.
  - So với R1: tốt hơn có ý nghĩa ở 3/4 dataset (C-MAPSS −0.11, không có ý nghĩa).
- **S2, mức độ và số kênh** (13 kịch bản × 5 baseline × 4 dataset, Holm trên 260):
  - so với OSC, `rpt_L` âm ở 51/52 ô, trong đó 41 ô có ý nghĩa; ô duy nhất dương là pin spike@3, +0.04;
  - so với ensemble, có ý nghĩa ở 47/52 ô.
  - Lợi ích tăng theo mức độ lỗi: bias@6 so với OSC là −1.14 / −0.39 / −1.65 / −3.17.

## R. Predictor bền (cùng nhiễu loạn dùng để train predictor)
- **Predictor bền ở mức đầy đủ coverage:**
  - kém hơn trên dữ liệu sạch ở C-MAPSS và vòng bi (RMSE 17.3 so với 16.1; 0.296 so với 0.276);
  - tốt hơn ở N-CMAPSS và pin khi có lỗi.
- **So sánh hệ thống, risk @90%:** predictor gốc + `rpt_L` tốt hơn có ý nghĩa so với mọi hệ thống dùng predictor bền ở
  C-MAPSS và vòng bi:
  - −1.43 / −3.12 so với predictor bền + `rpt_L`;
  - −1.86 / −4.40 so với predictor bền + OSC.

  Ở N-CMAPSS và pin, predictor bền + `rpt_L` tốt hơn (+0.87 / +1.11, không có ý nghĩa).
- **Diễn giải:**
  - làm bền predictor phụ thuộc dataset và có thể hại độ chính xác;
  - RPT ở selector cho lợi ích nhất quán;
  - hai cách bổ sung cho nhau: với predictor bền, `rpt_L` vẫn tốt hơn OSC ở cả 4 dataset (−0.43 / −0.24 / −1.28 / −0.82).

## B. Patch-Transformer (seed 72–76, 5 cặp, mô tả)
`rpt_L` tốt hơn `full`, `gauss_bias_aware` và `err_reg` ở cả 12 phép so sánh, mức giảm −0.25 đến −1.85, thắng 4–5/5 seed. Với
n = 5, Wilcoxon không đạt ý nghĩa (p nhỏ nhất 0.0625).

## Đính chính ngày 2026-11-14: trọng số fold vòng bi trong bảng theo từng lỗi
- **Lỗi:** các bảng theo từng lỗi gộp nhãn thành `dataset|lỗi`, nên điều kiện `dataset == "bearings"` không khớp và các fold
  vòng bi bị gán trọng số bằng nhau thay vì theo số unit. Đã sửa trong 20 script (dùng `str.startswith("bearings")`).
- **Cột vòng bi đúng** (`rpt_L` − `full`):

  | Lỗi | stuck | bias | noise | intermittent | spike | gain |
  |---|---|---|---|---|---|---|
  | Hiệu (điểm) | −1.03 | −1.38 | −0.89 | −1.01 | −1.01 | −1.03 |

  Trung bình −1.06, khớp bảng chính.
- **Kết luận không đổi:** cả 24 ô đều âm, từ −0.06 đến −3.17.
- Các kết quả gộp theo từng kịch bản riêng (S1, S2, EXT15 overlap) và các con số của dataset khác không bị ảnh hưởng.

## Phân tích độ nhạy post hoc (2026-11-15, review 9 W2): chạy lại quy tắc chọn trên các tập con lỗi
Dữ liệu dev seed 42–61, quy tắc giữ nguyên, chỉ đổi tập lỗi dùng để chấm.

| Tập lỗi dùng để chọn | Cấu hình được chọn |
|---|---|
| Bỏ stuck | `recon_rpt_L` |
| Bỏ bias | `rptw_L` |
| Bỏ noise, intermittent hoặc spike | `rpt_L` |
| Bỏ gain | `recon_rpt_L` |
| Chỉ {stuck, bias, noise} | `rpt_L` |
| Chỉ {intermittent, spike, gain} | `rptw_L` (hoà với `recon_rpt_L`; `rpt_L` kém 0.02) |

- Trong cả 8 trường hợp, cấu hình được chọn đều thuộc nhóm biên độ lớn (A ∈ [1, 8], k ≤ n/2).
- `rpt_L` luôn nằm trong top 3 và cách cấu hình được chọn ≤ 0.1 điểm.
