# Nhật ký tìm phương pháp (bắt đầu từ EXT9, theo EXT9_PROTOCOL.md)

Các vòng trước khi có quy tắc này đều khám phá hoặc xác nhận riêng lẻ. Không vòng nào vượt `full` trên cả 4 dataset.
- EXT2–EXT5;
- EXT6 và EXT6-C (`full_recon`: xác nhận ở C-MAPSS);
- EXT7 (`stack`: đạt tiêu chí cũ, nhưng hại vòng bi);
- EXT8 cùng holdout seed 57–61 (`rank_recon`: không xác nhận).

| Vòng | Phương pháp chính | Seed xác nhận | α | Kết quả |
|---|---|---|---|---|
| — (EXT9) | cổng dư thừa (`rg_*`) | không dùng | 0 | Giả thuyết bị bác bỏ ngay ở dữ liệu train (ρ ngược chiều), không xác nhận gì |
| 1 (EXT10) | `full_recon_rpt` (selector học thêm trên nhiễu loạn ngẫu nhiên) | 62–71 | 0.025 | **XÁC NHẬN**: tốt hơn có ý nghĩa ở cả 4 dataset (p Holm ≤ 0.0032). Dừng vòng lặp theo quy tắc 6 |

| 2 (EXT12) | `rpt_L` (RPT biên độ lớn, thưa, 2 bản, không tái tạo), chọn trên seed 42–61 bằng quy tắc so với baseline tốt nhất | 72–81 | 0.0125 | **XÁC NHẬN**: tốt hơn có ý nghĩa so với `full` và cả hai bộ ước lượng có hướng ở cả 4 dataset (Holm trên 12, p ≤ 0.004) |

| — (EXT18) | `rpt_L_h` (RPT + đặc trưng tham chiếu lịch sử unit) | không dùng (82–91 vẫn mới) | 0 | Không đạt quy tắc phát triển (lỗi bi quan C-MAPSS +1.48 > 1); không mở vòng 3 |

| 3 (EXT20) | `rpt_L`: tái lập, không đổi gì | 82–91 | 0.00625 | **XÁC NHẬN**: tốt hơn `full` ở cả 4 dataset; tốt hơn cả hai bộ có hướng ở 3/4 (pin không có ý nghĩa) |

| 4 (EXT22) | `rpt_auto` (chọn giữa `full` và `rpt_L` theo risk calibration dưới các shift đã thấy) | 92–101, 5 họ gồm HDD | 0.003125 | **XÁC NHẬN**: tốt hơn `full` ở C-MAPSS, vòng bi, pin; N-CMAPSS −0.27 (n.s. ở α_4); HDD +0.03 (n.s.); không họ nào kém |

| 5 (EXT29) | `rpt_auto` không đổi; thước đo chính là Δ risk dư trên window bị lỗi | 112–121, 5 họ | 0.0015625 | **XÁC NHẬN**: giảm có ý nghĩa ở C-MAPSS và pin; N-CMAPSS −0.80 (p = 0.0024, n.s. ở α_5); không họ nào tăng |

| 6 (EXT35) | `rpt_auto_m` (margin chuyển 0.01, chọn ở EXT34) | 142–151, 4 họ + hdd3 | 0.00078125 | **KHÔNG XÁC NHẬN**: không họ nào có ý nghĩa; hdd3 trùng với `full` (không chuyển) |

| 7 (EXT38) | `rpt_auto_d` (dùng RPT khi chẩn đoán calibration D > 0) | 152–161, 4 họ + hdd4 | 0.000390625 | **XÁC NHẬN**: C-MAPSS −0.64, N-CMAPSS −0.50; không kém trên hdd4 |

Tổng số vòng xác nhận đã chạy theo quy tắc này: **10** (α đã tiêu: 0.049951171875). EXT9 bị bác bỏ trước bước xác nhận. Vòng 2 được mở để thay phương pháp chính bằng biến thể đơn giản hơn và mạnh hơn (EXT12_PROTOCOL.md).

**Phân tích sau xác nhận (không tiêu α):** EXT23 (đối chứng cân lại trọng số lớp) và EXT24/EXT24b (triển khai với ngưỡng cố định
trên calibration, có tỷ lệ lỗi; so sánh offset với từ chối). Cả ba đều thăm dò, chạy trên seed đã dùng cho xác nhận.


## Vòng 8 (EXT40), ngày 2026-11-16: XÁC NHẬN
RPT-D trên 30 seed mới (162–191), có xoay unit test, kiểm định ở mức seed, α_8 = 0.0001953125.
- Có ý nghĩa ở cả 4 họ benchmark (27–30/30 seed cải thiện).
- Trên hdd kết quả bằng đúng OSC.

## EXT39 (thăm dò)
Đặc trưng độ nhạy theo từng kênh không thay thế được bản nhiễu: rpt_L tốt hơn sens 0.53–0.74 điểm.

## EXT41 (thăm dò): glitch tự nhiên trong dữ liệu pin
- Glitch làm tăng gấp đôi tỷ lệ lạc quan của predictor.
- Trên các window glitch được chấp nhận, RPT-D có risk 0.61%, OSC 2.14% (30/30 seed).

## EXT42 (post hoc): chi phí ở coverage tối ưu
RPT-D giảm chi phí 5.9–28.2% ở R = 10, trên cả 4 họ benchmark.

## EXT43 (thăm dò): đường cong học trên benchmark
Lợi ích của RPT tăng khi số unit selector giảm (C-MAPSS, N-CMAPSS, pin). Khớp với HDD: RPT đổi từ có ích sang gây hại ở khoảng 60 unit.

## EXT44 (thăm dò): D khi predictor đã robust
D giảm nhưng không đổi dấu trên benchmark. Không tạo được trường hợp âm mới.

## EXT45 (thăm dò): predictor bảo thủ
D vẫn dương trên benchmark. D ≤ 0 chỉ có hại khi tập selector lớn.

## EXT46: repair + conformal cục bộ (theo tinh thần RobustUQ)
RPT-D tốt hơn ở 4/5 họ; kém hơn trên vòng bi.

## EXT47: predictor C-MAPSS lớn hơn
RMSE gần như không đổi; lợi ích RPT-D không đổi.

## EXT49 (thăm dò): tách cỡ tập selector khỏi dấu của D
Tác hại trên ổ cứng đến từ cỡ tập selector, không phải từ dấu của D.

## Vòng 9 (EXT48), ngày 2026-11-17: XÁC NHẬN
Excess trên window bị ảnh hưởng của RPT-D giảm có ý nghĩa ở mức seed trên C-MAPSS, N-CMAPSS và pin.

## EXT50 (thăm dò): predictor kiểu ELPIS-J
Tốt hơn trên C-MAPSS và pin, kém hơn trên vòng bi. Trên ELPIS-J, RPT-D vẫn có ích.

## Vòng 10 (EXT59), ngày 2026-11-17: XÁC NHẬN
Trên predictor ELPIS-J-V13 (JEPA), ELPIS-G tốt hơn OSC ở cả 3 họ, kiểm định ở mức seed.
