# EXT48: vòng xác nhận 9 — XÁC NHẬN

Lợi ích riêng cho lỗi của RPT-D: excess trên window bị ảnh hưởng, kiểm định ở mức seed, 30 seed mới (192–221). 1260 ensemble, 30/30 seed đủ ở mọi họ. Hai lần lệch vận hành đã ghi trong protocol trước khi có kết quả.

Đơn vị: điểm phần trăm. Kiểm định Wilcoxon theo seed, hiệu chỉnh Holm.

## Kết quả chính: Δexcess_A, RPT-D − OSC (Holm trên 5, α_9 = 9.77e-5)
| Họ | Hiệu | Seed tốt hơn | p_Holm |
|---|---|---|---|
| C-MAPSS | −1.10 | 30/30 | 9e-9 |
| N-CMAPSS | −0.71 | 30/30 | 9e-9 |
| Pin | −2.04 | 30/30 | 9e-9 |
| Vòng bi | −0.04 | 19/30 | 0.25 (n.s.) |
| HDD | 0.00 | — | — (D < 0, RPT-D giữ OSC ở mọi deployment) |

Kết luận: **CONFIRMED**.

## Kết quả phụ (α = 0.05 trong mỗi họ, ngoài ngân sách α)
- **(a) Risk trên toàn tập test:** RPT-D tốt hơn OSC ở cả 4 họ benchmark, đều có ý nghĩa.

| Họ | Hiệu | Seed tốt hơn |
|---|---|---|
| C-MAPSS | −0.58 | 30/30 |
| N-CMAPSS | −0.28 | 30/30 |
| Vòng bi | −0.57 | 29/30 |
| Pin | −1.04 | 30/30 |

  Đây là lần lặp lại thứ hai của vòng 8, cũng ở mức seed.
- **(b) Excess của lỗi masking:**
  - Giảm có ý nghĩa ở C-MAPSS (−2.30), N-CMAPSS (−0.26) và pin (−0.58).
  - Vòng bi tăng +0.15, không có ý nghĩa.
- **(c) So với bộ ước lượng định hướng được huấn luyện cùng bản nhiễu:**
  - RPT-D tốt hơn trên C-MAPSS (−0.54 so với `err_reg_rptL`, −0.24 so với `gauss_rptL`) và trên HDD (−0.39, −0.42).
  - RPT-D kém hơn trên vòng bi (+0.90, +0.73), và kém nhẹ trên N-CMAPSS (+0.10) và pin (+0.05).
  - Như vậy đóng góp chính vẫn là cách huấn luyện bằng nhãn lỗi, không phải bộ phân loại.
- **(d) Mô phỏng bảo trì với ngưỡng đóng băng:**
  - Tỷ lệ hỏng gần 0 ở mọi selector trên C-MAPSS, N-CMAPSS và pin. Riêng vòng bi, RPT-D giảm tỷ lệ hỏng 0.75 điểm phần trăm (8.65% so với 9.40%).
  - Đổi lại, RPT-D cần thêm 0.2–0.6 lần kiểm tra trên mỗi unit, và lãng phí tuổi thọ nhiều hơn một chút.
- **(e) Glitch pin:** đang chạy, ghi kết quả sau.

## (e) Glitch pin (phụ, đặt trước; ext41 trên các run battery_cv vòng 9)
- Predictor lạc quan 6.61% trên window glitch, so với 3.03% khi ghép cặp theo tuổi pin.
- Risk trên window glitch được chấp nhận: RPT-D 0.71%, OSC 2.51%. Hiệu −1.80 điểm, 29/30 seed tốt hơn, p = 5.6e-9.
- Bootstrap theo cell (suy luận chính đã đặt trước): hiệu gộp −1.55, CI 95% [−3.98, −0.15]. 72 cell có window glitch: 16 cell nghiêng về RPT-D, 0 cell nghiêng về OSC, 56 cell bằng nhau.
- **Kết quả thăm dò EXT41 được lặp lại** trên seed mới, với phân tích đã đặt trước.
