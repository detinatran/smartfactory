# EXT36: đo hai thành phần của cận chuyển giao (2026-11-15). Thăm dò
240 run vòng 4 (seed 92–101, 4 họ benchmark), 0 lỗi.

## Sai số ước lượng: Brier score ×100, `rpt_L` − `full` (Holm trên 4)
| Họ | Trên P (calibration, shift đã thấy) | Trên Q (test, 6 lỗi) |
|---|---|---|
| C-MAPSS | −0.11 (−1%, n.s.) | −0.90 (−9%, 20/20) |
| N-CMAPSS | −1.65 (−25%, 20/20) | −2.05 (−27%, 20/20) |
| Vòng bi | −4.13 (−16%, 17/20) | −3.81 (−15%, 19/20) |
| Pin | −1.19 (−36%, 19/20) | −0.06 (n.s.) |

## Độ phủ (classifier phân biệt miền, φ train so với φ test có lỗi)
- **AUC, `rpt_L` − `full`:** C-MAPSS −0.02\*, N-CMAPSS +0.02\*, vòng bi +0.04\*, pin +0.02 (n.s.). Không nhất quán.
- **χ²:** giảm 20–38%. Nhưng giá trị ở mức hàng trăm đến hàng nghìn và bị chi phối bởi trọng số đã clip, nên không đáng tin.

## Kết luận theo diễn giải đã chốt
- Hiệu đo được nằm ở thành phần **ước lượng**.
- Trên C-MAPSS, hiệu chỉ có trên Q, khớp với nhận định rằng lợi ích ở đây là lợi ích riêng cho lỗi.
- Bài giữ nhận định cũ, có số liệu kèm theo.
