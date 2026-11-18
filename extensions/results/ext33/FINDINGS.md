# EXT33: lỗi bias có chủ đích (3 kênh predictor nhạy nhất, theo chiều lạc quan, 3σ), 2026-11-15. Post hoc
260 run vòng 5 (seed 112–121), 0 lỗi.

## Tác động của lỗi trên window bị ảnh hưởng
Tỷ lệ lạc quan của predictor, sạch → `tbias` (%):

| Họ | Sạch → lỗi |
|---|---|
| C-MAPSS | 17.1 → 34.7 |
| N-CMAPSS | 18.4 → 98.7 |
| Vòng bi | 21.8 → 58.4 |
| Pin | 0.5 → 95.5 |
| HDD | 78.2 → 89.6 |

Lỗi làm predictor lạc quan hơn ở **mọi** họ.

## T1: Δ`excess_A` (Holm trên 10)
- **`rpt_auto` − `full`:**
  - C-MAPSS −1.14 (17/20, p_holm = 0.013);
  - vòng bi −1.03 (16/20, 0.013);
  - pin −0.73 (p = 0.049, p_holm = 0.19);
  - N-CMAPSS +0.20 (n.s.); HDD +0.30 (n.s.).
- **`rpt_L` − `full`:** C-MAPSS −1.86\*; vòng bi −1.10 (p_holm = 0.08); HDD +0.87 (p_holm = 0.07).

## T2: `risk_matched`, `rpt_auto` − `full`
C-MAPSS −0.72\*, vòng bi −0.95\*, pin −0.71\*, N-CMAPSS 0.00, HDD +0.06.

## Kết luận
- Lợi ích riêng cho lỗi xuất hiện lần đầu trên vòng bi khi lỗi thực sự gây lạc quan.
- Trên N-CMAPSS và pin, lỗi làm gần như toàn bộ window bị ảnh hưởng thành lạc quan (vượt quá 10% mà coverage 90% cho phép từ chối), nên không giảm được risk dư.
