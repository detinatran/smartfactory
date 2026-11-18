# EXT16: head baseline có dữ liệu nhiễu, ensemble M=5, biến thể hồi quy sai số (seed 72–81; thăm dò)

Protocol: `selective-rul/EXT16_PROTOCOL.md`.

## B. Ensemble 5 thành viên (240 run, 0 lỗi)
- `rpt_L` tốt hơn có ý nghĩa ở cả 16 phép so sánh (Holm trên 16): so với ensemble spread, OSC, Gaussian có bias và hồi quy
  sai số, ở cả 4 dataset.
- Ví dụ so với OSC: −0.55 / −0.44 / −1.29 / −1.02.
- M=5 gần như không cải thiện ensemble spread (10.77 / 17.42 / 24.56 / 5.07). Lợi ích của `rpt_L` giữ nguyên hoặc tăng.

## C. Biến thể hồi quy sai số (240 run, 0 lỗi; khớp EXT12 chính xác)
- Loss tuyệt đối hoặc kẹp mục tiêu chỉ cải thiện nhẹ. Ở C-MAPSS, risk vẫn khoảng 10.4%; đây là hạn chế vốn có của hồi quy
  trên sai số có đuôi nặng (RUL bị cap trong khi nhãn không cap), không phải lỗi code.
- Theo quy tắc bảo thủ (biến thể mạnh nhất ở từng dataset), `rpt_L` vẫn tốt hơn có ý nghĩa ở cả 4 dataset: −1.46 / −0.17 /
  −1.03 / −0.51 (Holm trên 4).
- Có dữ liệu nhiễu L, các biến thể hồi quy ngang `rpt_L` ở N-CMAPSS và vòng bi, tốt hơn ở pin (+0.37 đến +0.40), kém ở
  C-MAPSS (−0.57 đến −0.83). Nhất quán với EXT15.

## A. Head baseline có dữ liệu nhiễu (240 run, 0 lỗi)
- Các head (deep ensemble NLL, evidential, CQR, SelectiveNet) được train trên cửa sổ fit sạch + 1 bản nhiễu L (nhãn RUL
  thật). Chúng **không** bắt kịp `rpt_L`: tốt hơn có ý nghĩa ở **16/16** phép so sánh (Holm trên 16).
- Mức chênh: deep ensemble −1.8 đến −7.2; evidential −1.9 đến −7.3; SelectiveNet −2.3 đến −8.2; CQR −1.09 / −3.74 / −2.03
  / −0.34.
- Dữ liệu nhiễu giúp CQR ở N-CMAPSS (15.43 → 13.55) và pin (4.98 → 3.73), nhưng hại hoặc không đổi với các head khác.
- Kết luận: việc dùng chung dữ liệu nhiễu không giải thích lợi ích; chính selector có hướng, được gán nhãn bằng sai số của
  predictor, mới là yếu tố cốt lõi.
