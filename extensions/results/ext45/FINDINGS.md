# EXT45: predictor bảo thủ, dịch dự đoán xuống k·δ (thăm dò; seed 162–166 của vòng 8; 210 run cho mỗi k)

Bảng đầy đủ ở `ext45_output.txt`.

## Kết quả
- **D trên benchmark vẫn dương ở cả hai mức k.**
  - k = 1: C-MAPSS +6.6, N-CMAPSS +13.2, pin +9.3.
  - k = 2: C-MAPSS +3.2, N-CMAPSS +8.2, pin +4.1.
  - Hiệu RPT − OSC trên các họ này: k = 1 từ −0.49 đến −0.56; k = 2 từ −0.24 đến −0.42.
- **Vòng bi:** D gần 0. Ở các deployment có D ≤ 0, RPT vẫn có ích: −1.08 điểm (k = 1) và −0.20 điểm (k = 2).
- **HDD:** D < 0 ở mọi deployment, và RPT gây hại: +0.62 (k = 1), +0.46 (k = 2).
- **Gộp mọi họ, k = 2:**
  - D > 0: RPT − OSC = −0.32;
  - D ≤ 0: RPT − OSC = +0.01;
  - Mann-Whitney p = 0.00075.
  - Tuy vậy, nhóm D ≤ 0 chỉ gồm vòng bi (nơi RPT có ích) và HDD (nơi RPT gây hại).

## Kết luận
- Dịch predictor sang bảo thủ cũng không làm D đổi dấu trên benchmark. Vẫn chưa có trường hợp âm nào trên benchmark mà ở đó RPT gây hại.
- D ≤ 0 chỉ đi kèm tác hại khi tập selector lớn (HDD). Tập selector nhỏ (vòng bi, HDD lấy mẫu con 15 unit) thì RPT vẫn có ích.
- Một cổng hai yếu tố, gồm D và cỡ tập selector, phù hợp với mọi dữ liệu hiện có. Cổng này chưa được xác nhận và không mở vòng mới.
