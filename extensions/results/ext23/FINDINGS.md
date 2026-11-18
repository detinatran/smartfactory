# EXT23: đối chứng cân lại trọng số lớp (seed 72–81, 240 run, 0 lỗi; thăm dò)

- **Tỷ lệ nhãn dương** (S → dữ liệu RPT): C-MAPSS 11.4% → 14.1%, N-CMAPSS 19.1% → 23.0%, vòng bi 24.8% → 24.3%, pin 4.6% →
  10.8%.
- **OSC được cân lại theo tỷ lệ đó** (`full_rw`) **hoặc cân bằng hoàn toàn** (`full_bal`) cho risk gần như bằng OSC, cả khi có
  lỗi lẫn trên dữ liệu sạch.
- **`rpt_L` tốt hơn cả hai đối chứng có ý nghĩa ở mọi dataset** (Holm trên 8):
  - so với `full_rw`: −0.53 / −0.39 / −1.08 / −0.90;
  - so với `full_bal`: −0.67 / −0.46 / −1.32 / −0.85;
  - thắng 18–20/20 cặp.
- **Kết luận:** lợi ích của RPT, kể cả trên dữ liệu sạch, không đến từ việc dịch chuyển tỷ lệ lớp.
