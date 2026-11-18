# EXT62 (thăm dò): predictor gộp (mặc định + RAMP-V13) trên seed mới 222–251

Số liệu đầy đủ ở `ext62_output.txt`. ELPIS-G dưới lỗi, predictor gộp so với predictor mặc định:
- **C-MAPSS:** −0.66 (28/30 seed); lỗi bi quan −0.72; RMSE 14.58 so với 15.85.
- **Pin:** −0.67 (30/30); lỗi bi quan −0.85; RMSE 0.083 so với 0.092.
- **Vòng bi:** +0.35 (15/30, p = 0.61; cận trên CI 90% là 1.21, không đạt biên không kém 0.3); lỗi bi quan −3.54 (26/30); RMSE 0.258 so với 0.275.

So với RAMP-V13, predictor gộp tốt hơn trên vòng bi (−1.63) và pin (−0.22), còn C-MAPSS hòa.

ELPIS-G vẫn tốt hơn OSC trên predictor gộp, ở cả 3 họ (28–30/30 seed).
