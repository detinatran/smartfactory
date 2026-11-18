# EXT64 (thăm dò, seed mới 252–281): cổng chọn predictor (mặc định hoặc gộp mặc định + ELPIS-J-V13) theo risk calibration

Cổng so với ELPIS-G trên predictor mặc định (xem `ext64_output.txt`):

| Họ | Hiệu | CI 90% | Seed tốt hơn | Wilcoxon | Kết luận |
|---|---|---|---|---|---|
| Vòng bi | −0.54 | [−0.97, −0.13] | 18/30 | p = 0.13 | đạt "hòa", không đạt "thắng" |
| C-MAPSS | −0.58 | | 29/30 | p < 1e-4 | thắng |
| Pin | −0.54 | | 30/30 | p < 1e-4 | thắng |

Lỗi bi quan giảm 0.43–0.58 điểm ở cả 3 họ. RMSE thấp hơn ở cả 3 họ.
