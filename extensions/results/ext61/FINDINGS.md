# EXT61 (thăm dò): sửa RAMP trên vòng bi, gồm V16 (hiệu chỉnh mức lạc quan) và V17 (ensemble gộp mặc định + RAMP-V13)

Vòng bi, seed 162–171. Số liệu đầy đủ ở `summary.txt`.

**Kết quả:** không biến thể nào đạt tiêu chí "sửa được" (hiệu ≤ 0 và ≥ 6/10 seed tốt hơn). Hiệu risk lạc quan của RPT-D so với mặc định:
- V13: +1.08;
- V16: +0.09;
- V17: +0.14.

**V17** là biến thể không kém trên chỉ số nào:
- RMSE 0.261 so với 0.280 của mặc định;
- lỗi bi quan ít hơn 3.15 điểm (9/10 seed, p = 0.004);
- risk lạc quan hòa.
