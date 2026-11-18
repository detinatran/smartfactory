# EXT51: ELPIS-J không có pretrain JEPA (Stage I = 0 epoch); thăm dò; seed 162–171; 300 run

Risk RPT-D đo dưới 6 loại lỗi, coverage 90% (%), trung bình trên 10 seed.

| Họ | Mặc định (2 giai đoạn) | ELPIS-J bỏ JEPA | ELPIS-J đầy đủ | Phần do JEPA (đầy đủ − bỏ JEPA) |
|---|---|---|---|---|
| C-MAPSS | 7.87 | 7.20 | 7.04 | −0.16 (6/10, n.s.) |
| Pin | 2.35 | 2.75 | 1.87 | **−0.88** (9/10, p = 0.004) |
| Vòng bi | 23.76 | 25.22 | 27.06 | **+1.84** (JEPA tốt hơn ở 1/10 seed, p = 0.049) |

## Tách nguồn gốc lợi ích và tác hại
- **C-MAPSS:** phần lớn lợi ích đến từ kiến trúc, tức nhánh symbolic có cổng, điều kiện chất lượng và head fusion (−0.66 so với mặc định). JEPA chỉ thêm một chút, không có ý nghĩa.
- **Pin:** chỉ riêng kiến trúc thì kém hơn mặc định (+0.40). Toàn bộ lợi ích đến từ pretrain JEPA: tỷ lệ lạc quan dưới lỗi giảm từ 8.3% xuống 6.3%.
- **Vòng bi:** cả hai thành phần đều gây hại. Kiến trúc làm tăng 1.46 điểm, JEPA làm tăng thêm 1.84 điểm, và predictor lạc quan hơn khi có lỗi.

Trên mọi biến thể predictor, RPT-D vẫn tốt hơn OSC: C-MAPSS −0.83, pin −0.99, vòng bi −0.52.

## Ý nghĩa cho EXT52 (đang chạy)
- V2 (tắt nhánh symbolic) nhắm vào phần hại do kiến trúc.
- V1 (thêm dữ liệu cho pretrain) nhắm vào phần hại do JEPA.
