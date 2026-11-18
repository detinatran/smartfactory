# EXT49: generator có hướng trên ổ cứng, tách cỡ tập selector khỏi dấu của D (thăm dò; run hdd của vòng 8, seed 162–191, 60 run)

## Dấu của D
- Generator thường: D trung bình −2.4 điểm, D < 0 ở 100% run.
- Generator có hướng: D_opt trung bình +3.6 điểm, D_opt > 0 ở 100% run. Nhiễu có hướng đúng là làm predictor lạc quan.

## Hiệu so với OSC theo số unit selector (điểm, 30 seed)
| Số unit selector | RPT thường | RPT có hướng |
|---|---|---|
| 15 | −0.72 (29/30 seed tốt hơn) | −0.65 (28/30) |
| 60 | +0.16 (9/30) | +0.16 (8/30) |
| 296 (toàn bộ) | +0.41 (0/30) | +0.42 (0/30) |

## Kết luận (theo diễn giải đã đặt trước)
- Tác hại trên ổ cứng đến từ cỡ tập selector, không phải từ dấu của D. Dù nhiễu đã gây lạc quan (D > 0), RPT vẫn hại khi dùng toàn bộ unit, và vẫn có ích khi chỉ có 15 unit.
- Cách hiểu "bộ điều chuẩn" được ủng hộ.
- Trên ổ cứng, cổng D đúng chỉ vì D < 0 trùng với tập selector lớn. Một cổng dựa trên cỡ tập selector giải thích dữ liệu ít nhất là tốt bằng.
