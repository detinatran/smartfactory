# EXT41: glitch cảm biến tự nhiên trong dữ liệu pin Severson (thăm dò, các run battery_cv của vòng 8, 300 run, 30 seed)

Số liệu đầy đủ ở `ext41_output.txt`. Đơn vị: % (risk, tỷ lệ) và điểm phần trăm (hiệu).

## Định nghĩa chính (đặt trước): glitch trên Q, IR, T
- **Mức chu kỳ:** 0.40% chu kỳ bị glitch.
- **Mức window:** 5.8% window test chứa ít nhất một chu kỳ glitch.

### (1) Glitch thật có làm predictor lạc quan hơn không
| So sánh | Window glitch | Window đối chứng | Hiệu | Seed glitch cao hơn | p |
|---|---|---|---|---|---|
| Mọi window không glitch | 6.66% | 3.53% | +3.13 | 29/30 | 2.6e-8 |
| (1b) Ghép cặp theo tuổi pin: cùng cell, trong ±50 chu kỳ | 6.66% | 3.02% | +3.63 | — | 3.7e-9 |

Glitch thật làm tăng gấp đôi tỷ lệ lỗi lạc quan.

### (2) Risk trên các window glitch được chấp nhận, coverage 90%
| Selector | Risk trên window glitch được chấp nhận | Tỷ lệ window glitch lạc quan bị loại |
|---|---|---|
| RPT-D | 0.61% | 94% |
| OSC | 2.14% | 79% |
| Gaussian có bias | 2.78% | 73% |
| Hồi quy lỗi có dấu (`err_reg`) | 2.62% | 76% |
| Độ phân tán ensemble | 4.06% | 55% |

- **RPT-D − OSC:** −1.53 điểm, 30/30 seed, p = 1.9e-9.
- **RPT-D so với các baseline khác:** tốt hơn 2.0–3.5 điểm, 29–30/30 seed.
- **Window không glitch:** risk giảm từ 1.10% xuống 0.65% (−41%), ít hơn mức giảm trên window glitch (−71%). Phần lợi ích trên glitch vì vậy lớn hơn lợi ích chung.

## Định nghĩa phụ (thêm `chargetime`)
Cùng xu hướng: RPT-D − OSC = −0.98 điểm, 30/30 seed.

## Giới hạn
- Glitch được định nghĩa bằng độ lệch khỏi median trượt. Vì vậy có thể lẫn một số sự kiện suy giảm đột ngột có thật, chứ không chỉ lỗi cảm biến.
- Kết quả chỉ trên một họ (pin). Dữ liệu HDD đã qua xử lý không còn glitch: không có ngày trống, gần như không có counter giảm.

## Bootstrap theo cell (post hoc)
- Gộp mọi run theo cell, resample 72 cell có window glitch.
- Hiệu RPT-D − OSC: −1.32 điểm, CI 95% [−3.28, −0.10].
- Hiệu này đến từ 15 cell: 14 cell nghiêng về RPT-D, 1 cell nghiêng về OSC. 57 cell còn lại không có window glitch lạc quan nào được chấp nhận, ở cả hai selector.
- Bằng chứng ở mức tài sản yếu hơn nhiều so với p ở mức seed. Bài đã ghi rõ điều này.

## Sensitivity: chỉ IR và nhiệt độ (bỏ Q vì dung lượng có thể bật lên sau giai đoạn nghỉ; post hoc)
- 4.8% window bị đánh dấu.
- Predictor lạc quan 6.97% trên window glitch, so với 3.71% khi ghép cặp theo tuổi pin.
- Risk trên window glitch được chấp nhận: RPT-D 0.66%, OSC 2.20% (−1.54 điểm, 30/30 seed). Kết quả không đổi so với định nghĩa chính.
