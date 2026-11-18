# EXT17: lỗi cảm biến vật lý trên tín hiệu rung thô của vòng bi (120 run, seed 72–81, 0 lỗi; thăm dò)

Protocol: `selective-rul/EXT17_PROTOCOL.md`. Chi tiết: `ext16_raw_output.txt`.

**Cách tạo lỗi:**
- Tiêm vào tín hiệu gia tốc thô (FEMTO 7 459 snapshot, XJTU-SY 2 761 snapshot) ở 1 trục, trong 30% vòng đời.
- Đặc trưng được tính lại bằng đúng `signal_features`.
- Kiểm tra: lỗi gain làm rms của đúng trục giảm chính xác ×0.5.

**Kết quả** (risk @90% matched, trung bình 5 lỗi):
- `rpt_L` 22.33%, so với OSC 23.27, err_reg 23.38, Gaussian có bias 23.72 và ensemble 25.45.
- `rpt_L` tốt hơn có ý nghĩa so với cả 4 baseline (Holm trên 4): −0.94 (18/20), −1.05 (17/20), −1.39 (18/20) và −3.11 (20/20).
- Từng lỗi đều thấp hơn OSC: gain −0.91, clip −1.03, lỏng gắn kết −0.96, nhiễu −0.77, xung EMI −1.04.
- **Risk dư không giảm:** excess(`rpt_L`) − excess(`full`) = +0.13 (8/20, p = 0.11). Lỗi thô làm risk tăng ít (0.05–0.77 điểm
  trung bình), chủ yếu do lỗi gain và clip. Ở vòng bi, lợi ích vẫn là cải thiện chung của selector, nhất quán với EXT15.
