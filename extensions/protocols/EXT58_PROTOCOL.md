# EXT58: đối chứng offset công bằng cho V15. Áp cùng hiệu chỉnh độ lệch cuối đời lên predictor mặc định (THĂM DÒ). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Lý do:** V15 (RAMP + offset cuối đời) làm giảm mạnh lỗi lạc quan trên vòng bi (RPT-D còn 8.8%). Nhưng lỗi bi quan tăng từ 35.9% lên 55.9% và RMSE kém đi (0.321). Như vậy về bản chất V15 là một offset bảo thủ. Muốn so công bằng thì predictor mặc định cũng phải được áp đúng offset đó.

**Thiết kế:**
- Script `scripts/ext58.py`: lấy `ext38.py`, tính offset bằng phần dư có dấu trung bình (có trọng số theo unit) trên các window validation sạch có y ≤ 0.3·cap.
- Offset này được trừ khỏi mọi dự đoán của ensemble mặc định: nhãn huấn luyện selector, D, calibration và đánh giá.
- Dữ liệu: vòng bi, seed 162–171.

**So sánh:** V15 với "mặc định + cùng offset". Hai chỉ số: lỗi lạc quan của RPT-D và lỗi bi quan dưới lỗi.
