# EXT39: kiểm tra cơ chế bằng đặc trưng độ nhạy (thăm dò, chạy trên các run của vòng 7, seed 152–161)

Số liệu đầy đủ ở `ext39_output.txt`. Đơn vị là điểm phần trăm; risk đo ở coverage 90% (matched), trung bình qua 6 loại lỗi.

## Kết quả
- **M1: `sens` − `full`.** Hiệu nằm trong khoảng −0.11 đến +0.20; không họ nào có ý nghĩa sau Holm. Đặc trưng độ nhạy của predictor theo từng kênh, khi đứng một mình, không giúp gì.
- **M2: `rpt_L` − `sens`.**
  - Bốn họ benchmark: C-MAPSS −0.53 (18/20 cặp), N-CMAPSS −0.69 (20/20), vòng bi −0.74 (15/20), pin −0.53 (16/20); p_Holm ≤ 0.034.
  - hdd4: −0.13, không có ý nghĩa.
  - Theo diễn giải đã đặt trước, bản nhiễu mang thông tin vượt ra ngoài độ nhạy tuyến tính theo từng kênh.
- **M3: `rpt_sens` − `rpt_L`.** Hiệu nằm trong khoảng −0.07 đến +0.14, không có ý nghĩa. Hai cách không bổ sung cho nhau.

## Kết luận
- Lợi ích của RPT không phải là "học độ nhạy cục bộ của window sạch".
- Bản nhiễu mở rộng dữ liệu huấn luyện của selector sang vùng mà lỗi đẩy window test tới. Mỗi bản được gắn nhãn bằng lỗi thật của predictor ở đó.
- Bài đã sửa ba chỗ trước đây nói "selector học độ nhạy": main L337, sec_novelty (thêm một bullet), và phần discussion của bản rút gọn.

## Phân tích phụ (`ext39_oracle_gap.py`, trên dữ liệu vòng 7)
- OSC thu hẹp được 27/86/29/48% khoảng cách từ "nhận hết" tới oracle (C-MAPSS/N-CMAPSS/vòng bi/pin).
- RPT-D thu hẹp được 34/91/32/59%, tức là thêm 10/39/4/20% phần khoảng cách mà OSC còn để lại.
- Kết quả này đã đưa vào main.tex, đoạn "Headroom".
- Phân tích mô tả "tương quan với max Δ" ghi trong protocol chưa làm: các file CSV không lưu Δ theo từng window.
