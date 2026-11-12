# EXT6-C: xác nhận `full_recon` bằng seed mới. Kết quả ngày 2026-11-11

Protocol: `research/selective-rul/EXT6C_PROTOCOL.md`, chốt trước khi train.
- Train 40 backbone mới: C-MAPSS FD001–4 × GRU/TCN × seed 52–56, CPU trên van, mất 15 phút, 0 run lỗi.
  - Config giống hệt các run của bài; seed mới nên split cũng mới.
- Phân tích: `scripts/ext6_repair.py` giữ nguyên, không sửa dòng nào. Kiểm định bằng `scripts/ext6c_test.py`.
- Có 2 240 hàng.

## Kiểm định xác nhận (một kiểm định duy nhất, chốt trước)
H: `full_recon` có risk thấp hơn `full` trên C-MAPSS. Chỉ số là risk ở coverage 90% matched, trung bình 6 lỗi chưa thấy.

| | GRU 52 | 53 | 54 | 55 | 56 | TCN 52 | 53 | 54 | 55 | 56 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hiệu (điểm) | −0.11 | −0.12 | −0.18 | −0.38 | −0.42 | −0.39 | −0.19 | −0.15 | −0.30 | −0.86 |

**Hiệu trung bình −0.31 điểm, thắng 10/10 cặp, Wilcoxon một phía p = 0.00098: XÁC NHẬN.**

- Theo subset: FD001 −0.20 (8/10 cặp), FD002 −0.29 (10/10), FD003 −0.31 (8/10), FD004 −0.45 (10/10).
- Hiệu ở seed mới (−0.31) cùng tầm với EXT6 (−0.25).

## Mô tả thêm
Số liệu ở seed mới, đơn vị %:

| Điểm | Risk @90% khi có lỗi | Risk @90% khi sạch | Ngưỡng frozen: risk / coverage |
|---|---|---|---|
| ensemble_std | 10.87 | 9.78 | 10.06 / 81.9 |
| impact | 8.99 | 10.79 | 7.42 / 72.9 |
| gauss_bias_aware | 10.10 | 9.00 | 10.07 / 90.6 |
| full (bài ESANN) | 9.38 | 8.59 | 9.23 / 89.6 |
| **full_recon** | **9.07** | **8.55** | 8.57 / 88.0 |
| repair_full | 9.32 | 8.60 | 8.98 / 88.8 |

- Kiểm tra lại: `full` ở seed mới là 9.38%, so với 9.37% ở seed 42–51, nên các seed mới cùng tầm với seed cũ.
- Dữ liệu sạch: `full_recon` không gây hại (8.55 so với 8.59).
- Ngưỡng frozen: risk thấp hơn 0.66 điểm, nhưng coverage cũng thấp hơn 1.5 điểm. Một phần lợi ích ở đây là nhờ từ chối nhiều
  hơn.
- Lỗi bi quan ở coverage 90%: 32.88 so với 32.70, gần như không đổi.

## Kết luận
1. Cải thiện của `full_recon` trên C-MAPSS là **thật**, không phải dương tính giả: lặp lại trên seed và split hoàn toàn mới,
   10/10 cặp, ở cả 4 subset.
2. Mức cải thiện **nhỏ**: khoảng 0.3 điểm, tức khoảng 3% tương đối. Và **chỉ đã được xác nhận trên C-MAPSS**: ở EXT6,
   N-CMAPSS, vòng bi và pin đều hoà.
3. Cách viết trung thực:
   - Thêm một đặc trưng "dự đoán thay đổi bao nhiêu khi thay các kênh khả nghi bằng giá trị tái tạo từ các kênh còn lại"
     giúp giảm risk trên C-MAPSS khoảng 0.3 điểm. Kết quả này được xác nhận bằng seed mới.
   - Trên ba họ dữ liệu còn lại, không có tác dụng.
   - Lý do hợp lý: các cảm biến của C-MAPSS tương quan mạnh và điều kiện vận hành ổn định.
4. Không đủ cơ sở để **thay** phương pháp chính đã đăng ký trước, vì chỉ lợi ở 1/4 họ dữ liệu. Hợp lý hơn là ghi thành một
   mở rộng đã được xác nhận.
