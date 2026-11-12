# EXT8 holdout: seed 57–61, chưa từng dùng. Kết quả ngày 2026-11-12

Protocol: `research/selective-rul/EXT8_PROTOCOL.md`, phần bổ sung ngày 2026-11-12, chốt trước khi đánh giá.
- 120/120 run, 0 lỗi. Code `ext8_select.py` không đổi; kiểm định bằng `ext8_holdout.py`. Output: `holdout_output.txt`.
- Kiểm tra lại: `full` ở holdout có risk 8.87 / 11.43 / 20.88 / 3.83%. Các seed này cùng tầm với seed cũ; vòng bi hơi thấp
  hơn.

## Kiểm định chính: `rank_recon` − `full` (Holm trên 4)

| Dataset | Hiệu (điểm) | Số cặp thắng | p Holm | Lỗi bi quan tăng |
|---|---|---|---|---|
| C-MAPSS | **−0.32** | **10/10** | **0.008** | +0.36 |
| N-CMAPSS | −0.21 | 7/10 | 0.48 | +0.01 |
| Vòng bi | **+0.32** | 3/10 | 0.48 | +0.06 |
| Pin | **+0.52** | 4/10 | 0.49 | +0.09 |

**KHÔNG XÁC NHẬN.**
- C-MAPSS tốt hơn có ý nghĩa, và không dataset nào kém có ý nghĩa.
- Nhưng hiệu ở vòng bi (+0.32) và pin (+0.52) vượt ngưỡng an toàn +0.3 điểm đã chốt trong protocol EXT8.
- Nếu chỉ dùng tiêu chí của EXT7 (không có ngưỡng +0.3), kết quả sẽ "đạt". Nhưng ngưỡng +0.3 đã được chốt trước, nên phải
  tính theo nó.
- Ở pin, hiệu dương do vài cặp lớn kéo lên: chỉ 4/10 cặp kém hơn, p = 0.49. Dù vậy, vẫn không có bằng chứng là an toàn.

## Phụ: mọi ứng viên trừ `full`, điểm (số cặp thắng trên 10)

| Ứng viên | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| stack_plus | −0.55 (10) | −0.22 (9) | +0.78 (3) | +0.34 (5) |
| stack | −0.52 (10) | −0.17 (7) | +0.69 (3) | +0.38 (5) |
| blend | −0.33 (10) | −0.18 (9) | +0.32 (3) | +0.14 (6) |
| rank_recon | −0.32 (10) | −0.21 (7) | +0.32 (3) | +0.52 (4) |
| full_recon | −0.30 (10) | −0.06 (5) | +0.17 (3) | +0.82 (2) |
| select_cv | −0.19 (8) | −0.27 (4) | +0.33 (2) | +0.21 (4) |
| stack_cv | −0.12 (5) | −0.01 (1) | +0.12 (5) | +0.29 (1) |
| gba_recon | +0.04 (4) | −0.16 (6) | +0.86 (1) | +0.10 (6) |

- **Không ứng viên nào đạt tiêu chí với Holm trên 32 kiểm định.**
- Lỗi thiết kế cần ghi nhận: với 10 cặp, p hai phía nhỏ nhất là 0.002. Nhân 32 lần thành 0.0625 > 0.05, nên họ phụ **không
  thể** đạt ý nghĩa dù kết quả có tốt đến đâu. Lẽ ra phải nhận ra điều này trước khi chốt.
  - Dù vậy, bảng trên cho thấy không ứng viên nào an toàn ở cả 4 dataset. Mọi ứng viên đều dương ở vòng bi, và ở pin trừ
    `gba_recon`.
  - Vì vậy dù hiệu chỉnh nhẹ hơn, kết luận vẫn không đổi.

## Tổng hợp các vòng EXT6–EXT8: điều gì lặp lại được?

| Hiệu ứng | EXT6 (seed 42–51) | EXT6-C / EXT7 (52–56) | Holdout (57–61) |
|---|---|---|---|
| `full_recon` giảm risk C-MAPSS | −0.25 (20/20) | −0.31 (10/10) | −0.30 (10/10) |
| `stack` giảm risk C-MAPSS | −0.41 (18/20) | −0.56 (10/10) | −0.52 (10/10) |
| `stack` tăng risk vòng bi | +0.95 (4/20) | +0.85 (2/10) | +0.69 (3/10) |
| Lợi ích ở N-CMAPSS và pin | có (seed cũ) | không | không |

## Kết luận
1. **Phương pháp ESANN (`full`) vẫn là lựa chọn tốt nhất khi dùng chung cho mọi loại dữ liệu.** Sau 7 vòng mở rộng và một
   holdout sạch, không phương pháp nào vượt nó trên cả 4 họ dữ liệu.
2. **Hiệu ứng duy nhất lặp lại chắc chắn là cải thiện trên C-MAPSS** nhờ đặc trưng tái tạo kênh. Nó lặp lại qua 3 bộ seed
   độc lập, mỗi lần thắng mọi cặp:
   - `full_recon`: khoảng −0.3 điểm;
   - `stack`: khoảng −0.5 điểm.
   - Đây là kết quả trung thực cho bài mở rộng: một cải tiến **riêng cho dữ liệu có cảm biến tương quan mạnh**.
3. **Vòng bi luôn bị hại bởi các biến thể stacking.** Ở pin, các phương pháp tái tạo cũng có xu hướng hại trên holdout.
4. **Seed cũ thường cho kết quả lạc quan giả.** Lợi ích ở N-CMAPSS và pin chỉ xuất hiện ở seed dùng để phát triển. Đây là
   bài học về phương pháp, đáng ghi vào bài.
