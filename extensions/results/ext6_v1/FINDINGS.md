# EXT6: sửa đầu vào nhờ dư thừa giữa các cảm biến, rồi mới chọn lọc. Kết quả ngày 2026-11-11

Protocol: `research/selective-rul/EXT6_PROTOCOL.md`, chốt trước khi chạy. Code: `scripts/ext6_repair.py`, `scripts/ext6_summary.py`.
- Quy mô: 240 run của bài ESANN, 13 440 hàng, 0 run lỗi. CPU trên van.
- Đối chiếu: `full` cho 9.37 / 12.49 / 23.29 / 4.47, trùng với EXT3–EXT5.
- Mọi kết quả là **exploratory**: đây là vòng thứ 5 dùng lại cùng tập đánh giá.

## Phát hiện kênh lỗi
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Báo động nhầm khi sạch | 0.7% | 2.2% | **12.6%** | 1.0% |
| Độ nhạy: bias / noise / spike | 96 / 91 / 91% | 97 / 98 / 99% | 98 / 96 / 94% | 85 / 84 / 69% |
| Độ nhạy: stuck / gain | 60 / 42% | 88 / 73% | 76 / 70% | 11 / 9% |
| Độ nhạy: intermittent | 100% | 100% | 100% | 100% |

- Khi có lỗi, tỷ lệ báo nhầm trên các kênh không bị lỗi tăng lên: 1–42%, tuỳ dataset và loại lỗi. Lý do là kênh lỗi làm
  hỏng giá trị tái tạo của các kênh còn lại.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| Điểm | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| ensemble_std | 11.16 | 19.51 | 24.08 | 5.52 |
| impact (không học) | 9.45 | 16.02 | 25.22 | 5.49 |
| gauss_bias_aware | 10.24 | 12.19 | 23.49 | **3.56** |
| full (bài ESANN) | 9.37 | 12.49 | 23.29 | 4.47 |
| **full_recon** | **9.12** | 12.53 | 23.25 | 4.41 |
| repair_full (dự đoán đã sửa) | 9.34 | 12.04 | 20.83 | 4.36 |

## Họ kiểm định chính (Holm trên 8 kiểm định, mỗi dataset 20 cặp)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| A: `full_recon` trừ `full` | **−0.25 (20/20), p_H < 1e-4** | +0.03, p_H = 1 | −0.04, p_H = 1 | −0.05, p_H = 1 |
| B: `repair_full` trừ `full` | −0.03, p_H = 1 | −0.46, p_H = 1 | −2.45 (17/20), p_H = 0.0004 | −0.11, p_H = 1 |
| B: thay đổi tỷ lệ lỗi bi quan | −0.56 | **+4.48** | **+5.64** | **+2.35** |

- **A, `full_recon`: ĐẠT tiêu chí đã chốt.**
  - Tốt hơn ở C-MAPSS trên cả 20/20 cặp; không kém có ý nghĩa ở đâu.
  - Nhưng hiệu số nhỏ: 0.25 điểm, chỉ ở C-MAPSS. Ba dataset còn lại hoà.
- **B, `repair_full`: KHÔNG ĐẠT.**
  - Điều kiện chống gian lận đã bắt được vấn đề: cái lợi trên vòng bi (−2.45) đến từ việc hạ dự đoán. Tỷ lệ lỗi bi quan
    tăng 5.6 điểm và RMSE tệ hơn (0.295 so với 0.266).
  - Trên vòng bi, bộ tái tạo báo nhầm 12.6% ngay cả khi sạch, vì điều kiện vận hành ở tập đánh giá khác lúc fit.

## Phân tích phụ (chưa hiệu chỉnh)
- **Bộ dự đoán đã sửa, ở coverage 100%, dưới lỗi:**
  - C-MAPSS **tốt hơn thật**: RMSE 19.76 so với 20.45; lỗi lạc quan 11.8% so với 12.2%; lỗi bi quan cũng giảm.
  - N-CMAPSS, vòng bi và pin: tệ hơn (RMSE tăng, lỗi bi quan tăng).
  - Trên dữ liệu sạch: gần như không đổi ở cả 4 dataset, nên việc sửa không gây hại khi không có lỗi.
- **`impact` so với spread:** tốt hơn mạnh ở C-MAPSS (−1.7, 20/20) và N-CMAPSS (−3.5, 20/20); kém hơn ở vòng bi (+1.1);
  ngang ở pin.
  - Đây là điểm không học tốt nhất từ trước đến nay. Trên N-CMAPSS, LOCO ở EXT5 chỉ được −1.3.
- **Ngưỡng frozen, `full_recon` so với `full`:**
  - Risk thấp hơn ở cả 4 dataset: 8.46 so với 9.09; 8.17 so với 8.92; 21.51 so với 21.61; 2.58 so với 2.64.
  - Nhưng coverage cũng thấp hơn: 86.8 so với 88.6; 83.0 so với 84.4; 82.6 so với 82.3; 68.6 so với 71.7.
  - Vì vậy chưa phải là thắng rõ ràng.

## Kết luận
1. **`full_recon` là biến thể đầu tiên đạt tiêu chí "tốt hơn bài ESANN"**, nhưng **chưa nên thay vào bài**:
   - Đây là vòng thứ 5 thử trên cùng tập đánh giá, nên có rủi ro forking paths. Holm chỉ hiệu chỉnh bên trong vòng này.
   - Hiệu số chỉ 0.25 điểm, và chỉ có ở C-MAPSS.
   - Protocol EXT6 đã ghi trước: nếu đạt tiêu chí thì **bắt buộc xác nhận bằng seed mới** trước khi viết thành kết quả.
2. Đề xuất vòng xác nhận EXT6-C:
   - C-MAPSS FD001–4 × GRU/TCN × seed mới 52–56: 40 backbone, train lại bằng CPU.
   - Chỉ kiểm định đúng một giả thuyết: `full_recon` trừ `full` trên C-MAPSS, Wilcoxon, α = 0.05, chốt trước.
3. Bài học về cơ chế: sửa bằng dư thừa giữa các cảm biến có ích khi các cảm biến tương quan mạnh và điều kiện vận hành ổn
   định, như ở C-MAPSS. Nó có hại khi điều kiện giữa fit và đánh giá khác nhau, như ở vòng bi.
