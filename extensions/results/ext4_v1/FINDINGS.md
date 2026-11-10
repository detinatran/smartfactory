# EXT4 — chọn điểm số theo calibration — kết quả (2026-11-10)

Protocol: `research/selective-rul/EXT4_PROTOCOL.md`, chốt trước khi chạy. Quy tắc này được đề xuất sau khi đã thấy unc_v1 và
EXT3, nên mang tính post hoc. Code: `scripts/ext4_select.py`, `scripts/ext4_summary.py`.
Quy mô: 240 run của bài ESANN, 9 600 hàng, 0 run lỗi.

## Tỷ lệ run chọn Gaussian có bias (`select_cal`)
C-MAPSS 22%, N-CMAPSS 85%, vòng bi 54%, pin 60%.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| full (bài ESANN) | **9.37** | 12.49 | 23.29 | 4.47 |
| gauss_bias_aware | 10.24 | 12.19 | 23.49 | **3.56** |
| select_cal | 9.54 | 12.20 | 23.35 | 3.82 |
| select_cal3 (+ rank_avg) | 9.58 | **12.15** | **23.23** | 3.90 |

## Họ kiểm định chính: `select_cal` trừ `full` (Holm trên 4 kiểm định, 20 cặp)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| Hiệu (điểm) | **+0.17** | −0.29 | +0.07 | −0.64 |
| Thắng / hoà | 2 / 6 | 12 / 3 | 8 / 1 | 10 / 8 |
| p Holm | **0.0075** | 0.030 | 0.57 | 0.023 |

**Tiêu chí "tốt hơn bài ESANN" đã chốt trước: KHÔNG ĐẠT.** Quy tắc tốt hơn có ý nghĩa trên N-CMAPSS và pin, nhưng kém hơn có ý
nghĩa trên C-MAPSS. Các biến thể phụ cho cùng một mẫu: `select_cal_90` +0.19 / −0.28 / +0.15 / −0.66; `select_cal3` +0.20 /
−0.34 / −0.06 / −0.56.

## Ngưỡng frozen: risk / coverage (%)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| full | 9.09 / 88.6 | 8.92 / 84.4 | 21.61 / 82.3 | 2.64 / 71.7 |
| select_cal | 9.33 / 88.8 | 7.91 / 83.6 | 22.30 / 84.5 | 1.86 / 73.3 |

## Kết luận
1. **Không thay phương pháp của bài ESANN.** Không có biến thể nào (EXT3: rank_avg, conformal; EXT4: các quy tắc chọn) tốt hơn
   `full` trên cả 4 dataset. C-MAPSS luôn là nơi `full` thắng.
2. Mẫu kết quả ổn định qua mọi biến thể:
   - bộ phân loại tốt nhất trên C-MAPSS;
   - Gaussian có bias tốt nhất trên N-CMAPSS và pin;
   - vòng bi hoà.

   Chênh lệch giữa hai điểm có bias này nhỏ (dưới 1 điểm), so với khoảng cách 2–7 điểm giữa chúng và nhóm variance.
3. Chọn theo calibration không phân biệt đủ tốt trên C-MAPSS, vì tập calibration chỉ có 10–26 unit; 22% số run chọn sai hướng.
4. Đề xuất **dừng tìm cách "thắng bài ESANN" trên cùng 240 run này.** Mỗi lần thử thêm lại tăng rủi ro chọn lọc sau khi đã thấy
   kết quả (forking paths). Nếu muốn có một phương pháp mới cho bài mở rộng, cần protocol mới và seed mới (train lại backbone).
