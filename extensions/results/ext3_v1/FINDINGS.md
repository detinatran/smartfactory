# EXT3 — conformal theo shift và điểm kết hợp — kết quả (2026-11-10)

Protocol: `research/selective-rul/EXT3_PROTOCOL.md`, chốt trước khi chạy. Code: `scripts/ext3_conformal.py`, `scripts/ext3_summary.py`.
- Quy mô: 240 run, đúng thiết lập của bài ESANN (GRU/TCN × seed 42–51 × 4 họ dữ liệu); 13 440 hàng, 0 run lỗi; CPU trên van.
- Đối chiếu: risk của `full` ở đây là 9.37 / 12.49 / 23.29 / 4.47, trùng CPU replay của bài.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| Điểm | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| ensemble_std | 11.16 | 19.51 | 24.08 | 5.52 |
| laplace_ev | 11.86 | 19.93 | 23.99 | 5.92 |
| **full (bài ESANN)** | **9.37** | 12.49 | 23.29 | 4.47 |
| gauss_bias_aware | 10.24 | 12.19 | 23.49 | **3.56** |
| rank_avg | 9.60 | **12.17** | **23.28** | 4.01 |
| conformal | 10.71 | 12.98 | 23.96 | 3.85 |
| conformal_w | 11.44 | 17.01 | 25.18 | 6.45 |
| mondrian | 10.78 | 12.97 | 23.61 | 3.93 |

## Họ kiểm định chính: điểm mới trừ `full` (Holm trên 12 kiểm định, 20 cặp)
| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| conformal_w | +2.06 (0/20), p_H < 1e-4 | +4.52 (0/20), p_H < 1e-4 | +1.89 (0/20), p_H < 1e-4 | +1.98 (0/20), p_H < 1e-4 |
| mondrian | +1.41 (0/20), p_H < 1e-4 | +0.47 (5/20), p_H = 0.022 | +0.32, p_H = 0.33 | −0.54 (15/20), p_H = 0.11 |
| rank_avg | **+0.22 (3/20), p_H = 0.0005** | **−0.32 (19/20), p_H < 1e-4** | −0.01, p_H = 0.93 | **−0.45 (20/20), p_H < 1e-4** |

Số trong ngoặc là số cặp điểm mới thắng.

- **Không có điểm nào tốt hơn bài ESANN trên mọi dataset.**
- `rank_avg` (ghép bộ phân loại của bài với Gaussian có bias):
  - tốt hơn có ý nghĩa sau Holm trên N-CMAPSS và pin;
  - kém hơn 0.22 điểm trên C-MAPSS;
  - hoà trên vòng bi.

  Ý tưởng này **post hoc**, vì sinh ra sau khi thấy unc_v1. Muốn viết là kết quả thì cần seed mới, tức là phải train lại backbone.
- `conformal_w` **thất bại nặng.**
  - Bộ phân loại miền phân biệt được unit đánh giá với unit calibration: khác điều kiện vận hành ở vòng bi, khác batch ở pin,
    và ở mức độ thấp hơn cả trên C-MAPSS và N-CMAPSS.
  - Khi đó trọng số suy biến: kích thước mẫu hiệu dụng chỉ còn 39% (vòng bi) đến 76% (pin), và q(x) = +∞ cho gần như mọi cửa sổ ở
    vòng bi và pin. Độ phủ thành 100% và quy tắc certify có coverage 0%.
  - Đây là thất bại kinh điển của weighted conformal khi không có vùng chồng lấn (overlap) giữa hai miền.
- `mondrian` (theo bin của RUL dự đoán) không giúp gì so với conformal thường.

## Độ phủ của biên conformal (mục tiêu 90%): sửa lại nhận định của EXT2
| | sạch | lỗi, mọi cửa sổ | **lỗi, chỉ cửa sổ bị ảnh hưởng** |
|---|---|---|---|
| C-MAPSS | 88.9 | 86.9 | **82.5** |
| N-CMAPSS | 89.4 | 90.3 | 91.8 |
| Vòng bi | 81.3 | 81.5 | 82.8 |
| Pin | 91.4 | 90.3 | 95.6 |

- **C-MAPSS:** trên các cửa sổ thực sự bị lỗi, độ phủ giảm khoảng 6 điểm so với khi sạch. Nhận định "biên giữ độ phủ dưới lỗi"
  của EXT2 là **phóng đại**, vì độ phủ ở đó tính trên mọi cửa sổ, mà chỉ 33–46% số cửa sổ bị lỗi.
- N-CMAPSS và pin vẫn giữ độ phủ.
- Vòng bi hụt ngay từ khi sạch, do shift giữa các điều kiện vận hành.

## Phân tích phụ
- **Laplace với prior chọn bằng evidence:** prior khoảng 50–80, nhưng vẫn kém spread trên C-MAPSS (+0.70), N-CMAPSS (+0.42)
  và pin (+0.40); ngang trên vòng bi. Như vậy Laplace chuẩn cũng thất bại, củng cố luận điểm "variance thất bại có hệ thống".
- **Ngưỡng frozen (triển khai thực tế):**
  - Pin: conformal 1.48% ở coverage 71.1%, gauss_bias_aware 1.50% ở 71.4%; `full` là 2.64% ở 71.7%. Tốt hơn ở cùng coverage.
  - N-CMAPSS: gauss_bias_aware 7.83% ở 83.5%; `full` là 8.92% ở 84.4%.
  - C-MAPSS và vòng bi: `full` tốt hơn.
- **Quy tắc certify** (nhận nếu chặn trên ≤ δ):
  - Pin: 2.0% risk ở coverage 77% dưới lỗi. Coverage đạt được cao hơn ngưỡng frozen của `full` (71.7%) và risk thấp hơn (2.64%).
  - N-CMAPSS: 1.8% ở 63%. C-MAPSS: 5.7% ở 54%. Vòng bi: 13.9% ở 35%.
  - Đây là một điểm vận hành tự nhiên, không cần chọn trước mức coverage.

## Kết luận
1. Mục tiêu "tốt hơn bài ESANN" **chưa đạt một cách đồng đều**. Gần nhất là `rank_avg`: thắng 2/4, thua nhẹ 1/4, và post hoc.
2. Weighted conformal không dùng được khi unit đánh giá khác hẳn unit calibration. Muốn sửa cần bộ phân loại miền chỉ nhìn các
   đặc trưng nhạy với lỗi, không nhìn đặc trưng của unit. Đó là một protocol mới, không chỉnh trên kết quả này.
3. Biên conformal có bias **không** giữ độ phủ trên cửa sổ bị lỗi ở C-MAPSS (82.5%). Bài mở rộng phải viết đúng như vậy.
4. Phần vững: mọi uncertainty dạng variance thất bại, kể cả Laplace đã tune bằng evidence. Điểm có bias tốt hơn hẳn. Trên pin
   và N-CMAPSS, họ điểm có bias còn tốt hơn bộ phân loại của bài, cả ở coverage matched lẫn ngưỡng frozen.
