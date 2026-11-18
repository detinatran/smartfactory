# EXT29: vòng xác nhận 5 (seed 112–121, α_5 = 0.0015625), 2026-11-15
Protocol: `research/selective-rul/EXT29_PROTOCOL.md`.
- 260 ensemble mới, train trên GPU. 3 run HDD được train lại vì hết RAM; đã ghi sai lệch trong protocol.
- 0 lỗi đánh giá.

## Chính: Δ`excess_A` (`rpt_auto` − `full`), 6 lỗi gốc, Holm trên 5 ở α_5
| Họ | Δ | Cặp tốt hơn / kém hơn | p_Holm | Kết luận |
|---|---|---|---|---|
| C-MAPSS | −0.89 | 19/1 | 2e-5 | có ý nghĩa |
| N-CMAPSS | −0.80 | 14/1 | 0.0024 | n.s. ở α_5 |
| Vòng bi | −0.18 | 13/7 | 0.11 | n.s. |
| Pin | −1.96 | 19/0 | 5e-4 | có ý nghĩa |
| HDD | +0.11 | 0/2 | 0.18 | n.s. |

**XÁC NHẬN:** giảm có ý nghĩa ở 2 họ, không họ nào tăng có ý nghĩa. Tổng α đã tiêu = 0.0484375.

## Phụ
- **S2, `risk_matched`, `rpt_auto` − `full`:** −0.49, −0.36, −0.75, −0.62 (có ý nghĩa ở cả 4 họ); HDD +0.04 (n.s.).
- **S3, `rpt_L` − `full` trên `excess_A`:**
  - C-MAPSS −1.06\*, N-CMAPSS −0.96\*, pin −2.01\* (có ý nghĩa);
  - vòng bi −0.19 (n.s.);
  - HDD +1.22, kém có ý nghĩa (0/20).
- **S1, lỗi `mask`:**
  - Tỷ lệ lạc quan của predictor trên A, sạch → mask: C-MAPSS 8.7 → 37.5, N-CMAPSS 21.6 → 34.0, vòng bi 52.7 → 56.1,
    pin 9.7 → 59.4, HDD 92.2 → 94.4 (%).
  - Δ`excess_A` của `rpt_auto` − `full` chỉ có ý nghĩa trên C-MAPSS (−1.43, 19/20). Các họ khác n.s.
  - **Diễn giải:** window bị che giấu suy giảm trông như window khỏe, nên giả định (A) bị vi phạm. Không selector nào chỉ dựa trên đầu vào phát hiện được.
- **S4, bảo trì (ngưỡng frozen):**
  - Tỷ lệ hỏng gần 0 với mọi điểm số ở C-MAPSS và pin; chi phí chủ yếu do kiểm tra.
  - `rpt_auto` tốn **hơn** `full` và `gauss_bias_aware` trên C-MAPSS (+0.008 và +0.012 ở c_i = 0.01, 19/20 cặp; vì kiểm tra nhiều hơn).
  - Các họ khác n.s. so với `full`. `rpt_auto` tốt hơn `ensemble_std` trên N-CMAPSS và HDD.
  - **Kết quả âm:** risk lạc quan thấp hơn không chuyển thành chi phí bảo trì thấp hơn, với chính sách đơn giản này.
- **Số run chọn `rpt_L`:** C-MAPSS 59/80, N-CMAPSS 15/20, vòng bi 89/120, pin 19/20, HDD 2/20.
