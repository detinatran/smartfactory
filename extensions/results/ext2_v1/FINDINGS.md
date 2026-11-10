# EXT2 pilot — kết quả (2026-11-10)

Protocol: `research/selective-rul/EXT2_PILOT_PROTOCOL.md`, chốt trước khi chạy. Code: `scripts/ext2_pilot.py`,
`src/selective_rul/ext_backbones.py`, tóm tắt `scripts/ext2_summary.py`.
Quy mô: 25 run (C-MAPSS FD001–4 × 5 seed, N-CMAPSS × 5 seed) × 3 backbone × 8 điểm × 7 scenario = 4 200 hàng; 0 run lỗi.
Chạy CPU trên van. Train FiLM + Lipschitz mất trung vị 164 s mỗi run C-MAPSS và 22 s mỗi run N-CMAPSS.
Mọi kết quả là **exploratory**. Chỉ có 5 cặp seed, nên p nhỏ nhất của Wilcoxon là 0.0625; "5/5" nghĩa là cả 5 seed cùng chiều.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| Điểm | C-MAPSS GRU | FiLM | Lipschitz | N-CMAPSS GRU | FiLM | Lipschitz |
|---|---|---|---|---|---|---|
| ensemble_std | 9.38 | 9.39 | 10.83 | 19.03 | 17.94 | 19.84 |
| mc_dropout | 9.97 | 9.64 | 11.15 | 19.88 | 18.18 | 20.46 |
| mc_ensemble | 9.36 | 9.43 | 10.93 | 19.03 | 18.07 | 19.80 |
| laplace | 10.11 | 9.58 | 10.58 | 19.57 | 18.48 | 20.75 |
| gauss_unbiased | 10.38 | 10.08 | 11.59 | 16.59 | 15.08 | 18.92 |
| gauss_bias_aware | 8.44 | 8.08 | 9.22 | 12.15 | **10.15** | 13.18 |
| conformal | 8.96 | 8.58 | 9.76 | 12.55 | 10.36 | 13.34 |
| full (classifier) | 7.77 | **7.48** | 8.70 | 12.23 | 10.30 | 13.24 |

## Theo giả thuyết
- **H1 (FiLM: member chung trọng số thì dịch chuyển chung nhiều hơn): ỦNG HỘ.**
  - shift/spread: C-MAPSS 2.69 so với 1.98 của GRU; N-CMAPSS 3.85 so với 2.15 (5/5 seed ở cả hai).
  - Spread của FiLM loại ít lỗi lạc quan do fault hơn: 21.2% so với 25.8% (C-MAPSS, 5/5) và 15.3% so với 19.8% (N-CMAPSS, 4/5).
  - Tức là ensemble ngầm (implicit) rẻ hơn nhưng "mù" hơn với common shift, đúng như Prop. 3 dự đoán.
- **H2 (Lipschitz giảm độ dịch do lỗi): CHỈ MỘT PHẦN, không có lợi.**
  - C-MAPSS: độ dịch nhỏ hơn 6% (0.083 so với 0.088 cap, 5/5), nhưng RMSE sạch tệ hơn (0.130 so với 0.123).
  - N-CMAPSS: độ dịch lớn hơn (0.169 so với 0.146).
  - Risk của Lipschitz cao hơn GRU ở mọi điểm.
  - ε = 0.1 chốt trước và không tune, nên chưa loại trừ được khả năng cấu hình khác tốt hơn.
- **H3 (điểm có bias tốt hơn họ variance): ỦNG HỘ trên cả 6 tổ hợp backbone × dataset.**
  - Laplace không tốt hơn spread, trừ một ngoại lệ: Lipschitz trên C-MAPSS (−0.25 điểm, 5/5).
  - Các điểm epistemic giảm khi lỗi đẩy dự đoán lên. Trên C-MAPSS: ρ là −0.26 đến −0.46 với spread, và **−0.39 đến −0.59 với
    Laplace**. Cách giải thích (chưa kiểm chứng): std của Laplace tỷ lệ với σ'(z), mà σ'(z) nhỏ lại khi dự đoán lệch về
    phía cap.
- **H4 (độ phủ conformal giảm khi có lỗi): YẾU hoặc KHÔNG.**
  - Biên một phía có bias e ≤ b + q s đạt độ phủ 89–92% trên dữ liệu sạch.
  - Dưới 6 lỗi chưa thấy: 87.6–88.2% (C-MAPSS, giảm 1–1.5 điểm) và 91.7–93.3% (N-CMAPSS, không giảm).
  - Vậy biên này gần như vẫn giữ được độ phủ dưới lỗi chưa thấy, vì b(x) và s(x) thích nghi theo đầu vào. Đây là điểm dương.
    Lưu ý: các cửa sổ trong cùng unit phụ thuộc nhau, nên "90%" chỉ là xấp xỉ.
- **Ngưỡng frozen (triển khai thực tế):**
  - conformal và gauss_bias_aware giữ coverage khoảng 90% trên C-MAPSS dưới lỗi, trong khi spread chỉ còn 77–80%.
  - Trên N-CMAPSS: 79–85% so với 67–69%.
  - Risk frozen thấp nhất trên N-CMAPSS là conformal: 6.2–7.6%.

## Lắp ghép
- Tổ hợp nổi bật: **FiLM-GRU (backbone rẻ, 1 mạng) + điểm có bias.**
  - C-MAPSS: full 7.48 so với 7.77 của GRU + full.
  - N-CMAPSS: gauss_bias_aware 10.15 so với 12.15.
  - Spread của FiLM tệ hơn, nhưng phần chấm điểm có bias bù lại và còn tốt hơn.
- Theo protocol, **không công bố tổ hợp này như kết quả chính**. Cần một vòng xác nhận với seed 47–51 và protocol riêng chốt trước.

## Đề xuất bước tiếp
1. Vòng xác nhận seed 47–51, chỉ cho C-MAPSS và N-CMAPSS: FiLM + {gauss_bias_aware, conformal, full} so với GRU + cùng điểm.
2. Mở rộng sang vòng bi và pin.
3. Câu chuyện cho bài mở rộng:
   - variance/epistemic (ensemble thường và ngầm, MC dropout, Laplace) thất bại có hệ thống dưới lỗi cảm biến;
   - ensemble ngầm còn thất bại nặng hơn, đúng như lý thuyết;
   - phân phối dự đoán có bias kèm conformal cho biên một phía gần như giữ được độ phủ, và tổ hợp với ensemble ngầm rẻ thì tốt nhất.
