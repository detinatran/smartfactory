# EXT46: baseline "sửa kênh + conformal cục bộ" (dựng theo tinh thần RobustUQ, Wang et al. 2025, có rút gọn). Họ phụ đặt trước; seed 162–171

Risk ở coverage 90% (matched), trung bình 6 lỗi, đơn vị %; hiệu tính bằng điểm.

| Họ | RPT-D | repair_lconf | Hiệu RPT-D − repair_lconf | Seed RPT-D tốt hơn | p_Holm |
|---|---|---|---|---|---|
| C-MAPSS | 7.86 | 9.58 | −1.72 | 10/10 | 0.0098 |
| N-CMAPSS | 8.00 | 16.48 | −8.48 | 10/10 | 0.0098 |
| Pin | 2.35 | 5.19 | −2.84 | 10/10 | 0.0098 |
| HDD | 15.55 | 19.88 | −4.33 | 10/10 | 0.0098 |
| Vòng bi | 23.76 | 21.42 | +2.35 | 1/10 | 0.0098 |

- **Vòng bi: baseline tốt hơn RPT-D.** Bước sửa kênh làm tỷ lệ lạc quan của predictor giảm từ 26.3% xuống 22.4%. Điều này khớp với kết quả FDI + điền giá trị đã có trong bài (FDI cộng RPT trên vòng bi đạt 18.8%).
- **N-CMAPSS:** score kNN cục bộ yếu, vì bước sửa kênh làm predictor lạc quan hơn: 16.2% lên 19.9%.
- **Kết luận:** RPT-D tốt hơn có ý nghĩa ở 4/5 họ. Trên vòng bi, phát hiện và sửa kênh lỗi là bổ sung hữu ích. Bài báo cáo trung thực cả hai chiều.
