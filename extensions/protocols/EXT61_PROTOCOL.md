# EXT61: sửa RAMP trên vòng bi, hiệu chỉnh mức lạc quan và ensemble gộp (THĂM DÒ / PHÁT TRIỂN, không tiêu α). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Lý do:** ở vòng 10 (seed 222–251), trên vòng bi hệ thống RAMP-V13 + ELPIS-G có kết quả như sau so với hệ thống mặc định + ELPIS-G:
- risk lạc quan cao hơn +1.99 điểm (không có ý nghĩa);
- lỗi bi quan ít hơn 5.89 điểm;
- RMSE tốt hơn (0.265 so với 0.275).

Như vậy, phần RAMP thua có thể chủ yếu do điểm làm việc lệch về phía lạc quan, chứ không do predictor kém. Bias correction bằng trung bình (V13) chưa đủ, vì phân phối phần dư bị lệch.

## Biến thể
Tất cả dùng `scripts/ext50.py`, chỉ khác biến môi trường. Mọi hiệu chỉnh chỉ dùng tập validation, không đụng tới test.

**V16 (hiệu chỉnh mức lạc quan):** `RAMP_FREEZE=1`, `RAMP_PMATCH=1`, không bias correction trung bình.
- Dịch toàn bộ dự đoán RAMP đi một hằng số, sao cho tỷ lệ lạc quan có trọng số (err > δ) trên validation sạch bằng đúng tỷ lệ đó của ensemble mặc định của cùng run.
- Đây là so sánh ở cùng điểm làm việc: khi RAMP lạc quan đúng bằng predictor mặc định, độ chính xác cao hơn của nó có thành risk thấp hơn không?

**V17 (đổi cách làm, ensemble gộp):** `RAMP_FREEZE=1`, `RAMP_BIASCAL=1`, `RAMP_HYBRID=1`.
- Gộp 5 thành viên mặc định của run với 5 thành viên RAMP-V13, thành ensemble 10 thành viên.
- Selector (OSC và ELPIS) được train trên ensemble gộp này.

## Dữ liệu
- **Phát triển:** chỉ vòng bi, seed 162–171, run `conf162_ext`, 120 run cho mỗi biến thể. Hai biến thể chạy song song, mỗi biến thể 4 tiến trình (tổng 8).
- **So sánh trên cùng seed:**
  - mặc định: EXT40;
  - V13: EXT56.

## Phân tích (giống EXT52/EXT56)
Các chỉ số:
- RMSE trên dữ liệu sạch;
- tỷ lệ lạc quan và tỷ lệ bi quan của predictor dưới 6 lỗi;
- risk của RPT-D (ELPIS-G) dưới lỗi, so với RPT-D trên predictor mặc định: hiệu theo seed, số seed tốt hơn, Wilcoxon.

## Tiêu chí "sửa được vòng bi" (đặt trước)
- Hiệu trung bình RPT-D so với mặc định ≤ 0, với ≥ 6/10 seed tốt hơn.
- Tỷ lệ bi quan dưới lỗi không cao hơn mặc định quá 1 điểm.

## Nếu có biến thể đạt tiêu chí
1. Chạy biến thể đó trên C-MAPSS và pin (seed 162–171), so với V13 và mặc định, để chắc không làm hỏng hai họ đó.
2. Lặp lại trên seed mới 222–251 (vòng bi, các run vòng 10 đã có), so với hệ thống mặc định của EXT59.

Đây là bước lặp lại có tính thăm dò, vì ngân sách α đã hết. Báo cáo bằng Wilcoxon theo seed, ghi rõ là thăm dò.

## Báo cáo
Mọi kết quả đều được báo cáo, kể cả khi không biến thể nào đạt.

## Kết quả, ngày 2026-11-17 (thăm dò): KHÔNG biến thể nào đạt tiêu chí đặt trước
Vòng bi, seed 162–171, 120/120 run mỗi biến thể, không có run lỗi. Hiệu RPT-D so với mặc định:

| | Risk lạc quan | Lỗi bi quan | RMSE |
|---|---|---|---|
| V13 | +1.08 (4/10 seed tốt hơn) | −5.29 | 0.265 |
| V16 (hiệu chỉnh mức lạc quan) | +0.09 (5/10, p = 1) | −2.41 (7/10) | 0.276 |
| V17 (ensemble gộp) | +0.14 (5/10, p = 1) | −3.15 (9/10, p = 0.004) | 0.261, tốt nhất |

RMSE của predictor mặc định là 0.280.

**Nhận xét:**
- Cả hai biến thể kéo phần thua về gần như hòa trên risk lạc quan, nhưng không thắng.
- V17 là biến thể duy nhất không kém trên chỉ số nào: RMSE tốt nhất, ít lỗi bi quan hơn có ý nghĩa, risk lạc quan hòa.
- ELPIS-G vẫn tốt hơn OSC trên cả hai biến thể (10/10 seed).
- Seed 162–171 đã được dùng để chọn nhiều biến thể, nên mọi bước tiếp theo phải dùng seed mới (xem EXT62).
