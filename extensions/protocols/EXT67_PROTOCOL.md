# EXT67: cổng calibration chọn giữa ELPIS-J V13 (EMA) và V20 (SIGReg), kiểm lại trên seed 222–251 (THĂM DÒ, không tiêu α). Chốt ngày 2026-11-19, TRƯỚC khi chạy V20 trên các seed này

**Lý do (ghi rõ là post hoc):**
- Sau EXT66 (seed 162–171), mình thử một cổng theo từng deployment: dùng V20 nếu risk calibration của selector mà ELPIS-G chọn trên V20 thấp hơn trên V13r, ngược lại dùng V13r (hoà thì V13r).
- Quy tắc này không mới và không được chỉnh: nó đúng là quy tắc chọn predictor của ELPIS-JG (EXT64), chỉ thay predictor gộp bằng V20. Chỉ dùng unit calibration, không dùng test.
- Kết quả post hoc trên seed 162–171 (risk ELPIS-G, cổng − V13r): C-MAPSS −0.69 (10/10, p = 0.002, chọn V20 ở 59% deployment); pin +0.07 (3/10, p = 0.38, chọn V20 13%); vòng bi −0.92 (6/10, p = 0.07, chọn V20 35%).
- Vì quy tắc được nhìn trên chính dữ liệu đó, kết quả này chỉ là giả thuyết. Cần kiểm trên seed không dùng để phát triển V20 hay cổng.

## Mẫu kiểm
- Seed 222–251 (30 seed) của vòng 10 (EXT59): run `conf222_cmapss` + `conf222_ext`, C-MAPSS FD001–4, `battery_cv`, vòng bi (6 fold), GRU và TCN, 900 run.
- Các seed này đã dùng cho vòng 10 (V13 so với OSC), nhưng **chưa từng dùng** để phát triển V20 hay cổng V13/V20. V13 trên chúng đã có sẵn (`all_ext59_elpisj.csv.gz`, EXT59).
- **Chỉ chạy mới V20** (`ELPISJ_SSL=sigreg`, `ELPISJ_FREEZE=1`, `ELPISJ_BIASCAL=1`, `scripts/ext50.py` của EXT66, không sửa) trên 900 run này. 6 tiến trình GPU, chạy bù 1 lượt.
- Cổng dùng V13 của EXT59. Kiểm tra trên EXT66 cho thấy V13 chạy lại bằng script mới lệch < 0.1 điểm so với bản cũ.

## Kiểm định (đặt trước)
- **Chính:** risk ELPIS-G (`rpt_auto_d`) dưới 6 lỗi, coverage 90% matched, **cổng − V13** theo từng họ; đơn vị seed (trung bình GRU/TCN, fold vòng bi có trọng số theo số unit, subset C-MAPSS và fold pin trọng số bằng nhau); 30 seed; Wilcoxon hai phía.
  - "**Tốt hơn**" ở một họ: trung bình < 0 và p < 0.05.
  - "**Không kém**" ở một họ: cận trên CI 90% bootstrap theo seed ≤ +0.3.
  - Cổng được coi là **cải thiện ELPIS-J** nếu tốt hơn ở ít nhất 2 họ và không kém ở cả 3 họ.
- **Phụ:** V20 − V13 (không cổng); tỷ lệ chọn V20; lỗi bi quan; RMSE; ELPIS-G − OSC trên V20; effective rank của V20.

## Báo cáo
Mọi kết quả đều được báo cáo. Kết quả là thăm dò (quỹ α đã hết), không đổi phát biểu xác nhận nào của bài. Nếu đạt, có thể viết trong bài như "a calibration gate between the EMA and SIGReg variants of ELPIS-J (exploratory, replicated on 30 seeds not used to design it)".

## Kết quả, ngày 2026-11-19 (thăm dò): ĐẠT tiêu chí "cải thiện ELPIS-J"
- 900/900 run V20, không lỗi, đủ 30/30 seed.
- **Chính, risk ELPIS-G cổng − V13:** C-MAPSS −1.03 (30/30, p < 1e-8, CI hi −0.89) tốt hơn; pin +0.03 (14/30, p = 0.41, CI hi +0.08) không kém; vòng bi −1.20 (21/30, p = 0.002, CI hi −0.66) tốt hơn. Tốt hơn ở 2 họ và không kém ở cả 3, nên đạt.
- **Phụ:** lỗi bi quan +1.24 (C-MAPSS), +0.20 (pin), +1.67 (vòng bi); RMSE không kém; tỷ lệ chọn V20 65% / 17% / 46%; V20 thay hẳn V13: −1.21 / +1.89 / −0.37. Chi tiết: `extensions/results/ext67/FINDINGS.md`.
