# EXT59: vòng xác nhận 10 (vòng cuối của ngân sách α). RPT-D (ELPIS-G) có còn giúp khi predictor là RAMP-V13, tức predictor được pretrain kiểu JEPA? Chốt ngày 2026-11-17, TRƯỚC khi train và chạy

**Lý do:** các kết quả thăm dò EXT50, EXT51 và EXT56 cho thấy RPT-D tốt hơn OSC trên mọi biến thể predictor RAMP, khoảng −0.5 đến −1.0 điểm. Câu hỏi cần xác nhận: phương pháp huấn luyện selector có bổ sung cho một predictor mạnh hơn, đã được pretrain theo hướng chống lỗi, hay không.

## Hệ thống (cố định, không đổi sau khi chốt)
- **Predictor RAMP-V13:** `scripts/ext50.py` với `RAMP_FREEZE=1` và `RAMP_BIASCAL=1`.
  - Stage I: masked JEPA căn theo horizon, 20 epoch.
  - Stage II: đóng băng encoder, chỉ train các module head.
  - Sau train: trừ phần dư trung bình đo trên tập validation.
- **Selector:** `full` (OSC) và `rpt_auto_d` (ELPIS-G), giống EXT38; cả hai dùng chung predictor RAMP-V13.

## Dữ liệu
- **Seed mới:** 222–251 (30 seed).
- **Họ:** C-MAPSS FD001–4, `battery_cv` (xoay unit), vòng bi (6 điều kiện).
- **Train:** cho mỗi seed, train trước một ensemble mặc định để có split, normalizer và config, giống vòng 8. Sau đó `ext50.py` train predictor RAMP-V13 trên cùng split.
- **Ngoài phạm vi:** N-CMAPSS và ổ cứng. N-CMAPSS có tập fit quá lớn cho pretrain JEPA. Trên ổ cứng, D < 0 ở mọi deployment nên RPT-D luôn chọn OSC.

## Kiểm định xác nhận (α_10 = 0.00009765625 / 2 = 0.000048828125; tổng α sau vòng = 0.049951171875)
- **Endpoint chính:** risk ở coverage 90% matched, trung bình 6 lỗi, `rpt_auto_d` − `full` trên predictor RAMP-V13.
- **Đơn vị:** seed. Lấy trung bình GRU và TCN, trung bình qua các fold; vòng bi có trọng số theo số unit.
- **Kiểm định:** Wilcoxon signed-rank hai phía trên 30 seed, Holm trên 3 họ.
- **Thành công (cùng tiêu chí với các vòng trước):**
  - giảm có ý nghĩa ở ít nhất 1 họ;
  - không họ nào tăng có ý nghĩa;
  - hiệu trung bình ≤ +0.3 ở mọi họ;
  - tỷ lệ bi quan tăng ≤ 1 điểm.

## Phụ (mô tả, ngoài ngân sách α)
- RAMP-V13 + RPT-D so với hệ thống mặc định + RPT-D (dùng EXT48 không được, vì khác seed; nên train cả hai trên cùng seed). Báo cáo hiệu theo cặp seed.
- RMSE, tỷ lệ lạc quan và tỷ lệ bi quan của predictor.

## Báo cáo
Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-17: XÁC NHẬN
ELPIS-G − OSC trên predictor RAMP-V13 cho kết quả: C-MAPSS −0.81, pin −1.14, vòng bi −1.03 điểm (30/30, 30/30, 29/30 seed), p_Holm = 5.6e-9. Chi tiết xem FINDINGS.md.
