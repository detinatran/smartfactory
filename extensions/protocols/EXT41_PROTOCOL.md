# EXT41: selector trên glitch cảm biến tự nhiên trong dữ liệu pin (THĂM DÒ, không tiêu α). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Câu hỏi (review lần 12 và 13):** mọi lỗi trong bài đều là lỗi giả lập. Trên các glitch có sẵn trong dữ liệu đo thật, selector có loại được các lỗi lạc quan không?

## Định nghĩa glitch (đặt trước, chỉ dựa trên dữ liệu thô, không xem kết quả selector)
- **Dữ liệu:** Severson, file `battery.csv`, bỏ chu kỳ 1. Mỗi cell, mỗi kênh, so với median trượt 11 chu kỳ (lấy tâm ở chu kỳ đang xét).
- **Chu kỳ glitch ở mức chính:** ít nhất một kênh lệch khỏi median vượt ngưỡng sau.

| Kênh | Ngưỡng |
|---|---|
| QDischarge, QCharge | lệch tương đối > 2% (dung lượng không thể đổi 2% trong một chu kỳ) |
| IR | lệch tương đối > 10% |
| Tavg, Tmin, Tmax | lệch tuyệt đối > 2 °C |

- **Phụ:** thêm `chargetime` với ngưỡng lệch tương đối > 5%. Kênh này được tách riêng vì thời gian sạc có thể đổi theo policy.
- **Khảo sát sơ bộ (chỉ trên dữ liệu thô):** khoảng 0.4% chu kỳ bị glitch ở mức chính, nằm rải ở 17–47 cell.
- **Window glitch:** window 30 chu kỳ có ít nhất một chu kỳ glitch.

## Dữ liệu và chạy
- Dùng các run `battery_cv` của vòng 8 (seed 162–191, 300 run, xoay unit test). Không train lại predictor.
- `scripts/ext41.py` (dựa trên `ext38.py`): ghi score theo từng window trên window test sạch và cờ chấp nhận ở coverage 90% (ngưỡng matched trên toàn bộ tập test) cho 7 score: `ensemble_std`, `gauss_bias_aware`, `err_reg`, `err_reg_rptL`, `full`, `rpt_L`, `rpt_auto_d`.

## Phân tích (đặt trước)
1. **Predictor:** tỷ lệ lỗi lạc quan trên window glitch so với window không glitch. Cho biết glitch thật có làm predictor lạc quan không.
2. **Chính:** risk trên window glitch được chấp nhận, tức tỷ lệ lạc quan trong các window glitch mà selector chấp nhận. So `rpt_auto_d` với `full`: trung bình theo seed, Wilcoxon trên 30 seed.
3. **Phụ:**
   - tỷ lệ window glitch bị từ chối;
   - risk trên window không glitch;
   - so sánh với `ensemble_std`, `gauss_bias_aware`, `err_reg`, `err_reg_rptL`;
   - định nghĩa glitch có thêm `chargetime`.

## Diễn giải đặt trước
- Nếu predictor không lạc quan hơn trên window glitch (hiệu ở mục 1 ≤ 0.5 điểm), thì glitch tự nhiên ở đây vô hại. Báo cáo như một giới hạn: không có lỗi thật nào gây hại để đánh giá.
- Mọi kết quả đều là thăm dò.

### Bổ sung trước khi có kết quả, ngày 2026-11-16
Thêm phân tích mô tả (1b): so sánh có ghép cặp theo tuổi pin. Lấy các window không glitch của cùng cell, trong khoảng ±50 chu kỳ quanh window glitch. Lý do: glitch tập trung ở cuối đời pin (trung vị khoảng 0.72 tuổi thọ), nên so sánh (1) dễ bị nhiễu bởi vị trí trong vòng đời. Bổ sung này viết sau khi chỉ mới xem kết quả chạy thử trên 1 run.

## Kết quả, ngày 2026-11-16 (thăm dò)
- Glitch thật làm tăng tỷ lệ lạc quan của predictor: 6.66% so với 3.02% khi ghép cặp theo tuổi pin.
- RPT-D giảm risk trên window glitch được chấp nhận từ 2.14% xuống 0.61% (30/30 seed).
- RPT-D loại 94% window glitch lạc quan, so với 79% của OSC.

Chi tiết xem FINDINGS.md.

### Bổ sung sau khi có kết quả (sensitivity, post hoc), ngày 2026-11-16
Theo review lần 3: sau giai đoạn nghỉ, dung lượng pin Severson có thể bật lên. Đây là hiện tượng vật lý thật, không phải lỗi cảm biến. Vì vậy thêm định nghĩa `g_irt`: chỉ dùng IR và nhiệt độ, bỏ hai kênh Q.
