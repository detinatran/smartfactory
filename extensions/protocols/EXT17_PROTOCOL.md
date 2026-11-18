# EXT17: lỗi cảm biến vật lý trên tín hiệu rung thô của vòng bi. Chốt ngày 2026-11-13, TRƯỚC khi chạy

**Lý do:** review lần 2 chỉ ra rằng lỗi được tiêm vào đặc trưng dẫn xuất (34 đặc trưng vòng bi), không phải tín hiệu cảm
biến thật. Không có dataset công khai nào vừa có lỗi cảm biến thật vừa có quỹ đạo chạy đến hỏng. EXT17 vì vậy tiêm lỗi vào
**tín hiệu gia tốc thô** (FEMTO 25.6 kHz, XJTU-SY), rồi tính lại đặc trưng bằng đúng `prepare_external.signal_features`.

## Lỗi
- Mỗi lỗi tác động lên 1 trục (h hoặc v, chọn ngẫu nhiên cố định theo unit), trong khoảng vị trí [onset, onset + 0.3) của
  vòng đời, với onset ~ U(0.2, 0.4).
- Hạt giống theo tên unit, nên không phụ thuộc seed run.
- r_ref = trung vị rms sạch của trục đó trong 10% đầu vòng đời.

| Tên | Định nghĩa trên tín hiệu thô x | Ý nghĩa vật lý |
|---|---|---|
| `raw_gain` | 0.5·x | độ nhạy giảm (cáp, giá gắn) |
| `raw_clip` | kẹp x trong [−2·r_ref, 2·r_ref] | bão hoà ADC hoặc khuếch đại |
| `raw_detach` | trung bình trượt 8 mẫu, nhân 0.6 | cảm biến lỏng (lọc thấp, suy giảm) |
| `raw_noise` | x + N(0, r_ref²) | nhiễu điện |
| `raw_spike` | 0.2% mẫu cộng ±10·r_ref | xung EMI |

## Đánh giá (`scripts/ext17_eval.py` = `ext12_main_v2.py` với kịch bản `raw_*`)
- Dữ liệu: 120 run vòng bi, seed 72–81, GRU/TCN; mô hình và selector giữ nguyên.
- So sánh `rpt_L` với `ensemble_std`, `full`, `gauss_bias_aware` và `err_reg`, trên risk @90% matched, trung bình 5 lỗi thô.
  Holm trên 4 phép thử. Báo cáo thêm risk dư (lỗi − sạch) và kết quả theo từng lỗi.
- Thăm dò; phương pháp không đổi.
