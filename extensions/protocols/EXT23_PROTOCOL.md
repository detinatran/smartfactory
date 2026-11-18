# EXT23: đối chứng cân lại trọng số lớp (review lần 6). Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Câu hỏi:** các bản nhiễu RPT có thể làm thay đổi tỷ lệ nhãn dương (cửa sổ lạc quan). Liệu lợi ích của `rpt_L`, nhất là trên
dữ liệu sạch, chỉ đến từ việc cân lại lớp?

**Biến thể** (đặc trưng `f`, dữ liệu S, HGB như OSC; `scripts/ext23_rw.py`):
- `full_rw`: trọng số của cửa sổ dương được nhân với hệ số c, sao cho tỷ lệ dương có trọng số bằng tỷ lệ dương có trọng số
  của S ∪ 2 bản nhiễu L (dữ liệu của `rpt_L`), tính theo từng run;
- `full_bal`: cân bằng lớp hoàn toàn (trọng số dương = tổng trọng số âm).

**Dữ liệu:** seed 72–81, 240 run. Thăm dò.

**Kiểm định:** `rpt_L` so với `full_rw` và `full_bal`, Holm trên 8. Báo cáo kèm tỷ lệ dương của S và của dữ liệu RPT.
