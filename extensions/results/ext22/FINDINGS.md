# EXT22: `rpt_auto`, xác nhận ở vòng 4 trên 5 họ dữ liệu (gồm ổ cứng thật)

- `rpt_auto` chọn theo từng run giữa OSC và `rpt_L`, dựa trên risk calibration dưới các shift đã thấy (không dùng dữ liệu nhiễu
  hay dữ liệu test).
- **Phát triển** (seed 42–61 cho 4 họ, HDD 42–51): hiệu so với `full` là −0.32 / −0.41 / −0.64 / −0.62 / +0.07; đạt quy tắc.
- **Vòng 4** (seed 92–101, Holm trên 15, α = 0.003125): **XÁC NHẬN.**
  - So với `full`: C-MAPSS −0.49, N-CMAPSS −0.27 (n.s. ở α_4), vòng bi −1.05, pin −0.67, HDD +0.03 (n.s.).
  - Không họ nào kém có ý nghĩa.
- **Trên HDD:** `rpt_auto` chọn OSC ở 16/20 run, nên tránh được thiệt hại của `rpt_L` (+0.31 so với OSC trên seed này).
- Chi tiết: `ext22_dev_output.txt`, `ext22_confirm_output.txt`, `confirm22.csv`.
