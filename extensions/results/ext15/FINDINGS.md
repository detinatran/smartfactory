# EXT15: trả lời review mô phỏng (seed 72–81, 240 run, 0 lỗi; thăm dò)

Protocol: `selective-rul/EXT15_PROTOCOL.md`. Chi tiết: `ext15_output.txt`, `reanalysis_output.txt`.

## Phân tích lại (không chạy thêm)
- **Mức seed** (n = 10, trung bình GRU/TCN): cả 40 phép so sánh có CI 95% không chứa 0, thắng 8–10/10 seed.
- **Risk dư do lỗi** (risk khi có lỗi − risk sạch):
  - giảm rõ ở C-MAPSS (0.69 → 0.19, 20/20) và với lỗi bias ở pin (−2.40);
  - ở N-CMAPSS, vòng bi và pin, lợi ích phần lớn cũng có trên dữ liệu sạch;
  - với vòng bi, lỗi không làm tăng risk.
- **Chi phí** r·lạc quan + bi quan: `rpt_L` tốt nhất so với mọi baseline bên ngoài khi r ≥ 2 ở cả 4 dataset. Khi r = 1, bộ
  phân loại đối xứng hoặc SelectiveNet tốt hơn ở 3/4 dataset.

## Thí nghiệm mới
- **H15a đúng: nhãn theo sai số của predictor là cốt lõi.** Cùng dữ liệu nhiễu:
  - gán nhãn kiểu outlier exposure (mọi cửa sổ nhiễu đều "từ chối") kém hơn `rpt_L` 1.63 / 5.18 / 2.33 / 2.00 điểm;
  - bộ phát hiện nhiễu kém hơn 2.5–7.3 điểm;
  - bộ phân loại đối xứng train trên dữ liệu nhiễu không được lợi gì.
- **Dữ liệu nhiễu có nhãn sai số cũng giúp các bộ ước lượng có hướng:**
  - `err_reg_rptL` và `gba_rptL` ngang `rpt_L` ở N-CMAPSS và vòng bi;
  - tốt hơn `rpt_L` ở pin (+0.36 đến +0.40, có ý nghĩa);
  - kém hơn ở C-MAPSS (−0.38 đến −0.77).

  Đóng góp vì vậy là sơ đồ huấn luyện, không phải một bộ ước lượng cụ thể.
- **Lỗi trên mọi kênh** (nằm ngoài support của bộ sinh): `rpt_L` tốt hơn có ý nghĩa so với ensemble, OSC và hai bộ có hướng
  ở cả 4 dataset (Holm trên 16).
- **Độ trùng giữa bộ sinh và lỗi:**
  - AUC > 0.8 ở 39/56 ô; vòng bi trùng nhiều (AUC 0.54–0.78);
  - tương quan giữa AUC và lợi ích của `rpt_L` yếu (ρ = 0.23, p = 0.10), nên lợi ích không chỉ đến từ các lỗi giống nhiễu.
- **Tiền đề của Mệnh đề 1:**
  - lỗi làm lệch trung bình ensemble gấp 1.4–25 lần so với thay đổi độ phân tán;
  - ở hai dataset turbofan, 11–40% cửa sổ mới trở nên lạc quan vẫn có độ phân tán dưới trung vị.
