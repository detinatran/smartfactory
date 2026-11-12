# EXT10: huấn luyện bộ chọn lọc với nhiễu loạn ngẫu nhiên (randomized-perturbation training, RPT). Chốt ngày 2026-11-12, TRƯỚC khi chạy

Áp dụng các quy tắc chung trong `EXT9_PROTOCOL.md`. EXT9 không tiêu α nào, nên EXT10 là **vòng xác nhận 1**: seed 62–71,
20 cặp mỗi dataset, α_1 = 0.025.

## Ý tưởng
- Bộ chọn lọc của bài ESANN (`full`) học "khi nào dự đoán lạc quan quá δ" chỉ từ 3 shift đã thấy: clean, dropout và drift.
- Các lỗi đánh giá thuộc một họ khác: stuck, bias, noise, intermittent, spike và gain.
- EXT10 áp dụng **domain randomization** cho bộ chọn lọc:
  - thêm một bản sao cửa sổ selector sạch, bị nhiễu loạn bởi một **họ hàm ngẫu nhiên tổng quát**;
  - bộ chọn lọc học đặc trưng nào báo hiệu "đầu vào bị lệch khiến mô hình đoán sai" trên một không gian lệch rộng, thay vì
    chỉ trên 2 loại lỗi cụ thể.

## Bộ sinh nhiễu loạn (cố định, chốt tại đây)
Áp dụng cho mỗi cửa sổ selector sạch, trong không gian đã chuẩn hoá, chỉ trên các kênh giá trị; cờ thiếu không đổi.
- Chọn k ~ Uniform{1, …, max(1, round(0.2·n_ch))} kênh.
- Với mỗi kênh, dùng hàm trơn ngẫu nhiên φ(t) = Σ_{m=0..2} a_m·cos(π·m·t/T + b_m), với a_m ~ N(0, 1) và b_m ~ U(0, 2π).
  - Chuẩn hoá sao cho max|φ| = 1.
  - Nhân với biên độ A ~ LogUniform(0.5, 4) độ lệch chuẩn.
  - Với xác suất 0.5, đặt φ = 0 trước một thời điểm bắt đầu t0 ~ U{0, …, T−1}.
- Hạt giống: seed + 50000.

**Ghi chú trung thực:** họ hàm này không dùng bộ sinh của 6 lỗi đánh giá. Nhưng vì là họ tổng quát, nó chồng lấn một phần
với một số lỗi:
- thành phần m = 0 gần giống bias;
- thành phần có thời điểm bắt đầu gần giống bước nhảy.

Bài báo phải nói rõ điều này. Lỗi đánh giá vẫn "chưa thấy" theo nghĩa bộ sinh của chúng không được dùng khi huấn luyện.

## Ứng viên
Code: `scripts/ext10_rpt.py`. Đó là `ext8_select.py` giữ nguyên mọi điểm cũ, thêm:

| Tên | Định nghĩa |
|---|---|
| `full_rpt` | HGB giống hệt `full`, nhưng huấn luyện trên selector × {clean, dropout, drift, **rand**} |
| `full_recon_rpt` | Tương tự với `full_recon`: thêm 3 đặc trưng tái tạo |

Trọng số theo unit dùng `engine_weights` trên toàn bộ dữ liệu ghép. Siêu tham số HGB giữ nguyên.

## Giai đoạn phát triển
- Chạy trên seed 42–61, tức 480 run, 40 cặp mỗi dataset.
- Chọn phương pháp chính trong {`full_rpt`, `full_recon_rpt`} bằng quy tắc an toàn:
  - hiệu so với `full` ≤ +0.10 điểm ở mọi dataset;
  - lỗi bi quan tăng ≤ 1 điểm;
  - trong số đó, chọn ứng viên có tổng hiệu âm nhất.
- Nếu không ứng viên nào an toàn và có tổng âm, không xác nhận gì và chuyển sang vòng sau. Vòng sau vẫn dùng seed 62–71
  với α_1, vì chưa tiêu α nào.

## Xác nhận
- Seed 62–71, chưa từng dùng. Train song song với giai đoạn phát triển, nhưng **không đánh giá** cho đến khi chốt phương
  pháp.
- Tiêu chí như `EXT9_PROTOCOL.md` mục 4, ở mức α_1 = 0.025.

## Chốt phương pháp chính, ngày 2026-11-12, TRƯỚC khi đánh giá seed 62–71
Kết quả giai đoạn phát triển: `ext10_v1/summary_output.txt` (480 run, 0 lỗi). Hiệu so với `full` trên 40 cặp:

| Ứng viên | C-MAPSS | N-CMAPSS | Vòng bi | Pin | Tổng |
|---|---|---|---|---|---|
| `full_recon_rpt` | −0.58 | −0.30 | −0.29 | −0.32 | −1.50 |
| `full_rpt` | −0.35 | −0.34 | −0.30 | −0.45 | −1.44 |

- Cả hai đều an toàn. Lỗi bi quan tăng tối đa 0.43 điểm.
- **Phương pháp chính: `full_recon_rpt`.** Kiểm định: `scripts/conf_test.py all.csv.gz full_recon_rpt 0.025 62 71`.
- Ghi nhận: trên 10 cặp seed 57–61, `full_recon_rpt` ở pin là +0.22, trong khi `full_rpt` là −0.04. Vẫn giữ quy tắc đã chốt
  và không đổi lựa chọn.
- `full_rpt` chỉ được báo cáo mô tả ở bước xác nhận, không có kiểm định.
