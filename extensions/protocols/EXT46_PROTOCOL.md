# EXT46: baseline "sửa kênh lỗi + conformal cục bộ" (dựng theo tinh thần RobustUQ của Wang et al. 2025, có rút gọn). PRE-REGISTERED SECONDARY, chốt ngày 2026-11-16, TRƯỚC khi chạy

**Lý do:** cả ba lần review đều yêu cầu so với một mô hình RUL chống lỗi cảm biến. Không có toàn văn công khai của RobustUQ. Theo abstract, phương pháp gồm học biểu diễn, điền giá trị thiếu bằng diffusion, và conformal prediction kết hợp metric learning trên phần dư. Bài báo sẽ ghi rõ đây là bản dựng lại có rút gọn, không phải cài đặt lại RobustUQ.

## Hệ thống `repair_lconf`
- **Sửa kênh:** `Reconstructor` có sẵn: ridge tái tạo chéo kênh, fit trên window selector sạch. Kênh bị coi là nghi ngờ khi độ lệch tái tạo vượt quantile 99.5% trên dữ liệu sạch, hoặc khi kênh bị thiếu. Kênh nghi ngờ được thay bằng giá trị tái tạo, lặp 3 lần.
- **Dự đoán:** predictor cố định chạy trên window đã sửa.
- **Score:** quantile 90% của phần dư có dấu (dự đoán đã sửa − y) trong k láng giềng gần nhất, tìm trên các window calibration đã sửa (clean, dropout, drift). k = min(50, max(10, N/20)); không gian là đặc trưng φ đã chuẩn hoá. Window có score cao nhất bị từ chối.
- Risk đo ở coverage 90% (matched), dùng dự đoán của chính hệ thống này.

## Dữ liệu
Các run vòng 8, seed 162–171, đủ 5 họ (C-MAPSS, `ncmapss_cv`, vòng bi, `battery_cv`, `hdd`). Không train lại predictor.

## Kiểm định (đặt trước, họ phụ, α = 0.05 trong họ, nằm ngoài ngân sách α)
- **So sánh:** `rpt_auto_d` − `repair_lconf`. Risk trung bình 6 lỗi, trung bình theo seed (10 seed). Wilcoxon theo seed, Holm trên 5 họ.
- **Mô tả:**
  - risk sạch;
  - tỷ lệ lạc quan của predictor trên window đã sửa so với chưa sửa;
  - so với `full`.

## Kết quả, ngày 2026-11-16
RPT-D tốt hơn repair_lconf ở 4/5 họ (−1.7 đến −8.5 điểm, 10/10 seed). Kém hơn trên vòng bi (+2.35). Chi tiết xem FINDINGS.md.
