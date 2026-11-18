# EXT21: họ dữ liệu ngoài chưa từng dùng, ổ cứng hỏng thật (Backblaze SMART). Chốt ngày 2026-11-14, TRƯỚC khi xử lý dữ liệu và chạy

**Lý do:** cả 5 review đều yêu cầu đánh giá trên dữ liệu chưa từng dùng khi phát triển phương pháp, tốt nhất là dữ liệu
thực tế. Người dùng cho phép tải dữ liệu công khai. Bộ NASA Li-ion được khảo sát nhưng bị loại **trước khi xem bất kỳ kết quả
nào**, vì chỉ khoảng 12–14 trên 34 cell thật sự suy giảm tới end-of-life và có cell dữ liệu bất thường.

## Dữ liệu (`scripts/prepare_hdd.py`)
- **Nguồn:** Backblaze Hard Drive Data, Q1/2023 đến Q2/2024 (6 archive, khoảng 5.7 GB; SHA256 ghi trong `dl_bb.log`).
- **Unit:** ổ có `failure = 1` trong khoảng thời gian trên và có đủ ≥ 90 ngày dữ liệu ngay trước ngày hỏng.
  - Quỹ đạo = 90 ngày cuối; nhãn `rul_steps` = số ngày tới khi hỏng (89 … 0).
  - Mục tiêu theo pipeline external là phần quỹ đạo còn lại; δ = 0.16, tương đương khoảng 14 ngày.
- **Model ổ:** 4 model có nhiều ổ hỏng đủ điều kiện nhất (`regime` = model).
- **Kênh:** thuộc tính SMART có mặt ở cả 4 model, mỗi kênh thiếu ≤ 50% hàng.
  - Dạng raw cho các bộ đếm: 5, 9, 12, 187, 188, 192, 193, 194, 197, 198, 199, 240, 241, 242.
  - Dạng normalized cho 1, 3, 7.
  - Bộ đếm được biến đổi log1p, sau đó forward-fill theo từng ổ.
- **Chia:** 30% ổ (theo hash của serial) làm tập `test` cho đánh giá; phần còn lại là `dev`, chia fit / validation / selector /
  calibration như N-CMAPSS.
- **Cửa sổ:** 30 ngày, bước nhảy 2.

## Huấn luyện và đánh giá
- GRU và TCN, cấu hình `configs/external.json` (mặc định), 10 seed (42–51), tức 20 cặp.
- Chạy `ext12_main_v2.py`, có `--stress` để đo cả lỗi held-out.
- **Phương pháp cố định:** `rpt_L`, không tune gì trên HDD.

## Kiểm định
- **Chính:** `rpt_L` so với `full`, Holm trên 4 cùng với `gauss_bias_aware`, `err_reg` và `ensemble_std`, ở mức α = 0.05.
  Đây là họ kiểm định riêng cho **kiểm định ngoài** (external validation), không thuộc ngân sách α của vòng tìm kiếm.
- **Phụ:** risk dư do lỗi, AURC, lỗi bi quan.
- Nếu `rpt_L` không tốt hơn `full` có ý nghĩa trên HDD, kết quả vẫn được báo cáo đầy đủ trong bài.

## Sai lệch, ghi ngày 2026-11-14, TRƯỚC khi so sánh các phương pháp: sửa cách dựng dữ liệu
- **Vấn đề:** bản dựng đầu tiên lấy đúng 90 ngày cuối của mỗi ổ, nên vị trí cửa sổ (đặc trưng thời gian đã chạy của selector,
  log của cycle) quyết định hoàn toàn RUL. Predictor dựa trên SMART gần như hằng số (độ lệch chuẩn dự đoán 0.05), nên mọi
  selector học được đều xếp hạng giống nhau, chỉ dựa vào thời gian đã chạy. Ở 4 run đầu, risk của các selector giống hệt
  nhau. Bài toán khi đó tầm thường và không đánh giá được gì.
- **Sửa:** mỗi ổ lấy toàn bộ lịch sử quan sát trong khoảng dữ liệu, tối đa 365 ngày cuối, nên độ dài thay đổi theo ổ. Nhãn
  `rul_steps` = min(số ngày tới khi hỏng, 90), tức RUL bị cap ở 90 ngày như C-MAPSS. Mục tiêu = `rul_steps` / 90.
  - Điều kiện: ổ có ≥ 120 ngày quan sát trước khi hỏng.
  - Các phần khác (4 model, kênh SMART, chia 30% test, cửa sổ 30 ngày) giữ nguyên; bước nhảy đổi thành 3 vì dữ liệu dài hơn.
- Kết quả bản dựng đầu được giữ lại trong log, nhưng không dùng. Kiểm định và phương pháp không đổi.
