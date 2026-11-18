# EXT13: thêm baseline cho bảng so sánh. Chốt ngày 2026-11-13, TRƯỚC khi chạy

Người dùng yêu cầu bảng so sánh có càng nhiều phương pháp càng tốt. EXT11 đã có 11 baseline; EXT13 thêm 6 baseline thuộc
các họ còn thiếu (Bayes xấp xỉ, phát hiện OOD trên không gian đặc trưng, phát hiện bất thường, phát hiện lỗi bằng dư thừa
giải tích, test-time augmentation).

- Code: `scripts/ext13_more.py`.
- Chạy trên **cùng run và cùng dự đoán** của EXT10 và EXT11 (seed 62–71). Sau khi có kết quả xác nhận vòng 2, chạy thêm
  trên seed 72–81.
- Chỉ tạo score. Mọi selector xếp hạng cùng dự đoán ensemble.
- Ngưỡng frozen lấy trên unit calibration sạch.

| Tên | Định nghĩa (chốt trước, không tune) |
|---|---|
| `laplace` | Laplace lớp cuối (Daxberger 2021, *Laplace Redux*). GGN trên cửa sổ fit sạch, prior chọn theo evidence trên lưới 1e-4…1e4 (code EXT3). Score = độ lệch chuẩn dự đoán, trung bình trên 3 thành viên |
| `mahalanobis` | Lee 2018: Gaussian (mean + covariance chung, thêm 1e-6·I) trên đặc trưng áp chót của cửa sổ fit sạch. Score = khoảng cách Mahalanobis, trung bình trên 3 thành viên |
| `knn_feat` | Sun 2022: đặc trưng áp chót chuẩn hoá L2. Score = khoảng cách tới láng giềng thứ k = 50 trong tập fit sạch (lấy mẫu tối đa 20 000 cửa sổ), trung bình trên 3 thành viên |
| `iforest` | Isolation Forest (Liu 2008), 200 cây, trên đặc trưng selector `f` của cửa sổ selector sạch. Score = −score_samples |
| `ar_residual` | Residual dư thừa giải tích (Chow và Willsky 1984): max_c s_c / τ_c của bộ tái tạo Ridge EXT6 |
| `tta_std` | Test-time augmentation (Wang 2019): 10 lần nhiễu N(0, 0.1²) trên các kênh giá trị đã chuẩn hoá (cờ thiếu giữ nguyên). Score = độ lệch chuẩn của dự đoán ensemble qua 10 lần |

**Kiểm định:** OURS (`full_recon_rpt`) so với từng baseline mới trên 4 dataset, tức 24 phép thử. Holm trên 24 ở mức 0.05.
Đây là họ riêng, bổ sung cho họ A của EXT11.

**Ghi chú:** `ar_residual` và `iforest` là bộ phát hiện bất thường hoặc lỗi, không ước lượng sai số. Chúng được đưa vào để
trả lời câu hỏi "chỉ phát hiện lỗi cảm biến có đủ không".
