# EXT65: kiểm lại độc lập ELPIS-JG trên vòng bi với 60 seed mới 282–341 (THĂM DÒ, không tiêu α; đặt trước đầy đủ). Chốt ngày 2026-11-18, TRƯỚC khi train và chạy

**Lý do:**
- Ở EXT64 (seed 252–281), trên vòng bi ELPIS-JG có hiệu −0.54 so với ELPIS-G trên predictor mặc định, CI 90% [−0.97, −0.13]. Đạt tiêu chí "hòa", nhưng Wilcoxon p = 0.13 nên chưa đạt tiêu chí "thắng".
- Độ lệch chuẩn theo seed khoảng 1.4. Với hiệu −0.54:
  - 30 seed chỉ có power khoảng 50%;
  - 60 seed có power khoảng 80%.
- Để tránh dừng tùy ý (chạy thêm tới khi đạt), kiểm định chính chỉ dùng **60 seed mới**, độc lập hoàn toàn với EXT64. Phân tích gộp 90 seed chỉ là phần phụ và được ghi rõ là có nhìn dữ liệu trước.

## Hệ thống (cố định, giống EXT64)
- ELPIS-JG chọn, cho mỗi deployment, giữa:
  - predictor mặc định;
  - predictor gộp (mặc định + ELPIS-J). Dùng `ext50.py` với `RAMP_FREEZE=1`, `RAMP_BIASCAL=1`, `RAMP_HYBRID=1`.
- Tiêu chí chọn: risk calibration của selector mà ELPIS-G chọn cho từng predictor. Khi bằng nhau thì dùng predictor mặc định.
- Phân tích bằng `scripts/ext64_summary.py`, không sửa.

## Dữ liệu
- Vòng bi (FEMTO + XJTU-SY, 6 fold leave-one-condition-out), GRU và TCN, seed 282–341 (60 seed, chưa từng dùng; đã kiểm tra trên van).
- **Train:** ensemble mặc định với `configs/external.json` (như `conf222_ext`), tối đa 8 tiến trình, 2 lượt để chạy bù.
- **Eval:** `ext38.py` cho hệ thống mặc định, rồi `ext50.py` cho hệ thống gộp. Mỗi bước 6 shard, chạy tuần tự.

## Kiểm định (đặt trước)
- **Chính:** hiệu risk dưới 6 lỗi ở coverage 90% matched, ELPIS-JG − ELPIS-G trên predictor mặc định; đơn vị là seed (trung bình GRU và TCN, có trọng số theo số unit qua fold); 60 seed.
  - "**Thắng**": trung bình < 0 và Wilcoxon signed-rank hai phía p < 0.05.
  - "**Hòa**": cận trên CI 90% bootstrap ≤ +0.3.
- **Phụ:**
  - gộp 90 seed (252–341, có nhìn dữ liệu trước);
  - hiệu lỗi bi quan;
  - tỷ lệ chọn predictor gộp;
  - RMSE.
- Seed nào thiếu run sau 2 lượt chạy bù thì loại và ghi vào đây.

## Báo cáo
Mọi kết quả đều được báo cáo, kể cả khi không thắng. Kết quả là thăm dò, không thay đổi phát biểu xác nhận nào của bài.

## Kết quả, ngày 2026-11-18 (thăm dò): "HÒA" đạt, "THẮNG" KHÔNG đạt ở kiểm định chính
- Train đủ 720/720 run, eval đủ 720/720 run mặc định và 720/720 run gộp. Không có run lỗi; đủ 60/60 seed.
- **Chính (60 seed mới 282–341), ELPIS-JG − ELPIS-G trên predictor mặc định:**
  - hiệu −0.56 điểm, CI 90% bootstrap [−0.97, −0.14];
  - 28/60 seed tốt hơn, 32/60 kém hơn; Wilcoxon p = 0.11;
  - lỗi bi quan −0.31; tỷ lệ chọn predictor gộp 0.43;
  - RMSE 0.267 so với 0.273.
  - Theo tiêu chí đặt trước: **đạt "hòa"** (cận trên CI ≤ +0.3, thậm chí < 0), nhưng **không đạt "thắng"** vì Wilcoxon p ≥ 0.05.
- **Phụ (gộp 90 seed 252–341, có nhìn dữ liệu trước):** hiệu −0.55, CI 90% [−0.87, −0.25], 46/90 seed tốt hơn, p = 0.036. Kết quả này không thay cho kiểm định chính.
- **Diễn giải:** hai mẫu độc lập cho cùng một hiệu trung bình (−0.54 và −0.56), và CI của trung bình đều nằm dưới 0. Nhưng chỉ khoảng một nửa số seed tốt hơn. Như vậy, lợi ích trên vòng bi là hiệu ứng trung bình, đến từ một số deployment có lợi lớn, chứ không phải cải thiện điển hình ở mọi seed. Vì vậy kiểm định theo hạng (Wilcoxon) không đạt.
