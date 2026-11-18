# EXT52: sửa ELPIS-J trên vòng bi bằng 3 biến thể (THĂM DÒ / PHÁT TRIỂN). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Lý do:** ở EXT50, ELPIS-J kém predictor mặc định 3.3 điểm trên vòng bi. Predictor ELPIS-J lạc quan hơn khi có lỗi (29.9% so với 26.3%).

## Biến thể
Tất cả dùng `scripts/ext50.py`, chỉ khác biến môi trường.

| | Biến môi trường | Ý tưởng |
|---|---|---|
| V1 | `ELPISJ_PRE_PARTS=fit,selector,calibration` | Stage I tự giám sát dùng window của cả ba phần fit, selector, calibration, không dùng nhãn. Unit test không bao giờ được đụng tới. |
| V2 | `ELPISJ_NOSYM=1` | Tắt nhánh symbolic (s = 0). |
| V3 | `ELPISJ_FREEZE=1` | Đóng băng encoder ở Stage II, chỉ train các module head. |

## Dữ liệu
Chỉ vòng bi, seed 162–171, 120 run cho mỗi biến thể. Chạy tuần tự, mỗi lần 4 tiến trình.

## Phân tích
Với từng biến thể, so với ELPIS-J gốc (EXT50) và với predictor mặc định (EXT40), trên cùng seed:
- RMSE trên dữ liệu sạch;
- tỷ lệ lạc quan dưới lỗi;
- risk của RPT-D dưới lỗi.

## Lưu ý
Chọn biến thể tốt nhất trên các seed này là phát triển, không phải xác nhận. Nếu một hệ thống ELPIS-J cuối cùng được đưa vào bài, nó phải được xác nhận lại trên seed mới. Ngân sách α hiện chỉ còn 9.8e-5, nên chỉ đủ cho đúng một vòng nữa.

## Kết quả V1, V2 (2026-11-17)
Risk RPT-D trên vòng bi so với mặc định: V1 +3.62, V2 +3.25. Cả hai không cải thiện. Nguyên nhân chính là JEPA (ELPIS-J bỏ JEPA chỉ +1.46).

## Kết quả V3 (2026-11-17)
RMSE 0.263, tốt hơn mặc định (0.280). RPT-D +1.35 so với mặc định, cải thiện từ +3.30.
