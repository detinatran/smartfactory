# EXT56: RAMP có hiệu chỉnh độ lệch hậu kiểm trên tập validation (PHÁT TRIỂN, thăm dò). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Chẩn đoán:** trên vòng bi, predictor RAMP lạc quan hơn ngay cả trên dữ liệu sạch: 29.8% so với 26.5% của predictor mặc định. RMSE thì gần như bằng nhau. Nghĩa là predictor bị lệch, dự đoán RUL cao hơn có hệ thống.

**Cách sửa (`RAMP_BIASCAL=1`):**
- Sau Stage II, tính phần dư có dấu trung bình (dự đoán − y, có trọng số theo unit) trên window validation sạch, rồi trừ phần dư này khỏi mọi dự đoán.
- Chỉ dùng unit validation, vốn đã dùng cho early stopping. Không đụng tới dữ liệu test.
- Lưu ý: predictor mặc định không được hiệu chỉnh kiểu này. Bài báo sẽ ghi rõ đây là một bước trong quy trình huấn luyện của RAMP, không phải so sánh ngang hàng.

## Biến thể
| Biến thể | Cấu hình |
|---|---|
| V13 | `RAMP_FREEZE=1`, `RAMP_BIASCAL=1` |
| V14 | `RAMP_BIASCAL=1` |

Chạy trên vòng bi trước, seed 162–171, 4 tiến trình. Nếu cải thiện được vòng bi, chạy lại biến thể tốt nhất trên C-MAPSS và pin để kiểm tra nó không làm hai họ này tệ đi.

## Kết quả V13 và bổ sung V15, ngày 2026-11-17 (V15 chốt TRƯỚC khi chạy)
- V13: RPT-D +1.08 so với mặc định trên vòng bi (4/10 seed tốt hơn, p = 0.62). Tỷ lệ lạc quan 28.0%.
- V15 = `RAMP_FREEZE=1`, `RAMP_BIASCAL=late`. Offset được tính chỉ trên các window validation cuối đời (y ≤ 0.3·cap), vì lỗi lạc quan tập trung ở cuối đời.
