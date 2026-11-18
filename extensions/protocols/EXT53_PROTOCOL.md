# EXT53: thêm các biến thể sửa ELPIS-J trên vòng bi (PHÁT TRIỂN, thăm dò). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Bối cảnh:**
- EXT51 cho thấy trên vòng bi, cả kiến trúc của ELPIS-J lẫn bước pretrain JEPA đều gây hại.
- EXT52 V1 (pretrain trên nhiều dữ liệu hơn) không cải thiện: +3.62 điểm so với predictor mặc định.

## Biến thể
Chỉ chạy trên vòng bi, seed 162–171, mỗi biến thể 120 run. Các biến thể chạy lần lượt, mỗi biến thể dùng 4 tiến trình. Code là `scripts/ext50.py`, cấu hình bằng biến môi trường.

| Biến thể | Cấu hình | Ý tưởng |
|---|---|---|
| V4 | `ELPISJ_NOSYM=1`, `ELPISJ_FREEZE=1` | kết hợp V2 và V3 |
| V5 | `ELPISJ_K=0` | target JEPA là chính window đó khi sạch (JEPA khử nhiễu), không dự đoán tương lai |
| V6 | `ELPISJ_CORR2=1` | ở Stage II fine-tune trên cả window sạch lẫn window bị hỏng (nhãn thật), để predictor bớt lạc quan khi có lỗi |
| V7 | `ELPISJ_NOSYM=1`, `ELPISJ_CORR2=1` | |
| V8 | `EXT50_E1=0`, `ELPISJ_NOSYM=1`, `ELPISJ_CORR2=1` | không JEPA, không nhánh symbolic, fine-tune bền với lỗi |

## Tiêu chí và cách dùng kết quả
- **Tiêu chí:** risk của RPT-D dưới 6 lỗi trên vòng bi, so với predictor mặc định trên cùng seed. Đồng thời phải không làm hỏng C-MAPSS và pin: biến thể nào được chọn sẽ được kiểm tra lại trên cả ba họ.
- Đây là bước phát triển trên seed đã dùng. Hệ thống cuối cùng phải được xác nhận trên seed mới, ở vòng α cuối cùng.

## Bổ sung V9–V10 (EXT54), chốt ngày 2026-11-17, TRƯỚC khi chạy
Lý do: kết quả EXT52 V2 cho thấy JEPA là nguyên nhân chính gây hại trên vòng bi.

| Biến thể | Cấu hình | Ý tưởng |
|---|---|---|
| V9 | `ELPISJ_CORRUPT=mask` | Stage I chỉ dùng mặt nạ kênh trong khối thời gian liền, bỏ nhiễu RPT |
| V10 | `EXT50_E1=5` | pretrain ngắn |

Hai biến thể này chạy sau EXT53, theo cùng quy trình.

## Bổ sung V11–V12 (EXT55), chốt ngày 2026-11-17, TRƯỚC khi chạy

Lý do: EXT52 V3 (đóng băng encoder) cho kết quả tốt nhất cho tới nay. RMSE 0.263; RPT-D +1.35 so với mặc định.

| Biến thể | Cấu hình |
|---|---|
| V11 | `ELPISJ_FREEZE=1`, `ELPISJ_CORR2=1` |
| V12 | `ELPISJ_FREEZE=1`, `ELPISJ_CORR2=1`, `ELPISJ_NOSYM=1` |

Hai biến thể này chạy sau EXT54.

## Huỷ các biến thể còn lại, ngày 2026-11-17 (quyết định phát triển, ghi trước khi có kết quả của các biến thể này)
V5 (+3.70) và V6 (+3.61) không giúp. Phân tích thêm cho thấy predictor ELPIS-J trên vòng bi lạc quan hơn cả trên dữ liệu sạch (29.8% so với 26.5%), nghĩa là predictor bị lệch, không phải phản ứng sai với lỗi. Vì vậy V7–V12 bị huỷ, thay bằng hiệu chỉnh độ lệch (EXT56). V7 bị dừng giữa chừng và không được phân tích.
