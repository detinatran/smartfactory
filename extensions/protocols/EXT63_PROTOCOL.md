# EXT63: pretrain JEPA trên mọi điều kiện vòng bi, kết hợp predictor gộp (THĂM DÒ / PHÁT TRIỂN, không tiêu α). Chốt ngày 2026-11-17, TRƯỚC khi chạy

**Lý do:**
- Ở EXT62, trên vòng bi, predictor gộp (mặc định + RAMP-V13) hòa về điểm với predictor mặc định (+0.35), nhưng không đạt biên không kém.
- Giả thuyết: pretrain JEPA thiếu dữ liệu. Fold vòng bi là leave-one-condition-out trên FEMTO và XJTU-SY gộp, nên fit đã chứa 5 điều kiện khác nhưng chỉ có khoảng 12–15 unit.

**Ghi chú về lần thử trước:** V1 (EXT52) đã pretrain trên fit, selector và calibration, nhưng encoder không đóng băng, không hiệu chỉnh độ lệch, không gộp. Kết quả +3.62.

## Biến thể
Dùng `scripts/ext50.py`. Cả hai có chung `RAMP_FREEZE=1`, `RAMP_BIASCAL=1`, `RAMP_HYBRID=1`.

**V18:** `RAMP_PRE_PARTS=fit,validation,selector,calibration`.
- Stage I tự giám sát trên mọi unit không thuộc test, tức đủ 5 điều kiện train.
- Không dùng nhãn trong Stage I, không dùng dữ liệu test.

**V19:** `RAMP_PRE_PARTS=fit,validation,selector,calibration,evaluation`.
- Thêm các window sạch, không nhãn, của điều kiện test.
- Đây là tự giám sát lúc test (transductive). Mô hình thấy input sạch, kể cả input tương lai, của chính các unit test, chỉ không thấy nhãn. Window có lỗi không bao giờ được dùng.
- Kết quả của V19 là **cận trên lạc quan**, không phải một quy trình triển khai thực tế, và sẽ được ghi rõ như vậy.

## Dữ liệu và tiêu chí
**Phát triển:** vòng bi, seed 162–171 (run `conf162_ext`). So với mặc định (EXT40) và V17 (EXT61).

Đạt ở bước phát triển nếu:
- hiệu risk RPT-D so với mặc định có trung bình ≤ 0, với ≥ 5/10 seed tốt hơn;
- lỗi bi quan không cao hơn mặc định.

**Lặp lại trên seed mới, chỉ khi V18 đạt:**
- Seed 252–281, chỉ vòng bi. Train ensemble mặc định mới, giống `conf222_ext`. Seed 222–251 đã được dùng ở EXT62 nên không dùng lại.
- Tiêu chí đặt trước (giống EXT62): hiệu RPT-D so với mặc định, cận trên CI 90% bootstrap theo seed ≤ +0.3, gọi là "hòa"; trung bình < 0 và Wilcoxon p < 0.05, gọi là "thắng".
- V19 chỉ được lặp lại kèm theo, dưới nhãn cận trên.

## Vận hành
- V18 và V19 chạy song song, mỗi biến thể 4 tiến trình.
- Sau đó train ensemble mặc định cho vòng bi seed 252–281, để có sẵn cho bước lặp lại. Việc này không phụ thuộc kết quả.

## Báo cáo
Mọi kết quả đều được báo cáo. Không đổi phát biểu xác nhận nào của bài.

## Kết quả, ngày 2026-11-18 (thăm dò): KHÔNG đạt; pretrain trên nhiều dữ liệu vòng bi hơn không giúp
Vòng bi, seed 162–171, 120/120 run mỗi biến thể. Hiệu risk RPT-D so với mặc định:

| Biến thể | Risk RPT-D | Lỗi bi quan | RMSE |
|---|---|---|---|
| V17 (tham chiếu, EXT61) | +0.14 | −3.15 | 0.261 |
| V18 (mọi unit không thuộc test) | +1.21 (3/10 seed) | −4.38 | 0.266 |
| V19 (thêm test không nhãn, cận trên transductive) | +0.36 (4/10 seed) | −2.35 | 0.277 |

- Thêm dữ liệu cho pretrain JEPA làm predictor gộp lạc quan hơn (V18), hoặc không cải thiện gì (V19).
- Kết luận: nguồn gây lạc quan trên vòng bi là chính bước pretrain JEPA, không phải do thiếu dữ liệu.
- V18 không đạt tiêu chí, nên không lặp lại trên seed mới. Các run vòng bi seed 252–281 được dùng cho EXT64.
