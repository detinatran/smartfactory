# EXT29: vòng xác nhận 5. Risk dư trên window bị lỗi là thước đo chính; thêm lỗi che giấu suy giảm và mô phỏng bảo trì. Chốt ngày 2026-11-15, TRƯỚC khi train và chạy

**Lý do:** review 8, W2, W3 và W8.
- Kết quả trên window bị lỗi (EXT25) là post hoc.
- Trên vòng bi và HDD, các lỗi đã dùng không làm predictor lạc quan, nên hai họ này không kiểm định được độ bền với lỗi.
- Kết quả chưa có ý nghĩa vận hành.

## Phương pháp
`rpt_auto` (đã cố định từ EXT22), không thay đổi gì.

## Dữ liệu và huấn luyện
- **Seed mới 112–121.**
- GRU và TCN cho:
  - C-MAPSS FD001–4, `configs/ext_capped.json`;
  - N-CMAPSS, vòng bi, pin, `configs/external.json`;
  - HDD (4 model gốc), `configs/external_hdd.json`.
- Tổng 260 ensemble mới, cùng launcher với vòng 4.

## Lỗi mới `mask` (che giấu suy giảm; thiết kế TRƯỚC khi xem bất kỳ kết quả nào; không dùng để train hay chọn phương pháp)
- 3 kênh ngẫu nhiên, seed SeedSequence([seed, unit, 99]).
- Khoảng thời gian: từ onset ~ U(0.60, 0.75) tuổi thọ đã qua đến hết đời.
- Giá trị = mức tham chiếu khoẻ + (giá trị − trung bình trượt 15 bước). Mức tham chiếu khoẻ là trung bình của kênh trong 30% đầu đời của unit.
- C-MAPSS tính trong đơn vị đã chuẩn hoá theo regime.

Lỗi này giữ dao động ngắn hạn nhưng xoá xu hướng suy giảm, nên được kỳ vọng làm predictor lạc quan ở mọi họ, kể cả vòng bi và HDD.

## Đánh giá (`scripts/ext29.py`)
Script dẫn xuất từ `ext25_affected.py`; predictor, selector và `rpt_auto` giống `ext22.py`.
- **Kịch bản:** clean, 6 lỗi gốc, `mask`.
- **Chỉ số trên window bị ảnh hưởng** như EXT25: `risk_A`, `excess_A`, `cov_A`.
- **Mô phỏng bảo trì** (phụ, mô tả). Ngưỡng 90% cố định trên calibration sạch (ngưỡng triển khai được). Mỗi unit test duyệt các
  window theo thời gian:
  - **Window bị từ chối:** kiểm tra (đếm 1 lần kiểm tra). Kiểm tra cho biết RUL thật; thay thế nếu RUL thật ≤ h.
  - **Window được nhận:** thay thế nếu RUL dự đoán ≤ h.
  - **Không thay thế trước window cuối:** tính là hỏng.

  Trong đó h = `critical_rul` của cấu hình (C-MAPSS 30 chu kỳ; các họ khác 0.24 tuổi thọ).

  Chỉ số:
  - tỷ lệ hỏng;
  - số lần kiểm tra trên mỗi unit;
  - tuổi thọ lãng phí = (RUL thật lúc thay − h)⁺ / RUL thật ở window đầu;
  - chi phí = 1 · hỏng + c_i · số lần kiểm tra + 0.5 · lãng phí, với c_i ∈ {0.01, 0.05}.

## Kiểm định xác nhận (vòng 5, α_5 = 0.0015625; tổng α sau vòng này = 0.0484375 ≤ 0.05)
- **Chính:** Δ`excess_A` (`rpt_auto` − `full`), trung bình 6 lỗi gốc, cho từng họ trong 5 họ.
  - Wilcoxon hai phía trên 20 cặp; Holm trên 5 ở mức α_5.
  - Gộp như thường lệ: subset C-MAPSS trung bình trong cặp; fold vòng bi theo trọng số unit.
- **Thành công:** giảm có ý nghĩa ở ít nhất 1 họ, và không họ nào tăng có ý nghĩa.

## Phụ (mô tả hoặc thăm dò; Holm trong từng họ kiểm định ở mức 0.05)
- **S1:** Δ`excess_A` trên lỗi `mask`, `rpt_auto` − `full`, 5 họ.
- **S2:** `risk_matched` (6 lỗi), `rpt_auto` − `full`, 5 họ. Lặp lại thước đo chính cũ.
- **S3:** `rpt_L` − `full` trên Δ`excess_A`.
- **S4:** chi phí bảo trì, `rpt_auto` − {`full`, `gauss_bias_aware`, `ensemble_std`}, mỗi c_i; tỷ lệ hỏng.
- Bootstrap phân tầng (như EXT28) cho thước đo chính, nếu có thời gian.
- Mọi kết quả đều được báo cáo, kể cả âm.

## Sai lệch, ghi ngày 2026-11-15, TRƯỚC khi có bất kỳ kết quả nào của vòng 5
- **Thiết bị:** theo đồng ý của người dùng, phần huấn luyện chuyển sang GPU (RTX 4090) cho nhanh hơn.
  - Run ghi vào `runs/conf112g_cmapss` và `runs/conf112g_ext`.
  - Lần chạy CPU dang dở (`conf112_*`, khoảng 10 phút) bị dừng và không dùng.
  - Seed, cấu hình và phương pháp không đổi. Các vòng trước train trên CPU, nên kết quả số học của từng run không so trực tiếp với các vòng đó (seed cũng đã khác).
- **Smoke test trên run vòng 4, chỉ để kiểm tra code:**
  - `mask` làm predictor lạc quan hơn trên C-MAPSS và pin. Trên vòng bi gần như không đổi, vì predictor vốn đã lạc quan khoảng 58% ở cuối đời.
  - Mô phỏng bảo trì gần như không có ca hỏng, do số lần kiểm tra lớn.
  - Thiết kế đã chốt được giữ nguyên, không sửa. Hạn chế này sẽ được báo cáo.
- **Sai lệch vận hành:** 3 run HDD (gru seed 112, tcn seed 112 và 117) lỗi hết RAM khi khởi động, lúc EXT28 còn chạy song song.
  - Ba run này được train lại bằng cùng lệnh, sau đó đánh giá bổ sung.
  - EXT28 sẽ chạy tiếp sau EXT29, đợi đến lúc đó mới chạy.

## Kết quả, ngày 2026-11-15: XÁC NHẬN
- Δ`excess_A` giảm có ý nghĩa ở mức α_5 trên C-MAPSS (−0.89) và pin (−1.96); N-CMAPSS −0.80 (p_Holm = 0.0024); không họ nào tăng.
- Phụ:
  - lỗi `mask` chỉ giảm được trên C-MAPSS;
  - bảo trì: không có lợi về chi phí, và trên C-MAPSS kém hơn.

Chi tiết xem FINDINGS.md.
