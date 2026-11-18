# EXT40: vòng xác nhận 8, vòng tổng hợp cho RPT-D trên 30 seed mới, kiểm định ở mức seed. Chốt ngày 2026-11-16, TRƯỚC khi train và chạy

**Lý do (review lần 12, điểm yếu 3):**
- Các vòng trước chỉ có 10 split độc lập. Test ở mức cặp coi 20 cặp GRU/TCN là độc lập, còn test ở mức seed thì không thể đạt α với 10 seed.
- Phương pháp khuyến nghị (`rpt_auto_d`) mới được xác nhận một lần.
- Vòng này kiểm định riêng RPT-D, coi đó là giả thuyết chính duy nhất, trên 30 seed mới, với đơn vị độc lập là seed.

## Phương pháp (cố định, KHÔNG thay đổi gì so với EXT38)
- `rpt_auto_d`: dùng `rpt_L` nếu D > 0, ngược lại dùng `full` (OSC).
- Code đánh giá: `scripts/ext38.py`, giữ nguyên.

## Dữ liệu
- **Seed mới:** 162–191 (30 seed, chưa từng dùng; đã kiểm tra trên van không có config nào dùng các seed này).
- **Backbone:** GRU và TCN, cấu hình như vòng 7, train trên GPU.

| Họ | Thiết lập |
|---|---|
| C-MAPSS | FD001–FD004 |
| N-CMAPSS | `ncmapss_cv`: xoay 5 fold trên 99 engine, nên unit test cũng thay đổi |
| Pin | `battery_cv`: xoay 5 fold trên 124 cell |
| Vòng bi | 6 fold leave-one-condition-out như trước |
| Ổ cứng | `hdd`: 4 model gốc, seed mới; chỉ dùng cho kiểm tra không kém |

## Kiểm định xác nhận (α_8 = 0.0001953125; tổng α sau vòng = 0.0498046875)
- **Đơn vị:** seed. Với mỗi seed, lấy trung bình GRU và TCN của risk @90% matched (trung bình 6 lỗi), trung bình qua fold. Vòng bi dùng trọng số theo số unit như trước, các họ còn lại trọng số đều.
- **Họ kiểm định:** hiệu `rpt_auto_d` − `full` trên 5 họ. Wilcoxon signed-rank hai phía trên 30 hiệu mức seed, Holm trên 5.
- **Thành công** (cùng tiêu chí với các vòng trước):
  - giảm có ý nghĩa ở ≥ 1 họ;
  - không họ nào tăng có ý nghĩa;
  - hiệu trung bình ≤ +0.3 ở mọi họ;
  - tỷ lệ bi quan tăng ≤ 1 điểm.
- **Phụ, đã chốt:**
  - **(a) Không kém trên `hdd`, biên chặt hơn: 0.15 điểm.** Lấy CI 90% bằng bootstrap theo seed (20 000 lần). Biên 0.15 bằng khoảng một nửa lợi ích nhỏ nhất đã xác nhận trên benchmark (N-CMAPSS khi xoay unit, −0.25).
  - **(b) Báo cáo:** số họ benchmark có ý nghĩa ở α_8 (mức seed), và CI 95% theo seed của từng họ.
- **Mô tả:**
  - risk ở coverage 70%;
  - tỷ lệ khoảng cách tới oracle mà mỗi phương pháp thu hẹp được (`ext39_oracle_gap.py`);
  - phân phối D;
  - tỷ lệ deployment chuyển sang `rpt_L`;
  - so sánh với `gauss_bias_aware`, `err_reg`, `err_reg_rptL`, `rpt_L`.

## Lệch protocol
Mọi lỗi train (bị retry) hoặc thiếu run sẽ được ghi ở đây. Một seed chỉ được đưa vào phân tích khi có đủ cả GRU và TCN, ở tất cả các fold.

### Lệch protocol 1, ngày 2026-11-16 (ghi khi chưa có kết quả nào)
- **Sự cố:** lần train đầu chạy 24 tiến trình song song nên hết RAM (MemoryError, OpenBLAS). Nhiều run C-MAPSS và một phần run external bị lỗi.
- **Xử lý:**
  - Dừng launcher trước bước eval.
  - Chạy lại toàn bộ job, tối đa 8 tiến trình cùng lúc, hai lượt. Run đã có `complete.json` được bỏ qua.
  - Sau đó mới chạy eval.
- Cấu hình, seed và phương pháp không thay đổi.

### Lệch protocol 2, ngày 2026-11-16 (ghi khi chưa có kết quả nào)
- **Thiếu run:** sau hai lượt chạy lại vẫn thiếu 20 run C-MAPSS. Nguyên nhân là `run_matrix.py` không cho dùng lại batch dir đã có (FileExistsError).
- **Xử lý:**
  - Dừng bước eval đang chạy dở.
  - Chuyển 45 file CSV có thể chưa ghi xong vào `runs/ext40_quarantine`; các file này sẽ được tính lại, kết quả xác định nên không đổi.
  - Train các seed còn thiếu trong batch dir mới (`fill_*`), rồi chạy lại eval. Eval bỏ qua các CSV đã có.
- Không thay đổi phương pháp hay seed.

## Kết quả, ngày 2026-11-16: XÁC NHẬN
RPT-D tốt hơn OSC trên cả 4 họ benchmark ở mức seed:
- C-MAPSS −0.64, N-CMAPSS −0.29, vòng bi −0.62, pin −1.04 điểm;
- 27–30/30 seed cải thiện, p_Holm ≤ 1.6e-7.

Trên hdd kết quả bằng đúng OSC: D < 0 ở mọi deployment, nên RPT-D luôn chọn OSC. Đạt tiêu chí không kém với biên 0.15. Chi tiết xem FINDINGS.md.

## Phân tích sau (post hoc, KHÔNG đặt trước), ngày 2026-11-16: chi phí ở coverage tối ưu chi phí (`scripts/ext42_cost.py`)
- **Mô hình chi phí:** mỗi window từ chối tốn 1 (kiểm tra); mỗi lỗi lạc quan được chấp nhận tốn R. Mỗi selector chọn coverage tối ưu trên lưới {0.5, …, 0.95, 1}. Coverage được chọn theo từng run trên đường risk–coverage của tập test, giống nhau cho mọi selector, nên các con số là cận lạc quan.
- **Dưới 6 loại lỗi, RPT-D giảm chi phí so với OSC ở cả 4 họ và mọi R từ 2 đến 50:**
  - R = 2: giảm 2.0–11.5%;
  - R = 10: giảm 5.9–28.2%;
  - R = 50: giảm 8.0–37.8%;
  - 26–30/30 seed cải thiện, p ≤ 4e-6.
- **hdd:** chi phí bằng OSC.
- **Dữ liệu sạch:** C-MAPSS không giảm khi R ≥ 10.
- Kết quả lưu ở `ext42_cost.txt`.
- **Bổ sung, ngày 2026-11-16 (post hoc):** risk của RPT-D thấp hơn OSC ở mọi coverage cố định từ 50 đến 95%, trên cả 4 họ benchmark (thấp hơn 0.07–1.65 điểm, 24–30/30 seed). Như vậy, với cùng ngân sách từ chối, RPT-D có chi phí thấp hơn ở mọi tỉ lệ chi phí. Bài dùng kết quả này làm phát biểu chính về chi phí; bản phân tích theo coverage tối ưu chuyển xuống làm phân tích phụ.

## Chi phí có tính lỗi bi quan (post hoc), ngày 2026-11-17
Xem `cost_with_pessimistic.txt`. Chi phí tính bằng r·lạc quan + bi quan, ở coverage 90%. ELPIS-G rẻ hơn OSC ở cả 4 họ ngay từ r = 1 (−0.17 đến −0.81 điểm, 23–30/30 seed).
