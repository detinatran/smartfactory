# EXT48: vòng xác nhận 9. Phần lợi ích riêng cho lỗi của RPT-D (excess trên window bị ảnh hưởng), kiểm định ở mức seed, 30 seed mới. Chốt ngày 2026-11-16, TRƯỚC khi train và chạy

**Lý do (review lần 4, điểm yếu W1):**
- Excess do lỗi trên window bị ảnh hưởng mới chỉ được xác nhận cho `rpt_auto` (vòng 5: C-MAPSS và pin).
- Phương pháp khuyến nghị là `rpt_auto_d`, nhưng chưa từng được kiểm định trên endpoint này.
- Vòng này dùng đúng endpoint đó làm endpoint chính cho `rpt_auto_d`.

## Phương pháp (cố định, KHÔNG đổi)
- `rpt_auto_d` = `rpt_L` nếu D > 0, ngược lại `full`.
- D và các selector giống hệt EXT38/EXT40.
- Code: `scripts/ext48.py`, tức ext29.py (vòng 5) cộng thêm D và `rpt_auto_d`, `err_reg_rptL`, `gauss_rptL`.

## Dữ liệu
- **Seed mới:** 192–221 (30 seed, chưa từng dùng).
- **Thiết lập:** như vòng 8. C-MAPSS FD001–4, `ncmapss_cv` và `battery_cv` (xoay unit), vòng bi (6 điều kiện), `hdd` (4 model gốc). GRU và TCN.
- **Train:** trên GPU, tối đa 8 tiến trình. Chạy bù các run lỗi; seed nào thiếu run sau khi chạy bù sẽ bị loại và ghi vào đây.

## Kiểm định xác nhận (α_9 = 0.00009765625; tổng α sau vòng = 0.04990234375)
- **Endpoint chính:** Δexcess_A = excess trên window bị ảnh hưởng (risk dưới lỗi trừ risk trên chính các window đó khi sạch, ở coverage 90% matched), trung bình 6 lỗi, tính `rpt_auto_d` − `full`.
- **Đơn vị:** seed (trung bình GRU và TCN, trung bình qua fold; vòng bi có trọng số theo số unit).
- **Kiểm định:** Wilcoxon signed-rank hai phía trên 30 seed, Holm trên 5 họ.
- **Thành công:**
  - giảm có ý nghĩa ở ≥ 1 họ;
  - không họ nào tăng có ý nghĩa;
  - hiệu trung bình ≤ +0.3 điểm ở mọi họ.
- **Ghi chú đặt trước:** trên HDD, D < 0 ở hầu hết deployment, nên `rpt_auto_d` ≈ `full` và hiệu xấp xỉ 0 theo cấu trúc. Trên vòng bi và HDD, lỗi làm predictor *bớt* lạc quan (vòng 5), nên excess ở đó có thể âm với mọi selector.

## Phụ (đặt trước, α = 0.05 trong mỗi họ, NGOÀI ngân sách α)
- **(a)** Risk trên toàn tập test dưới 6 lỗi, `rpt_auto_d` − `full`, mức seed. Đây là lần lặp lại vòng 8.
- **(b)** Excess của lỗi health-masking (`mask`) trên window bị ảnh hưởng, `rpt_auto_d` − `full`.
- **(c)** Risk dưới lỗi: `rpt_auto_d` − `err_reg_rptL` và `rpt_auto_d` − `gauss_rptL`, tức so với các bộ ước lượng định hướng được huấn luyện cùng bản nhiễu. Holm trên 10.
- **(d)** Mô phỏng bảo trì với ngưỡng đóng băng trên calibration (như vòng 5): tỷ lệ hỏng, số lần kiểm tra trên mỗi unit, tỷ lệ tuổi thọ bị lãng phí. So `rpt_auto_d` với `full`.
- **(e)** Glitch pin tự nhiên: chạy `scripts/ext41.py` trên các run `battery_cv` mới, định nghĩa `g_main` như EXT41. Endpoint là risk trên window glitch được chấp nhận, `rpt_auto_d` − `full`. Suy luận chính là khoảng tin cậy bootstrap theo cell (95%, resample cell); Wilcoxon theo seed chỉ báo cáo kèm.

## Báo cáo
Mọi kết quả đều được báo cáo, kể cả khi trái với kỳ vọng.
- **Lệch protocol (vận hành), ngày 2026-11-16, trước khi có kết quả:** đủ 1260/1260 run đã train xong lúc 10:18. Launcher bị treo ở `Wait-Process` sau lượt train 1, có thể vì PID bị tái sử dụng. Đã dừng launcher, rồi chạy một script tiếp nối, script này làm phần eval và phân tích glitch y như kế hoạch. Không thay đổi gì về phương pháp hay dữ liệu.
- **Lệch protocol (vận hành) 2, ngày 2026-11-16, trước khi có kết quả:** EXT50 chạy song song làm đầy bộ nhớ GPU. Một worker eval (shard 0/6) gặp lỗi CUDA "illegal memory access", nên 120 run của shard này bị FAILED. Xử lý: dừng EXT50; chờ các worker còn lại chạy xong; chạy lại eval với 4 shard (CSV đã có được bỏ qua, chỉ chạy lại các run bị fail), rồi mới merge. Phương pháp và dữ liệu không đổi.

## Kết quả, ngày 2026-11-17: XÁC NHẬN
Δexcess_A của RPT-D − OSC giảm có ý nghĩa ở mức seed trên C-MAPSS (−1.10), N-CMAPSS (−0.71) và pin (−2.04), 30/30 seed. Vòng bi −0.04, không có ý nghĩa. HDD bằng 0. Kết quả phụ và chi tiết xem FINDINGS.md.
