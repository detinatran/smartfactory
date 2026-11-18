# EXT34: quy tắc chuyển chặt hơn cho RPT-auto (giai đoạn phát triển). Chốt ngày 2026-11-15, TRƯỚC khi chạy

**Lý do:** review 9 và 10 (W7, W8).
- Quy tắc hiện tại R0 chọn `rpt_L` khi risk calibration trung bình trên S thấp hơn `full`.
- Trên 4 model ổ cứng chưa dùng (EXT26), R0 chuyển sang `rpt_L` ở 6/20 run, và lần nào chuyển cũng làm risk kém đi.
- Trên C-MAPSS, R0 giữ `full` ở 30/80 run, trong khi `rpt_L` tốt hơn ở hầu hết các cặp.

## Dữ liệu phát triển (không tiêu α)
- **Bốn họ benchmark:** các run seed 42–61 đã dùng ở EXT12/EXT22, 480 run.
- **HDD 4 model gốc:** seed 42–51, 92–101 và 112–121, 60 run.
- **hdd2 (4 model chưa dùng):** seed 102–111, 20 run.

Từ nay hdd2 thuộc dữ liệu phát triển. Kiểm định ngoài mới sẽ dùng hdd3 (model hạng 9–12), chưa từng được tải hay xem.

## Script `scripts/ext34_dev.py`
Selector `full` và `rpt_L` giống ext22.py. Với mỗi run, script ghi:
- risk calibration @90% matched cho từng shift đã thấy (clean, dropout, drift), cho cả `full` và `rpt_L`;
- a_u và o_u theo từng unit calibration của mỗi shift, phục vụ bootstrap;
- risk test @90% matched, trung bình 6 lỗi, cho `full` và `rpt_L`.

## Quy tắc ứng viên (đánh giá offline)
- **R0:** quy tắc hiện tại, so trung bình trên 3 shift.
- **M(m):** chỉ chọn `rpt_L` nếu cal_risk(`rpt_L`) < cal_risk(`full`) − m, với m ∈ {0.001, 0.0025, 0.005, 0.01} (tính theo đơn vị risk).
- **U:** `rpt_L` phải tốt hơn trên cả 3 shift.
- **B(q):** bootstrap có ghép cặp theo unit calibration (B = 500, seed 3434). Chọn `rpt_L` nếu P(risk của `rpt_L` < risk của `full`) ≥ q, với q ∈ {0.8, 0.9}.

## Quy tắc lựa chọn (chốt trước)
- Hiệu của mỗi quy tắc = risk test của selector được chọn − risk test của `full`, lấy trung bình theo cặp trong từng họ.
- Các họ được xét: C-MAPSS, N-CMAPSS, vòng bi, pin, HDD (gộp 60 run) và hdd2.
- **An toàn:** không họ nào có hiệu trung bình > +0.05 điểm.
- Trong các quy tắc an toàn, chọn quy tắc có tổng hiệu trên 6 họ nhỏ nhất. Nếu hai quy tắc chênh nhau ≤ 0.05 thì chọn quy tắc đơn giản hơn, theo thứ tự R0 < M (m nhỏ trước) < U < B.
- **Nếu R0 được chọn:** không mở vòng 6; RPT-auto giữ nguyên.
- **Nếu một quy tắc khác được chọn:** mở vòng xác nhận 6 với protocol riêng (EXT35). Dùng seed mới và hdd3; α_6 = 0.00078125, nên tổng α sau vòng 6 = 0.04921875.

## Kết quả, ngày 2026-11-15
Chọn M0.01, là quy tắc an toàn duy nhất. Mở vòng 6 (EXT35_PROTOCOL.md). Chi tiết xem `extension-results/2026-11-09/ext34/FINDINGS.md`.
