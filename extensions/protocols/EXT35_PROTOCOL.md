# EXT35: vòng xác nhận 6, RPT-auto với margin chuyển m = 0.01 (`rpt_auto_m`). Chốt ngày 2026-11-15, TRƯỚC khi train và chạy

**Kết quả EXT34** (phát triển, quy tắc chọn đã chốt trước): chỉ quy tắc M0.01 thoả điều kiện an toàn. Hiệu so với `full` trên dữ liệu phát triển:

| Họ | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD | hdd2 |
|---|---|---|---|---|---|---|
| `rpt_auto_m` − `full` (điểm) | −0.09 | −0.09 | −0.49 | −0.25 | 0.00 | 0.00 |

Quy tắc R0 hiện tại kém trên HDD (+0.08) và hdd2 (+0.12).

## Phương pháp
- `rpt_auto_m` = `rpt_L` nếu risk calibration trung bình trên S của `rpt_L` < của `full` − 0.01; nếu không thì `full`.
- Mọi thứ khác giữ nguyên như `rpt_auto` của ext22.py.

## Dữ liệu
- **Seed mới 142–151.** GRU và TCN cho C-MAPSS FD001–4, N-CMAPSS, vòng bi và pin; config giống vòng 4 và vòng 5. Train trên GPU.
- **hdd3:** 4 model ổ cứng hạng 9–12, chưa từng dùng (ST16000NM001G, ST12000NM001G, TOSHIBA MG08ACA16TA, ST12000NM0007).
  - 679 ổ hỏng, 197 ổ test.
  - `configs/external_hdd3.json`, seed 142–151.
- Tổng: 240 + 20 = 260 ensemble mới.

## Kiểm định xác nhận (α_6 = 0.00078125; tổng α sau vòng = 0.04921875)
- **Họ kiểm định:** `rpt_auto_m` − `full`; risk @90% matched, trung bình 6 lỗi; 5 họ (4 benchmark + hdd3). Wilcoxon hai phía, Holm trên 5.
- **Thành công:**
  - giảm có ý nghĩa ở ≥ 1 họ;
  - không họ nào tăng có ý nghĩa;
  - hiệu trung bình ≤ +0.3 ở mọi họ;
  - tỷ lệ bi quan tăng ≤ 1 điểm.
- **Phụ, đã chốt:** không kém trên hdd3 theo bootstrap seed (B = 20000). Cận trên của CI 90% cho `rpt_auto_m` − `full` < +0.3.
- **Mô tả:**
  - `rpt_auto` (R0) và `rpt_L` trên cùng các run;
  - số run chuyển sang `rpt_L`;
  - so sánh với `gauss_bias_aware` và `err_reg`.
- Mọi kết quả đều được báo cáo.

## Kết quả, ngày 2026-11-15: KHÔNG XÁC NHẬN
Không họ nào đạt α_6. Quy tắc margin hầu như không chuyển sang `rpt_L`. Chi tiết xem FINDINGS.md.
