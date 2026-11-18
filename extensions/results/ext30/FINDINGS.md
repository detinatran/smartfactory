# EXT30: chứng nhận LTT trên HDD (2026-11-15). Post hoc
60 run HDD:
- seed 92–101 và 112–121: 222 unit calibration mỗi run;
- hdd2 seed 102–111: 120 unit.

0 lỗi.

## Kết quả theo biên
- **Hoeffding, Bernstein:** 0% chứng nhận ở mọi f và mọi điểm số.
- **WSR betting:**
  - **f = 1.0** (không tệ hơn nhận hết): chứng nhận ở mức coverage 0.95 trong **100%** run có 222 unit, cho `full`, `gauss_bias_aware` và `rpt_auto`.
    - `err_reg`: 5–10%. `ensemble_std`: 0%.
    - Test sạch: risk 20.2–20.6% so với r* 24.2–24.7%; vi phạm 0%. Dưới lỗi cũng vi phạm 0%, vì lỗi làm predictor HDD bớt lạc quan.
  - **f = 0.85:** 10% run (5–10% tuỳ điểm số).
  - **f = 0.7:** 0%.
  - **hdd2 (120 unit):** 0% ở mọi f.
- **Plug-in** (không có biên): vi phạm 20–50% trên test sạch.
- **Coverage được chứng nhận, `rpt_auto` − `full`:** ≈ 0 (n.s.). Chứng nhận không phân biệt được các selector.

## Kết luận
- Đảm bảo hữu hạn mẫu đạt được với khoảng 200 unit có nhãn, nhưng chỉ cho mục tiêu yếu.
- Khớp với ước tính của ltt_v2 (cần 10²–10³ unit).
