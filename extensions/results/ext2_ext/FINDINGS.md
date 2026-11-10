# EXT2 mở rộng sang vòng bi và pin — kết quả (2026-11-10)

Protocol: `research/selective-rul/EXT2_CONFIRM_PROTOCOL.md` (phần B), chốt trước khi chạy. Seed 42–46.
- Vòng bi: 6 điều kiện để riêng × 5 seed = 30 run. Gộp theo trọng số số unit của từng fold.
- Pin: 5 run.
- 3 backbone × 8 điểm, tổng 5 880 hàng, 0 run lỗi.
- Mô tả, chưa hiệu chỉnh. Chỉ có 5 cặp seed, nên p nhỏ nhất là 0.0625.

## Risk (%) ở coverage 90% matched, trung bình 6 lỗi chưa thấy
| Điểm | Pin: GRU | FiLM | Lipschitz | Vòng bi: GRU | FiLM | Lipschitz |
|---|---|---|---|---|---|---|
| ensemble_std | 3.85 | 4.34 | 3.61 | 29.25 | 29.40 | 29.49 |
| mc_dropout | 4.47 | 4.24 | 3.45 | 28.74 | 29.77 | 28.87 |
| laplace | 4.26 | 4.38 | 3.62 | 28.86 | 29.93 | 28.54 |
| gauss_unbiased | 3.40 | 4.10 | 3.24 | 30.14 | 30.28 | 30.22 |
| gauss_bias_aware | 1.89 | 2.43 | **1.82** | 27.06 | 26.75 | 28.38 |
| conformal | 2.03 | 2.53 | 1.91 | 28.02 | 27.67 | 28.54 |
| full | 2.64 | 3.41 | 2.37 | **26.75** | 26.82 | 28.06 |

## Theo giả thuyết
- **H3 (điểm có bias tốt hơn họ variance):**
  - Pin: **ủng hộ** trên cả 3 backbone; nhóm có bias giảm 1.2–2.0 điểm so với spread, 5/5 seed. Trên pin,
    gauss_bias_aware còn tốt hơn bộ phân loại `full`, khớp với unc_v1.
  - Vòng bi: **cùng chiều nhưng yếu hơn**; giảm 1.1–2.7 điểm, 3–5/5 seed, trên nền risk khoảng 27–30%.
  - Laplace và MC dropout chỉ chênh với spread trong khoảng ±1 điểm và không nhất quán về chiều. Ví dụ, Laplace trên Lipschitz/vòng bi tốt hơn 0.96 điểm (5/5); MC dropout hơi tốt hơn trên pin với FiLM/Lipschitz và trên vòng bi.
- **H1 (FiLM dịch chuyển chung nhiều hơn):**
  - Vòng bi: shift/spread cao hơn (0.68 so với 0.52, 5/5), nhưng độ dịch ở đây nhỏ hơn spread, nên common-shift không phải
    cơ chế chính.
  - Pin: **ngược chiều** (3.26 so với 3.82, 5/5).
  - Spread của FiLM loại **nhiều** lỗi do fault hơn trên cả pin và vòng bi, ngược với H1.
- **H2 (Lipschitz giảm độ dịch):**
  - Pin: hơi giảm (−0.005, 4/5); shift/spread thấp nhất (2.07 so với 3.82); risk thấp nhất ở đa số điểm.
  - Vòng bi: độ dịch **lớn hơn** và RMSE kém hơn.
- **H4 (độ phủ conformal giảm khi có lỗi): không giảm.**
  - Pin: 92–95% khi sạch, 91–94% dưới lỗi.
  - Vòng bi: **83–84%** ở cả hai trường hợp. Đã hụt mức 90% ngay trên dữ liệu sạch, vì calibration và đánh giá đến từ các
    điều kiện vận hành khác nhau (leave-one-condition-out). Như vậy biên conformal bền với lỗi cảm biến, nhưng **không bền
    với shift giữa các điều kiện vận hành**.
- **Ngưỡng frozen:**
  - Pin: conformal và gauss_bias_aware cho risk thấp nhất (0.6–1.1%).
  - Vòng bi: gauss_bias_aware và full thấp nhất (25.3–27.5%).

## Tổng hợp qua 4 họ dữ liệu (pilot + xác nhận + mở rộng)
| Kết luận | C-MAPSS | N-CMAPSS | Pin | Vòng bi |
|---|---|---|---|---|
| Điểm có bias tốt hơn mọi điểm họ variance | có | có | có | có (yếu) |
| Laplace / MC dropout không hơn spread rõ rệt (chênh ≤ 1 điểm, chiều không nhất quán) | có | có | có | có |
| Độ phủ conformal giữ dưới lỗi chưa thấy (giảm ≤ 1.5 điểm) | có | có | có | có (nhưng chỉ 83–84%) |
| FiLM: shift/spread lớn hơn GRU | có | có | **không** | có |
| FiLM + điểm có bias tốt hơn GRU + điểm có bias | **không** (xác nhận thất bại) | có (10/10 seed) | không | ngang |
| Lipschitz có lợi | không | không | hơi | không |


Câu chuyện vững cho bài mở rộng:
1. Mọi uncertainty dạng variance đều không bắt được lỗi lạc quan do sensor fault. Điều này đúng với ensemble thường, ensemble
   ngầm FiLM, MC dropout và Laplace post-hoc, trên 3 backbone và 4 họ dữ liệu.
2. Thứ còn thiếu là bias có điều kiện. Biên conformal một phía có bias giữ được độ phủ dưới lỗi cảm biến chưa thấy, nhưng
   không giữ được dưới shift giữa các điều kiện vận hành.
3. Đổi kiến trúc (ensemble ngầm, Lipschitz) không giải quyết vấn đề và không cho lợi ích nhất quán.
