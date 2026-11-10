# EXT2 pilot — chốt ngày 2026-11-10, TRƯỚC khi có kết quả

Bối cảnh: leader gợi ý thử (i) các phương pháp post-hoc có cơ sở thống kê cho uncertainty, (ii) các block từ
FiLM-Ensemble (NeurIPS 2022) và LipschitzRNN (ICLR 2021), và lắp ghép ưu điểm. Kết quả dùng cho **bài mở rộng**,
không đưa vào bài ESANN. Code: `scripts/ext2_pilot.py`, `src/selective_rul/ext_backbones.py`. Hai backbone mới được
tự viết lại từ phương trình trong bài báo, không chép code từ các repo gốc (LipschitzRNN theo GPL-3.0). SALAD
(diffusion sinh chuyển động từ văn bản) không liên quan đến uncertainty hay RUL, nên không đưa vào.

## Dữ liệu và split
- Dùng lại đúng các run GRU đã có: cùng split, normaliser, cửa sổ và seed lỗi.
  - C-MAPSS FD001–FD004 (ext_v1, capped 125, tập development).
  - N-CMAPSS (39 unit test chính thức).
- Seed 42–46: 5 seed × 5 tập = 25 run.
- Không dùng official test của C-MAPSS. Vòng bi và pin chưa chạy ở pilot này.
- Chạy CPU trên van, `CUDA_VISIBLE_DEVICES=-1`.

## Backbone, cấu hình chốt trước (không tune trên tập đánh giá)
| Backbone | Cấu hình |
|---|---|
| `gru` | ensemble 3 GRU sẵn có (checkpoint cũ) |
| `film_gru` | 1 mạng GRU 2 lớp, hidden như run gốc; sau mỗi lớp GRU là LayerNorm không affine + FiLM riêng cho từng member; M = 8; khởi tạo γ ~ N(1, 0.5²), β ~ N(0, 0.5²); mỗi member có loss riêng trên cùng batch; early stopping theo RMSE của trung bình ensemble trên validation |
| `lipschitz` | 3 member độc lập, h ← h + ε(A h + tanh(W h + U x + b)), A, W dựng từ M_A, M_W với β = 0.75, γ = 0.001 (mặc định trong bài gốc); ε = 0.1 (bài gốc dùng 0.03 cho chuỗi 784 bước; cửa sổ ở đây chỉ 10–30 bước nên chọn trước ε = 0.1); hidden như run gốc; dropout 0.1 trước head |

Các thiết lập huấn luyện khác (lr, epoch, patience, batch, loss có trọng số unit, readout sigmoid) giống run gốc.

## Điểm số (giống uncertainty_v1, thêm 2 điểm)
- Họ variance: `ensemble_std`, `mc_dropout`, `mc_ensemble` (MC dropout T = 10), và **`laplace`**.
  - `laplace` là Laplace post-hoc ở lớp cuối: GGN cho readout sigmoid, prior precision 1, phương sai nhiễu bằng MSE trên tập fit.
    Lấy std dự đoán của từng member rồi trung bình.
- `gauss_unbiased`, `gauss_bias_aware`, bộ phân loại một phía `full`.
- **`conformal`** = b(x) + q·s(x), trong đó q là phân vị 90% (trọng số theo unit) của (e − b)/s trên cửa sổ calibration sạch.
  - Đây là biên một phía split-conformal: e ≤ b + q s.
  - Báo cáo độ phủ thực nghiệm của biên này trên tập đánh giá, cả khi sạch và khi có lỗi.
  - Lưu ý: các cửa sổ trong cùng một unit phụ thuộc nhau, nên đảm bảo 90% chỉ là xấp xỉ.

## Giả thuyết (chốt trước)
- **H1 (FiLM, common-shift):** dưới 6 lỗi chưa thấy, so với `gru`, spread của `film_gru` có `shift_over_spread` lớn hơn và
  `reject_new` không cao hơn. Nghĩa là các member dùng chung trọng số dịch chuyển cùng nhau nhiều hơn.
- **H2 (Lipschitz):** `abs_shift` (độ dịch dự đoán do lỗi, chuẩn hoá theo cap) của `lipschitz` nhỏ hơn `gru`.
  Báo cáo kèm RMSE sạch, vì giảm độ dịch mà mất độ chính xác thì không phải là lợi.
- **H3 (bias vs variance):** trên mỗi backbone, điểm có bias (`gauss_bias_aware`, `conformal`, `full`) có risk ở coverage 90%
  khớp (matched) thấp hơn mọi điểm họ variance (`ensemble_std`, `mc_*`, `laplace`). `laplace` không tốt hơn `ensemble_std`.
- **H4 (thống kê):** độ phủ của biên conformal xấp xỉ 90% trên tập sạch nhưng giảm khi có lỗi.
- **Lắp ghép:** báo cáo risk của mọi tổ hợp backbone × điểm. Không chọn "tổ hợp tốt nhất" để công bố chỉ dựa trên pilot này.
  Nếu một tổ hợp nổi bật, phải xác nhận bằng một vòng mới có seed 47–51 và protocol riêng.

## Phân tích
- Gộp như bài: trung bình 6 lỗi; C-MAPSS lấy macro qua subset.
- Cặp so sánh là 5 seed (C-MAPSS lấy trung bình 4 subset trong mỗi seed). Wilcoxon chưa hiệu chỉnh và mô tả thuần.
- Tất cả là **exploratory** (pilot). Với 5 cặp, p nhỏ nhất của Wilcoxon hai phía là 0.0625, nên chủ yếu đọc theo hướng và độ lớn.
