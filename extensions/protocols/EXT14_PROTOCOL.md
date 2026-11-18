# EXT14: bảng so sánh đầy đủ cho phương pháp cuối (`rpt_L`) trên seed 72–81. Chốt ngày 2026-11-13, TRƯỚC khi chạy

**Lý do:** EXT12 xác nhận `rpt_L` ở vòng 2. Bảng so sánh SOTA (EXT11 và EXT13) mới chỉ có trên seed 62–71 với R1. Để bảng
chính của bài dùng phương pháp cuối và seed xác nhận mới, chạy lại các baseline trên cùng 240 run seed 72–81.

**Code** (không đổi, chỉ đổi thư mục run):
- `scripts/ext11_sota.py`: SelectiveNet, deep ensemble NLL, CQR, MC dropout, evidential;
- `scripts/ext13_more.py`: Laplace, Mahalanobis, kNN, Isolation Forest, residual dư thừa giải tích, TTA;
- `scripts/ext11_ablate.py`: bộ phân loại đối xứng, bộ phát hiện lỗi, hồi quy sai số, nhiễu trắng thưa, cùng ablation và
  phép đổi bộ phân loại của R1;
- ensemble spread và Gaussian có bias lấy từ `ext12_conf`.

**Kiểm định chính của EXT14:**
- `rpt_L` so với 17 baseline trên 4 dataset, tức 68 phép thử. Wilcoxon hai phía, Holm trên 68 ở mức 0.05.
- 17 baseline gồm: `ensemble_std`, `mc_dropout`, `deep_ens_nll`, `evidential`, `laplace`, `tta_std`, `mahalanobis`,
  `knn_feat`, `iforest`, `ar_residual`, `selectivenet`, `cqr_lower`, `symmetric`, `fault_det`, `err_reg`,
  `gauss_bias_aware`, `ours_white`.
- Kiểm tra nhất quán: `full`, `err_reg` và `full_recon_rpt` tính trong `ext11_ablate` phải khớp `ext12_conf` (sai lệch
  ≤ 0.01 điểm).

Phần ablation và đổi bộ phân loại (cho R1) trên seed 72–81 chỉ để mô tả, dùng làm bằng chứng lặp lại cho EXT11.

## Kết quả ngày 2026-11-13
240 run, 0 lỗi; kiểm tra nhất quán đạt. `rpt_L` tốt hơn có ý nghĩa ở 67/68 phép so sánh, không kém ở phép nào. Chi tiết
trong `extension-results/2026-11-09/ext14/FINDINGS.md`.
