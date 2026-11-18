# EXT11: ablation, đổi bộ phân loại và so sánh SOTA cho `full_recon_rpt`. Chốt ngày 2026-11-12, TRƯỚC khi chạy

Lý do: leader góp ý bản nháp ACM (2026-11-12):
1. chưa giải thích vì sao dùng gradient boosting;
2. chưa có ablation các thành phần;
3. chưa so với các phương pháp SOTA.

## Nguyên tắc
- **Phương pháp chính đã chốt ở EXT10 (`full_recon_rpt`, gọi là OURS) và không đổi vì kết quả EXT11.** Nếu một biến thể
  trong EXT11 tốt hơn, chỉ báo cáo mô tả. Muốn đổi phương pháp chính thì phải mở vòng xác nhận 2 theo `EXT9_PROTOCOL.md`
  (seed 72–81, α_2 = 0.0125).
- Dữ liệu: seed 62–71 (240 run, 20 cặp (backbone, seed) mỗi dataset), cùng các backbone đã train cho EXT10.
  - Các seed này đã dùng **một lần**, cho phép kiểm định OURS so với `full`.
  - Các so sánh ở đây là giả thuyết mới, không dùng để chọn phương pháp.
- Metric và cách gộp giống `conf_test.py`:
  - risk @90% matched coverage, trung bình 6 lỗi chưa thấy;
  - C-MAPSS lấy trung bình trong cặp, vòng bi gán trọng số theo số unit;
  - Wilcoxon hai phía, trên các cặp.
- Kiểm tra tính nhất quán: `full` và `full_recon_rpt` tính lại trong EXT11 phải khớp EXT10 (`ext10_conf62`). Nếu lệch
  > 0.01 điểm thì dừng và tìm nguyên nhân trước khi báo cáo.

## A. So sánh SOTA (họ kiểm định chính của EXT11)
Code:
- `scripts/ext11_sota.py`: các baseline phải train thêm model;
- `scripts/ext11_ablate.py`: các baseline dựa trên HGB.

Mọi selector xếp hạng **cùng một** dự đoán của point ensemble.

| Tên | Định nghĩa |
|---|---|
| `ensemble_std` | độ lệch chuẩn ensemble (lấy từ EXT10) |
| `mc_dropout` | MC dropout, T = 20, trên ensemble đã train |
| `deep_ens_nll` | 3 thành viên Gaussian-NLL, độ lệch chuẩn dự đoán tổng (Lakshminarayanan 2017) |
| `evidential` | Deep Evidential Regression (Amini và cộng sự, NeurIPS 2020): 1 model, đầu Normal-Inverse-Gamma, λ = 0.01, score = độ lệch chuẩn dự đoán Student-t |
| `selectivenet` | SelectiveNet cho hồi quy (Geifman và El-Yaniv 2019), coverage mục tiêu 0.9 |
| `cqr_lower` | 3 thành viên hồi quy phân vị q = 0.1 (CQR, Romano 2019), score = dự đoán − q10 |
| `symmetric` | HGB giống `full` nhưng nhãn là \|E\| > δ |
| `fault_det` | HGB giống `full` nhưng nhãn là "cửa sổ bị lỗi" (bộ phát hiện lỗi) |
| `err_reg` | HGB hồi quy sai số có dấu E (dự đoán sai số, kiểu loss prediction); score = Ê |
| `gauss_bias_aware` | Gaussian có bias học được (lấy từ EXT10) |
| `ours_white` | OURS nhưng nhiễu loạn là **nhiễu trắng Gaussian** (augmentation nhiễu thường), xem mục B |

- Các head model (SelectiveNet, NLL, CQR, evidential) được train trên unit fit sạch và dừng sớm trên unit validation sạch.
  Cấu hình giống `extra_baselines.py` của bài ESANN.
- **Kiểm định:** OURS so với từng baseline trên 4 dataset, tức 11 × 4 = 44 phép thử. Holm trên cả 44 ở mức α = 0.05.
  Báo cáo hiệu trung bình, số cặp thắng và p sau Holm.

## B. Ablation các thành phần (thăm dò, Holm trong họ B)
Tất cả dùng đặc trưng của OURS (`fx` = đặc trưng `full` + 3 đặc trưng tái tạo), HGB và siêu tham số giữ nguyên. Chỉ đổi
dữ liệu huấn luyện selector. Các biến thể dùng cùng hạt giống bộ sinh với OURS (seed + 50000), để so sánh có ghép cặp.

| Tên | Thay đổi so với OURS |
|---|---|
| `ours_copy` | thêm bản sao **sạch, không nhiễu** (cùng trọng số), để kiểm tra lợi ích đến từ nhiễu hay chỉ từ việc thêm dữ liệu |
| `ours_nobias` | bỏ thành phần hằng m = 0 (a_0 = 0), để đo phần trùng với lỗi bias |
| `ours_noonset` | không bao giờ có thời điểm bắt đầu (nhiễu phủ cả cửa sổ) |
| `ours_amp_small` | A ~ LogUniform(0.25, 1) |
| `ours_amp_large` | A ~ LogUniform(2, 8) |
| `ours_k1` | luôn đúng 1 kênh |
| `ours_khalf` | k ~ U{1, …, max(1, round(0.5·n))} |
| `ours_x2` | 2 bản nhiễu độc lập cho mỗi cửa sổ (bản thứ hai dùng seed + 51000) |
| `ours_white` | nhiễu trắng N(0, A²) theo từng time step, cùng phân phối k, A và onset |
| `ours_gauss_all` | augmentation nhiễu kinh điển: N(0, σ²) trên mọi kênh, σ ~ LogUniform(0.1, 1), không có onset |
| `ours_noseen` | chỉ clean + rand, bỏ dropout và drift, để xem RPT có thay được các shift thiết kế tay không |
| `ours_oracle` | S + **6 lỗi đánh giá** trên unit selector. Đây là cận trên, không phải phương pháp hợp lệ |

Cùng với ma trận 2×2 đã có (`full`, `full_recon`, `full_rpt`, `full_recon_rpt`), lấy từ EXT10.

- **Kiểm định:** mỗi biến thể so với OURS trên từng dataset, Holm trên (biến thể × 4) ở mức 0.05, không tính `ours_oracle`.
- **Giả thuyết chốt trước:**
  - **H-B1 (cơ chế):** `ours_copy` kém OURS. Nếu không kém, lợi ích không thể quy cho nhiễu loạn.
  - **H-B2:** `ours_nobias` vẫn tốt hơn `full_recon` ở các lỗi không phải bias. Kiểm tra mô tả.

## C. Đổi bộ phân loại (thăm dò, Holm trong họ C)
Cùng đặc trưng, cùng dữ liệu, cùng trọng số theo unit. Chỉ đổi bộ phân loại:

| Hậu tố | Bộ phân loại |
|---|---|
| `_lr` | StandardScaler + LogisticRegression (C = 1) |
| `_rf` | RandomForest (200 cây, min_samples_leaf = 25, max_features = sqrt) |
| `_mlp` | StandardScaler + MLP (64, 64), alpha = 1e-3, early stopping |
| `_hgbdef` | HGB với siêu tham số mặc định của sklearn (31 lá, 20 mẫu mỗi lá, không L2), số vòng giữ nguyên |

Mỗi bộ phân loại được áp dụng cho hai cấu hình:
- `osc_*`: đặc trưng `f`, dữ liệu S;
- `ours_*`: đặc trưng `fx`, dữ liệu S ∪ rand.

**Kiểm định**, Holm trong họ C ở mức 0.05:
1. `osc_c` so với `full`, cho từng c, để biết HGB có phải lựa chọn hợp lý không.
2. `ours_c` so với `osc_c`, để biết lợi ích của RPT có giữ được khi đổi bộ phân loại không.

## Báo cáo
- Mọi kết quả của họ B và C được ghi là thăm dò.
- Bài báo phải nêu rằng EXT11 chạy **sau** khi leader góp ý, trên các seed đã dùng một lần.

## Kết quả ngày 2026-11-12
240 run, 0 lỗi; `full` và OURS khớp EXT10 chính xác. Chi tiết trong `extension-results/2026-11-09/ext11/FINDINGS.md`.
- A: OURS tốt hơn có ý nghĩa so với 8/8 baseline định lượng bất định và selective-prediction, ở 31/32 cặp
  (baseline × dataset).
  - So với hồi quy sai số và Gaussian có bias, chỉ tốt hơn có ý nghĩa ở C-MAPSS.
  - Không tốt hơn augmentation nhiễu trắng trên k kênh.
- B: H-B1 đúng: bản sao sạch kém OURS, có ý nghĩa ở 3/4 dataset. Thành phần hằng, onset và dạng hàm trơn đều không cần;
  biên độ lớn là yếu tố quan trọng. Oracle chỉ tốt hơn tối đa 0.16 điểm.
- C: HGB tốt hơn LR, RF và MLP; lợi ích của RPT giữ được với cả 4 bộ phân loại.
- Phương pháp chính không đổi, theo nguyên tắc đã chốt.
