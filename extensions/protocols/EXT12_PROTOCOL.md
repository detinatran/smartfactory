# EXT12: hướng tới bản đủ sức cho hội nghị A*. Chốt ngày 2026-11-13, TRƯỚC khi chạy

## Mục tiêu
Người dùng yêu cầu cải tiến bài và chạy thêm đến khi có khả năng A*. Từ EXT11 và góp ý của leader, các điểm reviewer A* sẽ
bắt bẻ:

| # | Điểm yếu | Thí nghiệm EXT12 |
|---|---|---|
| W1 | Lợi ích so với OSC nhỏ, và các bộ ước lượng sai số có hướng (`err_reg`, `gauss_bias_aware`) ngang OURS ngoài C-MAPSS | D + C: chọn và xác nhận phương pháp cuối, kiểm định cả so với các bộ có hướng; thử RPT làm sơ đồ huấn luyện chung cho các bộ đó |
| W2 | Biến thể đơn giản hơn (nhiễu trắng thưa) chưa được xác nhận | D + C |
| W3 | Lỗi đánh giá có thể trùng với họ nhiễu | S1: 4 loại lỗi **mới**, thiết kế sau khi đã chốt phương pháp |
| W4 | Chỉ một mức độ lỗi và 3 kênh | S2: thay đổi biên độ và số kênh |
| W5 | Chưa so với cách "làm predictor bền" | R: predictor train với cùng nhiễu loạn; so sánh ở mức hệ thống |
| W6 | Chỉ 2 backbone | B: thêm Patch-Transformer |

## Quy tắc chung
- Áp dụng `EXT9_PROTOCOL.md`. Đây là **vòng xác nhận 2**: seed mới **72–81**, α_2 = 0.0125. α đã tiêu từ trước là 0.025,
  nên tổng không vượt 0.0375.
- Metric và cách gộp như `conf_test.py`.
- Kết quả của S, R và B không được dùng để đổi phương pháp cuối.

## D. Phát triển: chọn phương pháp cuối (seed 42–61, 480 run đã có)
Code: `scripts/ext12_main.py`. Tất cả các score dùng HGB với siêu tham số cố định, cùng hạt giống bộ sinh seed + 50000.
Bản thứ hai (nếu có) dùng seed + 51000.

| Tên | Định nghĩa |
|---|---|
| `full` | OSC (ESANN) |
| `gauss_bias_aware`, `err_reg` | các bộ ước lượng có hướng (như EXT10 và EXT11) |
| `full_recon_rpt` | phương pháp đã xác nhận ở vòng 1 (R1) |
| `full_rpt` | RPT hàm trơn, không có đặc trưng tái tạo |
| `rptw` | đặc trưng `f`, S + 1 bản **nhiễu trắng thưa** (k và A như EXT10, có onset) |
| `rptw_L` | đặc trưng `f`, S + 2 bản nhiễu trắng thưa, A ~ LogUniform(1, 8), k ≤ round(0.5·n) |
| `rpt_L` | như `rptw_L` nhưng hàm trơn |
| `recon_rpt_L` | như `rpt_L` nhưng đặc trưng `fx` |
| `err_reg_rpt` | `err_reg` train trên S + rand (hàm trơn, như EXT10) |
| `gba_rpt` | Gaussian có bias train trên S + rand |

Ghi chú trung thực: cấu hình `_L` được gợi ý từ ablation EXT11 trên seed 62–71. Chúng không dùng seed 72–81, nên bước xác
nhận vẫn hợp lệ.

**Quy tắc chọn** (40 cặp, seed 42–61):
1. Với mỗi ứng viên c và dataset d, tính hiệu Δ_d(c) = mean(c) − min{mean(`full`), mean(`gauss_bias_aware`),
   mean(`err_reg`)}. Đây là hiệu so với **baseline tốt nhất của dataset đó**.
2. c được coi là an toàn nếu hiệu so với `full` ≤ +0.10 ở mọi dataset và lỗi bi quan tăng ≤ 1 điểm.
3. Trong các ứng viên an toàn, chọn c có Σ_d Δ_d(c) âm nhất. Hoà (chênh < 0.02) thì chọn ứng viên đơn giản hơn, theo thứ
   tự `rptw` < `full_rpt` < `rptw_L` < `rpt_L` < `err_reg_rpt` < `gba_rpt` < `full_recon_rpt` < `recon_rpt_L`.
4. Nếu chọn được `full_recon_rpt` thì vòng 2 xác nhận lại R1 trên seed mới, kèm so sánh với các bộ có hướng.

## C. Xác nhận vòng 2 (seed 72–81, 240 backbone GRU/TCN train mới, 20 cặp mỗi dataset)
- Gọi ứng viên đã chọn là FINAL.
- **Họ kiểm định chính:** FINAL so với {`full`, `gauss_bias_aware`, `err_reg`} trên 4 dataset, tức 12 phép thử. Wilcoxon
  hai phía, Holm trên 12, ở mức α_2 = 0.0125.
- **Tiêu chí thành công** (như EXT9 mục 4, so với `full`): tốt hơn có ý nghĩa ở ≥ 1 dataset, không kém có ý nghĩa ở
  dataset nào, hiệu ≤ +0.3 và lỗi bi quan tăng ≤ 1 điểm ở mọi dataset.
- **Tiêu chí phụ** (báo cáo, không quyết định thành công): số dataset mà FINAL tốt hơn có ý nghĩa so với **cả hai** bộ có
  hướng.

## S. Stress test (seed 72–81, mô tả, Holm trong từng họ)
Các phương pháp: `ensemble_std`, `full`, `gauss_bias_aware`, `err_reg`, `full_recon_rpt`, FINAL.

**S1, lỗi mới chưa từng dùng ở bất kỳ vòng nào** (thêm vào `faults.py` dưới tên `HELDOUT`, 3 kênh, cùng khoảng thời gian
như 6 lỗi cũ):

| Lỗi | Định nghĩa |
|---|---|
| `clip` | bão hoà: giá trị bị kẹp trong [mean − 0.5·scale, mean + 0.5·scale] |
| `lag` | trễ: giá trị tại t là giá trị tại t − 10 bản ghi |
| `quant` | lượng tử hoá: mean + 2·scale·round((v − mean) / (2·scale)) |
| `invert` | đảo cực: mean − (v − mean) |

**S2, mức độ và số kênh:**
- `bias@1.5`, `bias@6`
- `noise@1`, `noise@4`
- `spike@3`, `spike@12`
- `gain@1.25`, `gain@2`
- `stuck#1`, `stuck#6`
- `bias#1`, `bias#6`
- `noise#6`

**Kiểm định:** FINAL so với từng baseline trên từng nhóm (S1 trung bình 4 lỗi; S2 từng kịch bản), Holm trong mỗi họ ở mức
0.05.

## R. Predictor bền (seed 72–81)
- Code: `scripts/ext12_robust.py`. Với mỗi run, train lại ensemble 3 thành viên với cùng cấu hình, cùng split, cùng
  normalizer. Dữ liệu train = cửa sổ fit sạch + 1 bản nhiễu loạn (bộ sinh EXT10, nhãn RUL thật). Run được lưu dưới dạng
  thư mục mirror, sau đó chạy `ext12_main.py` trên đó.
- **So sánh ở mức hệ thống** (mỗi hệ thống dùng predictor riêng của nó), risk @90% matched coverage, 6 lỗi:
  - predictor gốc + FINAL;
  - predictor bền + `ensemble_std`;
  - predictor bền + `full`;
  - predictor bền + FINAL.
- Báo cáo thêm RMSE trên dữ liệu sạch. Wilcoxon, Holm trên (3 so sánh × 4 dataset).

## B. Backbone thứ ba (seed 72–76)
- Train `patch_transformer` (cấu hình mặc định của repo) trên 4 dataset, rồi chạy `ext12_main.py`.
- Mô tả: FINAL so với `full`, `gauss_bias_aware` và `err_reg` trên 5 seed, có Wilcoxon (không nằm trong họ chính).

## Sai lệch so với protocol, ghi ngày 2026-11-13
- **Giai đoạn B, N-CMAPSS:** cửa sổ N-CMAPSS dài 10 bước. Với cấu hình mặc định (patch 10, bước nhảy 5), Patch-Transformer
  chỉ có 1 patch và báo lỗi "at least two patches". Vì vậy dùng `configs/external_pt_ncmapss.json`, tức `external.json` với
  patch 4 và bước nhảy 2. Các dataset khác giữ cấu hình mặc định. Thay đổi được quyết định trước khi xem bất kỳ kết quả nào
  của B.
- **Script đánh giá:** `ext12_main.py` có thêm cột risk theo nhiều mức coverage (0.5–0.95) để vẽ đường cong risk–coverage.
  Các cột cũ không đổi. Bước C, S, R và B dùng bản này (`ext12_main_v2.py` trên van). Giai đoạn D chạy bằng bản trước, vốn
  cho cùng các cột cũ.

## Chốt FINAL, ngày 2026-11-13, TRƯỚC khi đánh giá seed 72–81
Giai đoạn D: 480 run, 0 lỗi. `full`, `gauss_bias_aware` và `full_recon_rpt` khớp EXT10 chính xác. Bảng đầy đủ trong
`extension-results/2026-11-09/ext12/select_output.txt`.

Hiệu so với baseline tốt nhất của từng dataset (min của `full`, `gauss_bias_aware`, `err_reg`), seed 42–61, 40 cặp:

| Ứng viên | C-MAPSS | N-CMAPSS | Vòng bi | Pin | Tổng | An toàn |
|---|---|---|---|---|---|---|
| `rpt_L` | −0.59 | −0.23 | −0.86 | −0.11 | **−1.79** | có |
| `recon_rpt_L` | −0.79 | −0.20 | −0.86 | +0.07 | −1.77 | có |
| `rptw_L` | −0.51 | −0.22 | −0.76 | −0.15 | −1.63 | có |
| `full_recon_rpt` (R1) | −0.58 | −0.07 | −0.29 | +0.19 | −0.75 | có |
| `err_reg_rpt` | +0.53 | −0.30 | −0.07 | −0.41 | −0.25 | không (C-MAPSS +0.53 so với `full`) |
| `gba_rpt` | +0.12 | −0.30 | +0.03 | −0.39 | −0.54 | không |

- **FINAL = `rpt_L`**: đặc trưng `f` (không tái tạo), S + 2 bản nhiễu hàm trơn, A ~ LogUniform(1, 8), k ≤ round(0.5·n).
  `recon_rpt_L` nằm trong khoảng hoà 0.02 điểm, nên theo quy tắc chọn ứng viên đơn giản hơn.
- Kiểm định: `python scripts/ext12_summary.py confirm all_ext12conf.csv.gz rpt_L` (Holm trên 12, α_2 = 0.0125).
- Ghi nhận: áp RPT cho các bộ ước lượng có hướng (`err_reg_rpt`, `gba_rpt`) giúp ở N-CMAPSS và pin, nhưng hại ở C-MAPSS,
  nên không an toàn.
