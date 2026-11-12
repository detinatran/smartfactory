# EXT9 và quy tắc cho vòng lặp tìm phương pháp. Chốt ngày 2026-11-12, TRƯỚC khi tính bất cứ thứ gì mới

## Quy tắc chung cho mọi vòng từ EXT9 trở đi
Người dùng yêu cầu chạy tiếp cho đến khi có phương pháp tốt hơn và mới hơn. Lặp nhiều vòng làm tăng nguy cơ dương tính giả,
nên chốt các quy tắc sau.

1. **Phát triển:** dùng seed 42–61, tức 40 cặp (backbone, seed) mỗi dataset. Tất cả các seed này đều đã từng được xem.
2. **Xác nhận:** mỗi vòng chốt **một** phương pháp chính, rồi kiểm định trên một bộ **10 seed mới chưa từng dùng**, tức 20
   cặp mỗi dataset.
   - Vòng 1 (EXT9) dùng seed 62–71, vòng 2 dùng 72–81, và cứ thế tiếp.
   - Config giống hệt bài.
3. **Chia α theo vòng (alpha spending):** vòng thứ j dùng α_j = 0.05 / 2^j, tức 0.025, 0.0125, 0.00625, …
   - Tổng tỷ lệ dương tính giả qua mọi vòng không vượt quá 0.05, dù chạy bao nhiêu vòng.
   - Với 20 cặp, p nhỏ nhất khoảng 2e-6, nên ngưỡng vẫn đạt được qua nhiều vòng.
4. **Tiêu chí "tốt hơn ESANN"**, so với `full`, theo risk @90% matched trung bình 6 lỗi chưa thấy, Wilcoxon hai phía, Holm
   trên 4 dataset ở mức α_j:
   - tốt hơn có ý nghĩa ở ít nhất một dataset;
   - không kém có ý nghĩa ở dataset nào;
   - hiệu trung bình ≤ +0.3 điểm ở mọi dataset;
   - lỗi bi quan tăng ≤ 1 điểm ở mọi dataset.
5. **Tiêu chí "mới":** phương pháp phải có thành phần không có trong bài ESANN.
6. **Dừng** khi một vòng đạt cả hai tiêu chí. Mỗi vòng ghi vào `SEARCH_LOG.md`: số vòng đã thử, α đã dùng và kết quả. Bài
   báo phải nêu tổng số vòng đã thử.

## EXT9: bộ chọn lọc có cổng dư thừa giải tích (redundancy-gated selector)
**Lý do:** qua EXT6–EXT8, đặc trưng tái tạo và stacking giúp C-MAPSS một cách nhất quán (3 bộ seed), nhưng hại vòng bi. Giả
thuyết: tái tạo chỉ có ích khi các cảm biến **dư thừa giải tích**, tức mỗi kênh dự đoán được từ các kênh còn lại. Đây là
khái niệm kinh điển trong phát hiện và cô lập lỗi (FDI). Nếu không có dư thừa, giá trị tái tạo chỉ là nhiễu.

**Định nghĩa:**
- Với mỗi run, tính độ dư thừa ρ = trung vị theo kênh của R²_c = 1 − σ_c² / Var(x_c).
  - σ_c là độ lệch chuẩn phần dư của bộ tái tạo Ridge ở EXT6 (từ các kênh khác, cộng settings với C-MAPSS).
  - Tính trên time step sạch của unit fit, **chỉ dùng dữ liệu train**, không dùng nhãn lỗi hay dữ liệu đánh giá.
- Ngưỡng **τ = 0.9** ("90% phương sai giải thích được bởi các kênh khác"). Ngưỡng được chốt tại đây, **trước khi** tính ρ
  trên bất kỳ run nào.

**Ứng viên** (cổng ở mức run):

| Tên | Định nghĩa |
|---|---|
| `rg_stack_plus` | `stack_plus` nếu ρ ≥ τ, ngược lại `full` |
| `rg_stack` | `stack` nếu ρ ≥ τ, ngược lại `full` |
| `rg_full_recon` | `full_recon` nếu ρ ≥ τ, ngược lại `full` |
| `rg_blend` | `blend` nếu ρ ≥ τ, ngược lại `full` |
| `rg_soft` (chỉ là phân tích phụ) | λ·ECDF(`stack_plus`) + (1−λ)·ECDF(`full`), với λ = clip((ρ − 0.5)/0.4, 0, 1) |
| `minrank` (chỉ là phân tích phụ) | max(ECDF `full`, ECDF `stack_plus`): chỉ nhận cửa sổ khi cả hai đều thấy an toàn |

**Phát triển:**
- Vì cổng ở mức run, risk của các ứng viên `rg_*` trên seed 42–61 được lấy trực tiếp từ hàng của `stack_plus`, `stack`,
  `full_recon`, `blend` hoặc `full` trong `ext8_v1` và `ext8_hold`.
- ρ được tính bằng `scripts/ext9_r2.py`.
- Chọn phương pháp chính bằng quy tắc an toàn của EXT8 (40 cặp):
  - hiệu ≤ +0.10 điểm ở mọi dataset;
  - lỗi bi quan tăng ≤ 1 điểm;
  - trong số đó, chọn ứng viên có tổng hiệu âm nhất.
- Nếu không có ứng viên `rg_*` nào an toàn, EXT9 không xác nhận gì và chuyển sang vòng sau.

**Xác nhận:** seed 62–71, 20 cặp mỗi dataset, α_1 = 0.025. Code: `scripts/ext9_gate.py`. Đó là `ext8_select.py` thêm ρ và
các ứng viên ở trên, ngoài ra không đổi gì.

**Ghi chú trung thực:**
- Giả thuyết được hình thành **sau khi** đã biết tái tạo chỉ giúp C-MAPSS.
- Vì vậy bước xác nhận trên seed mới chỉ kiểm tra rằng lợi ích lặp lại và cổng hoạt động như dự kiến. Nó **không** chứng
  minh giả thuyết dư thừa trên một họ dữ liệu mới.
- Muốn chứng minh giả thuyết đó, cần thêm một dataset ngoài.

## Kết quả ngày 2026-11-12: giả thuyết bị bác bỏ ngay ở bước tính ρ, chưa dùng α nào
Tính ρ trên dữ liệu train (smoke test, seed 42) cho thấy:
- C-MAPSS có ρ ≈ 0.73, thấp nhất.
- N-CMAPSS có ρ ≈ 0.998; vòng bi ≈ 1.000 (34 đặc trưng gần như là hàm của nhau); pin ≈ 0.94–0.97.

Độ dư thừa **ngược** với dự đoán: tái tạo giúp C-MAPSS, dataset có ρ thấp nhất. Với τ = 0.9, cổng sẽ tắt ở C-MAPSS và bật ở
ba dataset còn lại, tức hoàn toàn ngược ý đồ. EXT9 dừng tại đây:
- không chạy giai đoạn phát triển;
- không dùng bộ seed xác nhận;
- không tiêu α.

Không đổi τ hay đảo chiều cổng sau khi đã thấy ρ, vì như vậy là chỉnh theo dữ liệu.
