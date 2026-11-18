# EXT26: kiểm định ngoài thật sự chưa đụng tới, gồm các model ổ cứng chưa từng dùng (hdd2). Chốt ngày 2026-11-15, TRƯỚC khi xử lý dữ liệu và chạy

**Lý do (review 7, W3):** `rpt_auto` được thiết kế sau khi thấy kết quả âm trên HDD, và dùng HDD seed 42–51 để phát triển. Vì vậy
HDD không còn là kiểm định ngoài thật sự cho `rpt_auto`. EXT26 dùng ổ của **các model chưa từng dùng**, lấy từ chính 6 archive
Backblaze đã tải: không cần tải thêm, và chưa ai xem các model này.

## Dữ liệu (`scripts/prepare_hdd.py --ranks 5-8 --name hdd2`)
- Cách dựng giống hệt EXT21b:
  - giữ toàn bộ lịch sử, tối đa 365 ngày, và cần ≥ 120 ngày;
  - RUL cap ở 90 ngày;
  - 17 kênh SMART giống EXT21b, biến đổi log1p và forward-fill;
  - 30% ổ (theo hash serial) làm tập test;
  - regime = model.
- Model: hạng 5–8 theo số ổ hỏng đủ điều kiện. Không giao với 4 model của EXT21b.
- Một kênh hoàn toàn thiếu ở các model này được điền 0 (hằng số). Đây là thay đổi duy nhất của script và không ảnh hưởng `hdd`.

## Huấn luyện và đánh giá
- GRU và TCN, `configs/external_hdd.json` (cửa sổ 30, bước nhảy 3).
- **Seed mới 102–111**, tức 20 cặp.
- Đánh giá bằng `scripts/ext22.py`, không sửa gì: `rpt_auto`, `rpt_L`, `full`, `gauss_bias_aware`, `err_reg`, `ensemble_std`.

## Kiểm định (họ riêng cho kiểm định ngoài, như EXT21; không thuộc ngân sách α của vòng tìm kiếm)
- **Chính:** `rpt_auto` so với {`full`, `gauss_bias_aware`, `err_reg`}. Risk @90% matched, trung bình 6 lỗi. Wilcoxon hai phía, Holm trên 3, α = 0.05.
- **Thành công (đã chốt):** `rpt_auto` không kém `full` có ý nghĩa, và hiệu trung bình ≤ +0.3 điểm.
- **Phụ (mô tả):**
  - `rpt_L` so với `full`;
  - số run chọn `rpt_L`;
  - risk dư do lỗi;
  - lỗi bi quan;
  - risk trên dữ liệu sạch.
- Mọi kết quả, kể cả âm, đều được đưa vào bài.

## Kết quả, ngày 2026-11-15
- `rpt_auto` − `full` = +0.16 (p_Holm = 0.055). Tiêu chí đạt nhưng sát biên.
- `rpt_auto` kém `gauss_bias_aware` có ý nghĩa (+0.27, p_Holm = 0.017).
- `rpt_L` − `full` = +0.62 (0/20).

Chi tiết xem `extension-results/2026-11-09/ext26/FINDINGS.md`.
