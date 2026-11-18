# EXT14: 17 baseline so với phương pháp cuối `rpt_L` (seed 72–81, 240 run, 0 lỗi)

Protocol: `selective-rul/EXT14_PROTOCOL.md`. Chi tiết: `summary_output.txt`, `familyEXT14.csv`. Bảng LaTeX:
`paper-acm-rpt/v3/table_sota.tex`.

- **Nhất quán:** `full`, `err_reg` và `full_recon_rpt` tính lại khớp `ext12_conf` chính xác (sai lệch 0.0).
- **Tổng quát:** `rpt_L` tốt hơn có ý nghĩa ở **67/68** phép so sánh (Holm trên 68, α = 0.05) và không kém có ý nghĩa ở
  phép nào.
  - So với 14 baseline định lượng bất định, selective prediction, OOD và bất thường: −0.9 đến −8.3 điểm, thắng 18–20/20
    cặp.
  - So với các bộ ước lượng có hướng: −0.25 đến −1.79 điểm ở mọi dataset.
  - Ngoại lệ duy nhất là biến thể nhiễu trắng thưa của chính phương pháp (cấu hình vòng 1, có tái tạo) ở C-MAPSS: +0.05,
    không có ý nghĩa.
- **Dữ liệu sạch:** `rpt_L` có risk thấp nhất ở cả 4 dataset.
- **Lỗi bi quan:** cao hơn baseline đối xứng hoặc phân tán tốt nhất 2.5–4.8 điểm, đây là đánh đổi của thiết kế một phía.
