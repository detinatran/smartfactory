# EXT26: kiểm định ngoài chưa đụng tới, 4 model ổ cứng chưa dùng (hdd2), 2026-11-15
Protocol: `research/selective-rul/EXT26_PROTOCOL.md`.
- **Dữ liệu:** 1 160 ổ hỏng, 361 ổ test. Model: ST4000DM000, ST8000DM002, ST14000NM001G, HGST HUH721212ALE604.
- **Chạy:** seed 102–111, 20 ensemble mới, 0 lỗi.

| Điểm số | Risk @90% khi có lỗi (%) | Risk @90% sạch (%) |
|---|---|---|
| `ensemble_std` | 20.83 | 22.42 |
| `full` | 15.71 | 17.84 |
| `gauss_bias_aware` | 15.61 | 17.93 |
| `err_reg` | 15.66 | 17.96 |
| `rpt_L` | 16.33 | 17.81 |
| `rpt_auto` | 15.87 | 17.82 |

**Kiểm định chính** (Holm trên 3):

| `rpt_auto` − … | Hiệu | p_Holm |
|---|---|---|
| `full` | +0.16 (tốt hơn 0/20, kém 6/20) | 0.055 |
| `gauss_bias_aware` | +0.27 | 0.017 |
| `err_reg` | +0.21 | 0.17 |

- **Tiêu chí đã chốt** (không kém `full` có ý nghĩa, hiệu ≤ +0.3): **ĐẠT, nhưng sát biên.** `rpt_auto` kém Gaussian có bias có ý nghĩa.
- **`rpt_L` − `full`:** +0.62 (0/20). Kết quả âm của HDD lặp lại trên các model mới.
- **`rpt_auto` chọn `rpt_L` ở 6/20 run**, và lần nào chọn cũng làm risk kém đi. Bước kiểm tra trên calibration giới hạn được thiệt hại
  nhưng không loại bỏ được nó.
- Bài đã được cập nhật trung thực: abstract, mục HDD, kết luận.
