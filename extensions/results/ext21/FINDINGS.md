# EXT21: kiểm định ngoài trên ổ cứng hỏng thật (Backblaze SMART), KẾT QUẢ ÂM cho RPT

Protocol: `selective-rul/EXT21_PROTOCOL.md`, kèm sai lệch đã ghi trước khi so sánh. Dữ liệu dựng lại (EXT21b):
- 2 091 ổ hỏng thật thuộc 4 model, lịch sử 120–365 ngày, RUL bị cap ở 90 ngày;
- 612 ổ test; GRU/TCN × seed 42–51, tức 20 cặp; 0 lỗi.

Risk (%) @90% matched:

| | Ensemble spread | OSC (`full`) | Gaussian có bias | Hồi quy sai số | `rpt_L` |
|---|---|---|---|---|---|
| 6 lỗi | 21.35 | 15.44 | 15.28 | 15.26 | 15.80 |
| Sạch | 23.25 | 17.93 | 17.96 | 17.98 | 17.82 |
| Lỗi held-out | 23.48 | 17.84 | 17.85 | 17.88 | 17.95 |

**Kiểm định chốt trước** (Holm trên 4, α = 0.05), `rpt_L` − baseline:

| Baseline | Hiệu (điểm) | Số cặp `rpt_L` tốt hơn | p sau Holm |
|---|---|---|---|
| `full` | **+0.36** | 0/20 | 8e-6 |
| `gauss_bias_aware` | **+0.52** | 1/20 | 8e-6 |
| `err_reg` | **+0.54** | 0/20 | 8e-6 |
| `ensemble_std` | −5.55 | 20/20 | 8e-6 |

- Lỗi bi quan: `rpt_L` − `full` = −0.13.

**Diễn giải:**
- Trên HDD, perturbation training **làm hại** selector khi có lỗi (kém OSC và các bộ có hướng). Trên dữ liệu sạch thì ngang
  hoặc tốt hơn một chút.
- Lỗi không gây hại ở đây: risk dư âm với mọi phương pháp, và tỷ lệ lạc quan ở 100% coverage giảm từ 24.0 (sạch) xuống 22.5.
- Predictor dựa trên SMART yếu.
- Điều kiện mà RPT nhắm tới (lỗi làm predictor lạc quan) không xảy ra trên HDD, và RPT không tổng quát hoá sang tình huống
  này.
- **Kết luận chính về một phía vẫn đúng:** mọi selector một phía hoặc có hướng đều tốt hơn ensemble spread khoảng 5.5–6 điểm.
