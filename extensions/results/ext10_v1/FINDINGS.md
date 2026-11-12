# EXT10: giai đoạn phát triển (seed 42–61). Kết quả ngày 2026-11-12

Protocol: `EXT10_PROTOCOL.md`. 480 run, 0 lỗi. Output đầy đủ: `summary_output.txt`. Đây là phân tích khám phá; p chưa hiệu
chỉnh.

Hiệu so với `full`, risk @90% matched, điểm (số cặp thắng trên 40):

| Ứng viên | C-MAPSS | N-CMAPSS | Vòng bi | Pin | Tổng |
|---|---|---|---|---|---|
| `full_recon_rpt` | −0.58 (40) | −0.30 (36) | −0.29 (25) | −0.32 (34) | −1.50 |
| `full_rpt` | −0.35 (35) | −0.34 (39) | −0.30 (25) | −0.45 (36) | −1.44 |
| `stack` (EXT7) | −0.48 (38) | −0.18 (28) | +0.85 (9) | −0.10 (26) | +0.10 |
| `full_recon` (EXT6) | −0.28 (40) | +0.05 (16) | +0.08 (18) | +0.21 (16) | +0.06 |

- Quy tắc an toàn chọn `full_recon_rpt` làm phương pháp chính để xác nhận trên seed 62–71.
- Kết quả xác nhận nằm ở `ext10_conf62/FINDINGS.md`.
