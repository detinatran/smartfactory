# EXT34: quy tắc chuyển chặt hơn cho RPT-auto (phát triển, 2026-11-15)
560 run, 0 lỗi:
- 4 họ benchmark, seed 42–61;
- HDD seed 42–51, 92–101, 112–121;
- hdd2 seed 102–111.

## Hiệu so với `full` (điểm) và tỷ lệ run chọn `rpt_L`
| Quy tắc | C-MAPSS | N-CMAPSS | Vòng bi | Pin | HDD | hdd2 | Tổng | An toàn |
|---|---|---|---|---|---|---|---|---|
| R0 | −0.36 | −0.41 | −0.60 | −0.60 | +0.08 | +0.12 | −1.77 | không |
| M0.001 | −0.34 | −0.39 | −0.57 | −0.50 | +0.01 | +0.12 | −1.68 | không |
| M0.0025 | −0.27 | −0.34 | −0.58 | −0.50 | 0.00 | +0.06 | −1.64 | không |
| M0.005 | −0.18 | −0.24 | −0.53 | −0.39 | 0.00 | +0.06 | −1.27 | không |
| **M0.01** | −0.09 | −0.09 | −0.49 | −0.25 | 0.00 | 0.00 | −0.92 | **có** |
| U | −0.21 | −0.28 | −0.55 | −0.38 | 0.00 | +0.06 | −1.35 | không |
| B0.8 | −0.20 | −0.18 | −0.31 | −0.48 | 0.00 | +0.06 | −1.11 | không |
| B0.9 | −0.14 | −0.09 | −0.27 | −0.43 | 0.00 | +0.06 | −0.87 | không |

- Nếu luôn chuyển sang `rpt_L`: C-MAPSS −0.60, N-CMAPSS −0.47, vòng bi −0.86, pin −0.60, HDD +0.37, hdd2 +0.59.
- **Chọn: M0.01**, là quy tắc an toàn duy nhất. Nó mất phần lớn lợi ích trên C-MAPSS và N-CMAPSS để đổi lấy an toàn trên HDD.
- Theo protocol, mở vòng 6 (EXT35) trên seed mới 142–151 và hdd3.
