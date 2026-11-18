# EXT18: đặc trưng tham chiếu lịch sử unit + RPT, vòng xác nhận 3. Chốt ngày 2026-11-14, TRƯỚC khi chạy

**Lý do:** sau EXT15 và EXT17, việc giảm risk dư do lỗi chỉ có ý nghĩa ở C-MAPSS. Selector hiện chỉ nhìn cửa sổ hiện tại,
nên không biết cửa sổ có "lệch so với chính unit đó lúc khoẻ" hay không. Khi triển khai, dữ liệu giám sát đến tuần tự, nên dùng
được cửa sổ đầu đời của unit làm tham chiếu (nhân quả, không nhìn trước).

## Đặc trưng lịch sử h(x)
- Tham chiếu = cửa sổ có cycle nhỏ nhất của cùng unit trong cùng tập. Với bản nhiễu RPT, tham chiếu lấy từ cửa sổ **sạch**
  của unit, vì nhiễu đóng vai một lỗi xảy ra sau thời điểm bắt đầu giám sát.
- Với mỗi kênh giá trị c:
  - d_c = trung bình theo thời gian của x_c − cùng đại lượng của tham chiếu;
  - r_c = log((std_c + 1e-3) / (std_ref,c + 1e-3)).
- Thêm max_c |d_c| và mean_c |d_c|.
- Tổng cộng 2n + 2 đặc trưng, nối vào đặc trưng OSC `f`.

## Ứng viên (`scripts/ext18_hist.py`; mọi thứ khác giống `ext12_main.py`)
| Tên | Định nghĩa |
|---|---|
| `full_h` | OSC + h, dữ liệu S (ablation: chỉ đặc trưng lịch sử) |
| `rpt_L_h` | `rpt_L` + h (RPT cấu hình L, 2 bản, cùng seed bộ sinh) |

Chạy kèm `ensemble_std`, `full`, `gauss_bias_aware`, `err_reg` và `rpt_L`, đều tái tạo đúng EXT12. Khi chạy trên seed xác
nhận, có thêm `--stress`.

## Giai đoạn phát triển (seed 42–61, 480 run)
`rpt_L_h` trở thành FINAL3 nếu thoả **cả ba** điều kiện:
1. Σ_d (`rpt_L_h` − `rpt_L`) ≤ −0.20 điểm;
2. không dataset nào kém `rpt_L` quá +0.10;
3. lỗi bi quan tăng ≤ 1 điểm so với `full`.

Nếu không thoả, không mở vòng 3; `rpt_L` vẫn là phương pháp cuối, còn EXT18 được báo cáo như một hướng không thành công.

## Xác nhận vòng 3 (seed 82–91, 240 backbone mới, α_3 = 0.00625)
- **Họ kiểm định:** FINAL3 so với {`rpt_L`, `full`, `gauss_bias_aware`, `err_reg`} × 4 dataset, tức 16 phép thử. Holm ở mức α_3.
- **Thành công** khi thoả cả hai:
  1. quy tắc EXT9 so với `full`;
  2. tốt hơn `rpt_L` có ý nghĩa ở ít nhất 1 dataset và không kém có ý nghĩa ở dataset nào.
- **Phụ:** risk dư (lỗi − sạch) của FINAL3 so với `rpt_L` theo từng dataset, cùng stress test.
- Tổng α sau vòng 3 là 0.025 + 0.0125 + 0.00625 = 0.04375 ≤ 0.05.

## Kết quả giai đoạn phát triển, ngày 2026-11-14: KHÔNG mở vòng 3
480 run, 0 lỗi. `full`, `rpt_L`, `gauss_bias_aware` và `err_reg` khớp EXT12 chính xác.

Risk (%), 6 lỗi, 40 cặp:

| | C-MAPSS | N-CMAPSS | Vòng bi | Pin |
|---|---|---|---|---|
| `rpt_L` | 8.66 | 11.36 | 22.30 | 3.73 |
| `rpt_L_h` | 7.36 | 11.43 | 22.19 | 3.73 |
| Hiệu | −1.30 | +0.07 | −0.11 | +0.00 |

- Tổng hiệu −1.34, đạt điều kiện 1. Không dataset nào kém quá +0.10, đạt điều kiện 2.
- Lỗi bi quan so với `full` tăng **+1.48** ở C-MAPSS, nên **không đạt điều kiện 3**.
- Risk dư do lỗi không giảm (C-MAPSS 0.14 → 0.35). Đặc trưng lịch sử chủ yếu mang thông tin suy giảm (độ lệch so với lúc
  khoẻ), không phải thông tin phát hiện lỗi.
- **Theo quy tắc, không mở vòng 3; `rpt_L` vẫn là phương pháp cuối.** Không tiêu α. Seed 82–91 (240 backbone đã train) chưa
  được đánh giá, nên vẫn là seed mới cho một vòng sau, nếu có.
