# EXT40: vòng xác nhận 8, RPT-D trên 30 seed mới (162–191), kiểm định ở mức seed. Kết quả: XÁC NHẬN

1260 ensemble, 30/30 seed đầy đủ ở mọi họ, 0 run lỗi. Hai lần lệch protocol (thiếu RAM, chạy bù các run C-MAPSS còn thiếu) đã ghi trong protocol trước khi có kết quả.

Đơn vị phân tích là seed: mỗi seed lấy trung bình GRU và TCN. Kiểm định Wilcoxon, hiệu chỉnh Holm trên 5 họ, α_8 = 0.0001953125.

## Kết quả xác nhận: hiệu `rpt_auto_d` − `full`

| Họ | Hiệu (điểm) | Số seed cải thiện | p_Holm | CI 95% bootstrap theo seed | Thay đổi tỷ lệ pess |
|---|---|---|---|---|---|
| C-MAPSS | −0.64 | 30/30 | 9e-9 | [−0.72, −0.56] | +0.47 |
| N-CMAPSS (xoay unit) | −0.29 | 30/30 | 9e-9 | [−0.33, −0.24] | +0.01 |
| Vòng bi | −0.62 | 27/30 | 1.6e-7 | [−0.81, −0.45] | +0.26 |
| Pin (xoay unit) | −1.04 | 30/30 | 9e-9 | [−1.15, −0.95] | +0.23 |
| hdd | 0.00 | — | — | — | — |

- **hdd:** RPT-D không chuyển sang RPT ở deployment nào vì D < 0 ở mọi deployment, nên kết quả bằng đúng OSC.
- **CONFIRMED.**
  - (a) Không kém OSC trên hdd với biên 0.15: đạt (cận trên 0.00).
  - (b) Có ý nghĩa ở 4/4 họ benchmark.

## Mô tả
- **RPT − OSC:** C-MAPSS −0.64, N-CMAPSS −0.29, vòng bi −1.43, pin −1.04, hdd +0.41 (0/30 seed). Riêng RPT gây hại trên ổ cứng, và RPT-D tránh được hoàn toàn.
- **RPT-D so với các baseline định hướng:**
  - so với `gauss_bias_aware`: tốt hơn trên benchmark từ −0.16 đến −1.08 điểm; trên hdd kém hơn +0.12;
  - so với `err_reg`: tốt hơn trên benchmark từ −0.14 đến −1.53 điểm.
- **Bộ hồi quy lỗi có dấu huấn luyện với perturbation (`err_reg_rptL`):**
  - tốt hơn RPT-D trên vòng bi 1.01 điểm (27/30 seed), và nhỉnh hơn một chút trên N-CMAPSS (+0.07) và pin (+0.08);
  - kém hơn RPT-D trên C-MAPSS (−0.48) và hdd (−0.44).
  - Như vậy đóng góp chính là cách huấn luyện; bài đã nêu điều này.
- **Ở coverage 70%:** RPT-D giảm risk 0.62–1.16 điểm trên các họ benchmark.
- **Tỷ lệ chuyển sang RPT:** 100% trên C-MAPSS, N-CMAPSS và pin; 42% trên vòng bi; 0% trên hdd.
- **Khoảng cách tới oracle:**
  - OSC thu hẹp được 27/87/26/65% (C-MAPSS/N-CMAPSS/vòng bi/pin);
  - RPT-D thu hẹp được 34/90/34/81%, tức thêm 10/25/11/47% phần OSC còn để lại.

## Bài báo
Đã cập nhật:
- bảng `tab:confirm`: thêm khối vòng 8;
- abstract, multiplicity (α = 0.0498), protocol, mục HDD (bullet vòng 8), kết luận;
- đoạn "Headroom": thay số liệu vòng 7 bằng số liệu vòng 8.

Ở bản rút gọn, khối vòng 7 được bỏ khỏi bảng (vòng 8 thay thế) để giữ 8 trang.
