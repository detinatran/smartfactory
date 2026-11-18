# EXT64: cổng chọn predictor (mặc định hay gộp mặc định + RAMP-V13) theo risk calibration, kiểm trên seed mới 252–281 (THĂM DÒ, không tiêu α vì ngân sách đã hết; đặt trước đầy đủ). Chốt ngày 2026-11-18, TRƯỚC khi eval bất kỳ run nào của seed 252–281

## Nguồn gốc (post hoc, ghi rõ)
Quy tắc này được tìm ra sau khi đã xem dữ liệu EXT61 (seed 162–171) và EXT62 (seed 222–251). Kết quả trên các seed đó, cổng so với ELPIS-G trên predictor mặc định:

| Seed | Họ | Hiệu | Seed cổng tốt hơn |
|---|---|---|---|
| 162–171 | Vòng bi | −0.75 | 7/10 |
| 222–251 | Vòng bi | −0.97 | 22/30 |
| 222–251 | C-MAPSS | −0.56 | 29/30 |
| 222–251 | Pin | −0.63 | 30/30 |

Lỗi bi quan cũng thấp hơn ở mọi trường hợp. Vì đây là kết quả trên dữ liệu đã thấy, cần kiểm lại trên seed chưa từng dùng.

## Hệ thống (cố định)
Với mỗi deployment, tức mỗi (họ, backbone, seed, fold):
1. Có hai predictor:
   - **mặc định:** ensemble của run;
   - **gộp:** các thành viên mặc định cộng các thành viên RAMP-V13. Dùng `ext50.py` với `RAMP_FREEZE=1`, `RAMP_BIASCAL=1`, `RAMP_HYBRID=1`.
2. Với mỗi predictor, ELPIS-G (`rpt_auto_d`) chọn selector như bình thường (`full` hoặc `rpt_L` theo D).
3. **Cổng:** dùng predictor gộp nếu risk calibration của selector được chọn (`calrisk_rpt_L` nếu chọn `rpt_L`, ngược lại `calrisk_full`; đo trên unit calibration dưới các shift đã thấy) thấp hơn hẳn risk đó của predictor mặc định. Nếu không, dùng predictor mặc định; khi bằng nhau cũng dùng mặc định.

Quy tắc chỉ dùng unit calibration, không dùng dữ liệu test hay lỗi test.

## Dữ liệu
- **Seed mới:** 252–281, chưa từng dùng.
- **Họ:** vòng bi (6 điều kiện), C-MAPSS FD001–4, `battery_cv` (xoay unit). GRU và TCN.
- **Train:** ensemble mặc định được train như vòng 10; riêng vòng bi đã chạy trong EXT63.
- **Eval:**
  - `ext38.py`: hệ thống mặc định;
  - `ext50.py` với cấu hình trên: hệ thống gộp.

## Phân tích (đặt trước; đơn vị là seed, cách gộp như EXT59/EXT62)
- **Endpoint:** risk dưới 6 lỗi ở coverage 90% matched. So cổng với ELPIS-G trên predictor mặc định.
- **Chính, vòng bi:**
  - "**hòa**" nếu cận trên CI 90% bootstrap theo seed (20 000 lần) ≤ +0.3;
  - "**thắng**" nếu trung bình < 0 và Wilcoxon hai phía p < 0.05.
- **Phụ:**
  - C-MAPSS và pin với cùng tiêu chí; báo cáo cả 3 họ với Holm trên 3;
  - hiệu lỗi bi quan;
  - tỷ lệ deployment chọn predictor gộp;
  - cổng so với gộp-luôn-luôn;
  - RMSE.

## Báo cáo
- Mọi kết quả đều được báo cáo.
- Kết quả là thăm dò, vì ngân sách α đã hết và quy tắc được tìm ra post hoc. Không thay đổi phát biểu xác nhận nào của bài.

## Vận hành
1. Chờ EXT63 chạy xong; EXT63 đã gồm train vòng bi 252–281.
2. Train C-MAPSS và pin 252–281, tối đa 8 tiến trình, 2 lượt để chạy bù.
3. Eval mặc định với 6 shard, rồi eval gộp với 6 shard, chạy tuần tự.

## Kết quả, ngày 2026-11-18 (thăm dò, seed mới 252–281): vòng bi ĐẠT "hòa" nhưng chưa đạt "thắng"; C-MAPSS và pin THẮNG
- Train: đủ 240 run C-MAPSS và 660 run vòng bi + pin.
- Eval: đủ 900 run hệ thống mặc định và 900 run hệ thống gộp. Không có run lỗi; đủ 30/30 seed ở mọi họ.

Cổng so với ELPIS-G trên predictor mặc định (risk dưới lỗi @90%):

| Họ | Hiệu | CI 90% | Seed tốt hơn / kém hơn | p (p_Holm) | Hòa | Thắng | Lỗi bi quan | Tỷ lệ chọn gộp | RMSE |
|---|---|---|---|---|---|---|---|---|---|
| Vòng bi | −0.54 | [−0.97, −0.13] | 18 / 12 | 0.13 (0.13) | **có** | không (p ≥ 0.05) | −0.58 | 0.43 | 0.272 so với 0.279 |
| C-MAPSS | −0.58 | [−0.70, −0.46] | 29 / 1 | < 1e-4 | có | **có** | −0.43 | 0.74 | 14.81 so với 15.72 |
| Pin | −0.54 | [−0.62, −0.46] | 30 / 0 | < 1e-4 | có | **có** | −0.44 | 0.77 | 0.085 so với 0.091 |

- Trên vòng bi, cổng tốt hơn gộp-luôn-luôn 1.13 điểm. Trên C-MAPSS và pin, cổng gần bằng gộp-luôn-luôn (+0.01 và +0.06).
- **Diễn giải đúng như đặt trước:**
  - Vòng bi đạt "hòa", và CI 90% bootstrap còn nằm hoàn toàn dưới 0. Tuy vậy, tiêu chí "thắng" (Wilcoxon p < 0.05) không đạt, vì chỉ 18/30 seed tốt hơn.
  - Lỗi bi quan giảm ở cả 3 họ.
  - Đây là kết quả thăm dò: quy tắc được tìm ra post hoc rồi kiểm trên seed mới. Không đổi phát biểu xác nhận nào của bài.
