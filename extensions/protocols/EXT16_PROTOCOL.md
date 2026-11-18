# EXT16: đẩy hướng A* theo review lần 2. Chốt ngày 2026-11-13, TRƯỚC khi chạy

Người dùng chọn tiếp tục nhắm A*. Review lần 2 (weak reject 4/10) yêu cầu:
1. cho các head baseline dùng cùng dữ liệu nhiễu;
2. dùng ensemble lớn hơn (M ≥ 5);
3. gỡ lỗi `err_reg` trên C-MAPSS, nơi risk gần như phẳng khoảng 11% theo coverage;
4. bổ sung lỗi thực tế hơn (làm riêng);
5. đặt bài trong tương quan với ATC, dự đoán accuracy trên meta-dataset và OpenMix (phần viết).

Dữ liệu: seed 72–81 (đã dùng cho xác nhận vòng 2), GRU/TCN. **Mọi kết quả là thăm dò; phương pháp cuối không đổi.**

## A. Head baseline có dữ liệu nhiễu (`scripts/ext16_heads.py`)
- `deep_ens_nll_aug`, `evidential_aug`, `cqr_lower_aug`, `selectivenet_aug`: cấu hình y như EXT11/EXT14.
- Dữ liệu train = cửa sổ fit sạch + 1 bản nhiễu cấu hình L (A ∈ [1, 8], k ≤ 0.5n, seed + 52000), nhãn RUL thật. Đây là cách
  duy nhất để "cho dữ liệu nhiễu" vào các mô hình dự đoán y.
- **Kiểm định:** `rpt_L` so với từng bản `_aug`, Holm trên 16. Báo cáo kèm bản gốc từ EXT14.

## B. Ensemble 5 thành viên (`scripts/ext16_m5.py` + `ext12_main_v2.py`)
- Với mỗi run, train thêm thành viên 3 và 4: cùng cấu hình, cùng dữ liệu, seed thành viên = seed run + 100·m.
- Run mirror có `ensemble_size = 5`, chạy `ext12_main_v2.py` trên đó.
- **Kiểm định:** `rpt_L` so với `ensemble_std`, `full`, `gauss_bias_aware` và `err_reg`, Holm trên 16.

## C. Biến thể hồi quy sai số (`scripts/ext16_errreg.py`)
- `err_reg_l1`: HGB với loss tuyệt đối (trung vị, bền với đuôi).
- `err_reg_clip`: mục tiêu E được kẹp trong [−3δ, 3δ].
- `p_opt_reg`: HGB phân loại E > δ. Đây chính là OSC, dùng làm đối chứng; không đưa vào bảng.
- Mỗi biến thể có thêm bản dùng dữ liệu nhiễu L (`_rptL`).
- **Quy tắc bảo thủ (chốt trước):** trong bảng chính, `err_reg` được thay bằng biến thể có risk thấp nhất ở **từng dataset**
  (chọn trên seed 72–81, tức là có lợi cho baseline).
- **Kiểm định:** `rpt_L` so với biến thể mạnh nhất, Holm trên 4. Kiểm tra nhất quán: `full`, `err_reg` và `rpt_L` phải khớp
  EXT12.
