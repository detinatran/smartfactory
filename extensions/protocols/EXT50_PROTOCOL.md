# EXT50: predictor theo cấu trúc JEPA (dựa trên một thiết kế JEPA hai stage nội bộ, chưa công bố) thay cho predictor hai giai đoạn hiện tại (THĂM DÒ). Chốt ngày 2026-11-16, TRƯỚC khi chạy

**Yêu cầu của người dùng:** thử đổi sang cấu trúc giống JEPA theo một thiết kế nội bộ của nhóm, rồi xem kết quả.

## Thiết kế (`scripts/ext50.py`)
Mỗi thành viên của ensemble (3 thành viên, backbone GRU/TCN giống vòng 8) được train qua hai stage.

**Stage I, tự giám sát, 20 epoch:**
- Encoder online nhận window bị hỏng. Một nửa số window nhận nhiễu RPT (A ∈ [1, 8], tối đa n/2 kênh). Nửa còn lại bị che một tập kênh trong một khối thời gian liền: giá trị đặt về 0, cờ thiếu đặt về 1.
- Một predictor MLP dự đoán latent của window sạch do target encoder tạo ra. Target encoder cập nhật bằng EMA, m = 0.99.
- Loss là năng lượng cosine 2 − 2cos, giống thiết kế JEPA nội bộ.

**Stage II, có giám sát:** fine-tune encoder và head RUL trên window sạch. Loss MSE có trọng số theo unit, AdamW, tối đa 40 epoch, early stopping với patience 8 trên tập validation.

**Đặc trưng thêm:** độ lệch JEPA = 1 − cos(pred_I(enc_I(x)), tgt_I(x)), lấy trung bình qua các thành viên, thêm vào φ. Các selector dùng đặc trưng này có hậu tố `_jd`.

**Selector:** mọi thứ khác giữ nguyên như EXT38 (D, `rpt_auto_d`, các baseline). Mọi selector đều dựa trên predictor JEPA.

## Dữ liệu
- Splits của các run vòng 8, seed 162–171: C-MAPSS FD001–4, `battery_cv`, vòng bi.
- Train lại predictor theo JEPA trong script.
- N-CMAPSS bị loại ra vì stride bằng 1 nên tập fit rất lớn. Lý do này ghi trước khi chạy.

## Phân tích (đặt trước, thăm dò; so với predictor hai giai đoạn mặc định trên cùng seed và cùng split, lấy từ EXT40)
1. **Predictor:**
   - RMSE trên dữ liệu sạch;
   - tỷ lệ lạc quan khi chấp nhận toàn bộ dưới 6 lỗi, để xem JEPA có làm predictor bền hơn với lỗi không.
2. **Hệ thống:** risk dưới 6 lỗi ở coverage 90% (matched) của RPT-D khi dùng predictor JEPA, so với RPT-D khi dùng predictor mặc định. Hai hệ thống có predictor khác nhau nên so theo cặp seed, Wilcoxon theo seed.
3. **Bên trong JEPA:**
   - `rpt_auto_d` − `full`: lợi ích của RPT có còn khi predictor đã được pretrain chống lỗi không;
   - `rpt_auto_d_jd` − `rpt_auto_d`: đặc trưng độ lệch JEPA có giúp không.

Mọi kết quả đều được báo cáo.

## SỬA THIẾT KẾ TRƯỚC KHI CHẠY (ngày 2026-11-16, sau khi đọc tài liệu thiết kế JEPA nội bộ; chưa có kết quả nào ngoài một lần chạy thử kỹ thuật trên 1 run)
Người dùng cung cấp tài liệu thiết kế nội bộ (chưa công bố). Thiết kế được sửa cho sát cấu trúc trong tài liệu đó như sau.

**Stage I: masked JEPA căn theo horizon.**
- Context: window hiện tại bị hỏng, theo cùng cơ chế làm hỏng như trên.
- Target: latent EMA của window *sạch ở 3 vị trí sau* trên cùng unit; nếu không đủ 3 vị trí thì lấy window cuối của unit.
- Predictor P([z ‖ E_s(s) ‖ E_q(q)]), trong đó:
  - s: đặc trưng symbolic của từng kênh (độ dốc, cuối − đầu, độ lệch chuẩn);
  - q: chất lượng quan sát (tỷ lệ thiếu của từng kênh).
- Loss cosine; target encoder cập nhật bằng EMA với m = 0.99.

**Stage II:** chuyển mọi module sang bài toán có giám sát.
- v = F(z) + σ(γ)·r⁺, với r⁺ = P(...).
- Đầu ra = cap·σ(H(v) + α·C(s)), với α = σ(G([z ‖ E_s(s) ‖ E_q(q)])).
- Train toàn bộ trên window sạch, loss MSE có trọng số theo unit, early stopping.
- Bỏ target encoder.

**Bỏ đặc trưng độ lệch `_jd`:** với target là tương lai, độ lệch không còn đo được lúc suy luận. Phân tích 3b vì vậy bị bỏ; các phân tích khác giữ nguyên.

- **Vận hành, ngày 2026-11-16:** EXT50 bị tạm dừng sau 69 run vì GPU quá tải khi chạy song song với vòng 9. Một CSV có thể chưa ghi xong đã được chuyển vào `runs/ext50_quarantine`. EXT50 sẽ tự chạy lại sau vòng 9; các CSV đã xong được bỏ qua.

## Kết quả, ngày 2026-11-17 (thăm dò)
ELPIS-J tốt hơn trên C-MAPSS (−0.82) và pin (−0.49), nhưng kém hơn nhiều trên vòng bi (+3.30). Trên predictor ELPIS-J, RPT-D vẫn tốt hơn OSC ở cả 3 họ. Chi tiết xem FINDINGS.md.
