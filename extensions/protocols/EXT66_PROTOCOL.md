# EXT66: ELPIS-J pretrain theo LeWorldModel (end-to-end + SIGReg) thay cho EMA teacher (THĂM DÒ / PHÁT TRIỂN, không tiêu α). Chốt ngày 2026-11-19, TRƯỚC khi sửa code và chạy

**Yêu cầu của người dùng:** kiểm tra JEPA đang dùng có đúng kiểu "dự đoán và học biểu diễn trong latent, không decoder" không, và thử theo LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv 2603.19312) với Embedding Regularization loss SIGReg (LeJEPA, Balestriero & LeCun, arXiv 2511.08544).

**Hiện trạng (V13, predictor ELPIS-J của bài):**
- Stage I đã dự đoán trong latent, không có decoder: từ cửa sổ bị làm hỏng, predictor `P` dự đoán latent của cửa sổ sạch ở 3 vị trí sau.
- Chống sụp biểu diễn chỉ bằng EMA teacher (m = 0.99) + stop-gradient + predictor bất đối xứng (kiểu BYOL / I-JEPA), loss cosine. Không có regularizer nào trên phân phối embedding và chưa từng đo mức sụp.
- V13 đóng băng encoder ở Stage II (`ELPISJ_FREEZE=1`), nên nếu biểu diễn sụp thì head không sửa được.

## Biến thể mới V20 (`ELPISJ_SSL=sigreg`), mọi thứ khác giống V13 (`ELPISJ_FREEZE=1`, `ELPISJ_BIASCAL=1`)
Stage I theo LeWM / LeJEPA:
- **Bỏ EMA teacher và stop-gradient.** Target là latent của cửa sổ sạch 3 vị trí sau do **chính encoder online** tạo ra, gradient đi qua cả hai nhánh (end-to-end).
- **Loss dự đoán:** MSE giữa latent dự đoán `r` và latent target `t` (không chuẩn hoá về cosine).
- **SIGReg** trên embedding của encoder (ghép embedding context và target trong batch, kích thước 2B × d): chiếu lên 256 hướng ngẫu nhiên chuẩn hoá (lấy lại mỗi bước), thống kê Epps–Pulley so với N(0, 1), lưới t ∈ [0, 3] với 17 điểm, trọng số hình thang × φ(t) = exp(−t²/2), nhân với số mẫu, lấy trung bình qua các hướng. Giống cài đặt tham chiếu `lejepa`.
- **Tổng loss:** (1 − λ)·MSE + λ·SIGReg, **λ = 0.05** (giá trị khởi đầu khuyến nghị của LeJEPA). Chỉ một giá trị, không dò λ.
- Batch 128, 20 epoch, AdamW lr 1e-3 như V13.

## Chẩn đoán sụp biểu diễn (mới, cho cả V13 và V20)
Sau khi train mỗi thành viên, tính trên cửa sổ validation sạch (không dùng test):
- effective rank của embedding encoder (exp entropy của các giá trị kỳ dị đã chuẩn hoá, Roy & Vetterli 2007);
- trung bình độ lệch chuẩn từng chiều;
- thống kê SIGReg (cùng công thức trên, 256 hướng cố định).
Lấy trung bình 3 thành viên, ghi vào mỗi dòng CSV (`emb_effrank`, `emb_std`, `emb_sigreg`).

## Dữ liệu và chạy
- Seed phát triển 162–171 (đã dùng ở EXT50–EXT58), run `conf162_cmapss` + `conf162_ext`: C-MAPSS FD001–4, `battery_cv`, vòng bi (6 fold), GRU và TCN; 300 run.
- **V13r:** chạy lại V13 bằng đúng script mới (có chẩn đoán) để so cùng phiên bản code. Kiểm tra chéo: risk của V13r phải gần V13 của EXT56 (`ext56_v13`, `ext56_v13_other`).
- **V20:** cùng run, cùng seed.
- Tối đa 6 tiến trình GPU một lúc, V20 chạy trước rồi đến V13r, tuần tự. Chạy bù 1 lượt cho run lỗi.

## Phân tích (đặt trước; thăm dò)
- **Chính:** risk của ELPIS-G (`rpt_auto_d`) dưới 6 lỗi ở coverage 90% matched, V20 − V13r, theo từng họ (C-MAPSS, pin, vòng bi); đơn vị là seed (trung bình GRU/TCN, fold/subset có trọng số như các vòng trước); Wilcoxon theo seed, 10 seed.
- **Phụ:** risk OSC (`full`); RMSE sạch; tỷ lệ lạc quan khi nhận hết dưới lỗi; lỗi bi quan; ELPIS-G − OSC bên trong V20; effective rank, độ lệch chuẩn và SIGReg của embedding (V20 so với V13r).
- **Tiêu chí để đi tiếp:** V20 được coi là "hứa hẹn" nếu trung bình hiệu risk ELPIS-G < 0 trên ít nhất 2 họ và không họ nào > +0.3. Khi đó mới kiểm lại trên seed mới (từ 342) bằng một protocol riêng. Nếu không đạt, V13 giữ nguyên là ELPIS-J của bài.

## Báo cáo
Mọi kết quả đều được báo cáo, kể cả khi V20 kém hơn. Kết quả là thăm dò trên seed phát triển, không thay đổi phát biểu xác nhận nào của bài (vòng 10 vẫn là V13). ELPIS và ELPIS-G không phụ thuộc vào kết quả này.

## Kết quả, ngày 2026-11-19 (thăm dò): KHÔNG đạt "hứa hẹn"
- Đủ 300/300 run mỗi biến thể, không run lỗi. Kiểm tra chéo V13r − V13 (EXT56): −0.01, −0.01, +0.08.
- **Chính, risk ELPIS-G V20 − V13r:** C-MAPSS −0.90 (10/10, p = 0.002); pin +2.49 (0/10, p = 0.002); vòng bi +0.86 (3/10, p = 0.23). Chỉ giảm trên 1 họ và vượt +0.3 trên 2 họ, nên không đạt tiêu chí; V13 giữ nguyên.
- **Sụp biểu diễn:** effective rank V20 / V13r = 31.2 / 11.2 (C-MAPSS), 20.4 / 8.9 (pin), 20.4 / 15.6 (vòng bi). EMA teacher làm sụp một phần; SIGReg chữa được nhưng không làm risk thấp hơn một cách nhất quán (predictor V20 lạc quan hơn dưới lỗi trên pin +4.17 và vòng bi +1.03).
- Bên trong V20, ELPIS-G − OSC: −0.79, −0.60, −0.56 (10/10, 10/10, 8/10). Chi tiết: `extensions/results/ext66/FINDINGS.md`.
