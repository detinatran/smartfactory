# So sánh Qwen2-7B và Qwen3.5-9B trên 22 test LLM4IAS và 112 test SOP aircraft

Ngày kiểm tra: 30/10/2026. Báo cáo dựa trên raw output và trace đã lưu; không gọi lại model và không thay nhãn tham chiếu.

Bảng ngắn theo số đạt/tổng số và tỷ lệ: [bang_rubric_ngan_gon.md](bang_rubric_ngan_gon.md). All là toàn bộ test; C/R chưa duyệt, không thay bằng AST hoặc final-field match.

Hai bộ được báo cáo riêng: LLM4IAS yêu cầu sinh lệnh tiếp theo từ event log, còn SOP aircraft yêu cầu agent thực hiện nhiều lượt mock tool rồi nộp kết quả. Không cộng hai bộ thành 134 test để tính một accuracy chung.

## 1. LLM4IAS — 22 test gốc

| Chỉ số | Qwen2-7B chạy lại | Qwen3.5-9B | Qwen2 tác giả lưu sẵn |
| --- | --- | --- | --- |
| Số test | 22 | 22 | 22 |
| F: JSON hợp lệ | 22/22 (100%) | 22/22 (100%) | 19/22 (86,36%) |
| AST match toàn bộ | 4/22 (18,18%) | 11/22 (50%) | 5/22 (22,73%) |
| AST match SOP | 4/11 (36,36%) | 11/11 (100%) | 5/11 (45,45%) |
| AST match Unexpected Events | 0/11 | 0/11 | 0/11 |
| C ngữ nghĩa cuối cùng | Chưa duyệt | Chưa duyệt | Chưa duyệt |
| R: lý do 1–5 | Chưa chấm | Chưa chấm | Chưa chấm |
| Tổng latency suy luận | 85,72 giây | 189,33 giây | Không có log tương đương |
| Output token | 587 | 1.242 | Không có log tương đương |

Qwen3.5 tăng 7 ca AST, tương đương 31,82 điểm phần trăm, tất cả ở SOP. Không có ca AST Qwen2 đúng mà Qwen3.5 mất. Tuy nhiên các alert khác cách diễn đạt vẫn có thể đúng nội dung, nên 0/11 AST Unexpected không phải bằng chứng mọi ca đều sai ngữ nghĩa. C/R chưa có nhãn cuối cho các nguồn; không dùng số 0/pending trong CSV làm điểm 0%.

Đọc raw thấy case 014–015 Qwen3.5 retry release holder thay alert, còn Qwen2 chọn alert; case 012 và 019 Qwen3.5 chuyển sang báo lỗi phù hợp hơn để xem xét. Case 017 vẫn C=U vì reference mâu thuẫn. Không thay rubric để ưu tiên model mới. Output tác giả là Qwen2 pretrained trong workbook, không phải checkpoint SFT và không phải kết quả Qwen3.5 do tác giả chạy.

22 prompt trước wrapping đã xác nhận giống nhau giữa hai run, rubric hash giống nhau. FP16, greedy, seed 42, tối đa 256 output token/case; Qwen3.5 thinking=False. Tokenizer/template và Transformers khác theo model. Đây là 22 Inspection Station cases, không phải đầy đủ 100 test trong Table II của paper.

## 2. SOP aircraft — 112 test

| Chỉ số (112 test) | Qwen2-7B | Qwen3.5-9B |
| --- | ---: | ---: |
| Khớp đủ 7 trường kết quả cuối | 68/112 (60,71%) | **80/112 (71,43%)** |
| Đúng thứ tự 3 bước kiểm tra đầu | 50/112 (44,64%) | **102/112 (91,07%)** |
| Phản hồi đúng định dạng JSON (theo lượt) | **100%** | 90,35% |
| Tổng thời gian suy luận | **2 giờ 19 phút** | 4 giờ 15 phút |

**Nhận xét:** Qwen3.5 khớp kết quả cuối và thứ tự kiểm tra tốt hơn, nhưng JSON kém ổn định và chạy lâu hơn khoảng 1,83 lần.

*Khớp kết quả cuối chưa chứng minh tuân thủ toàn bộ SOP; C/R thủ công chưa hoàn tất. [Bảng chi tiết](bang_112_chi_tiet.md) giữ các chỉ số bổ sung.*

Qwen3.5 cải thiện final-field match 12 case, tức 10,71 điểm phần trăm. Thứ tự ba bước đầu tăng 52 case, tức 46,43 điểm phần trăm. Đổi lại, JSON kém ổn định hơn, output dài hơn, chạm token cap nhiều hơn và tổng latency cao hơn khoảng 1,83 lần trên các run này. Không quy toàn bộ chênh lệch thời gian cho kích thước model: kiến trúc, token đầu ra, thư viện, kernel fallback và điều kiện session đều có thể ảnh hưởng. Tổng latency là cộng các lần generate được lưu, không gồm tải trọng số, cài thư viện, thời gian ngắt kết nối hoặc toàn bộ wall-clock của notebook.

Đối chiếu cùng từng case theo final-field match:

| Kết quả cặp | Số case |
| --- | --- |
| Cả Qwen2 và Qwen3.5 khớp | 46 |
| Chỉ Qwen3.5 khớp | 34 |
| Chỉ Qwen2 khớp | 22 |
| Cả hai chưa khớp | 10 |

Vì vậy, Qwen3.5 không cải thiện đồng đều: 34 ca tăng nhưng 22 ca mất, ròng +12. Chưa thực hiện kiểm định ý nghĩa thống kê; không gắn nhãn statistical significance từ chênh lệch tỷ lệ này.

## 3. Những giới hạn quan trọng khi đánh giá SOP

`finished=True` chỉ nghĩa đã gọi adapter nộp kết quả, không có nghĩa đúng hoặc an toàn. Khớp 7 trường chỉ đánh giá final report theo CSV, không chứng minh toàn bộ SOP được thực hiện đúng. Thứ tự ba tool đầu cũng chỉ là chỉ số chẩn đoán một phần.

Trong các ca khớp đủ trường, cả hai model đều có 21 case mang ít nhất một cờ: giá trị cuối khác null chưa có observation tương ứng, hoặc bằng chứng đầu vào reporting tool thiếu/khác observation trước đó. Với Qwen2 là 21/68, Qwen3.5 là 21/80. Không tự trừ chúng để tạo một accuracy “an toàn” mới: các cờ cần duyệt và chỉ bao phủ một phần dependency; một số observation có thể không áp dụng theo nhánh SOP.

Cờ dependency chỉ được kiểm tra trên các tool call có dữ liệu tên/argument đã lưu; một số call thất bại trước khi gán metadata tool sẽ không được bao phủ. Không coi “không có cờ” là “đã chứng minh tuân thủ SOP”. Bộ tools là mock dựa CSV, không phải simulator vật lý. Đã có điểm chưa thống nhất giữa nhánh điều kiện trong SOP, trường bắt buộc của mock tool và reference; đây là giới hạn benchmark, không quy mọi mismatch cho model.

## 4. Nhận xét cho hướng nghiên cứu

Các run cho thấy nâng model có thể cải thiện lệnh SOP và báo cáo cuối, nhưng chưa loại bỏ lỗi format, retry sai policy, khai báo kết quả chưa có bằng chứng và xử lý sự cố chưa thống nhất. Điều này tạo động lực cho một lớp verification dựa trên policy/dependency/evidence. Hiện đây là phân tích baseline, chưa chứng minh phương pháp CD-SV hay thuật toán đề xuất tạo ra cải thiện.

Bước tiếp theo là chốt nhãn C/R nhất quán trên 22 test và chọn các nhóm lỗi có bằng chứng trong 112 trace để đánh giá verifier. Khi thử phương pháp mới, dùng cùng model/task/budget và báo cả lỗi giảm được lẫn chi phí tăng thêm. Chưa cần chạy thêm model chỉ để tăng số bảng.

## 5. Kiểm tra artifact và khả năng đối chiếu

- ZIP Qwen3.5 mới đủ 112 case ID duy nhất, không hỏng CRC hoặc trùng tên member. Đã tính lại điểm từ trace, khớp summary lưu trong ZIP.
- Dataset hash, revision, precision FP16, seed, JSON protocol v2, giới hạn 384 token/12 lượt và 112 input/prompt ban đầu giống giữa Qwen2/Qwen3.5. System prompt SHA-256: 50a6dce6cddaa49f4c43434ebe48697bb33a6c67b47d79bcf3421c0379fc445e.
- Model revision Qwen2: f2826a00ceef68f0f2b946d945ecc0477ce4450c. Qwen3.5: c202236235762e1c871ad0ccb60c8ee5ba337b9a, thinking=False.
- Môi trường model khác: Transformers 4.51.3 vs 5.21.1; Accelerate 1.6.0 vs 1.15.0. Không có dữ liệu kết quả SOP aircraft của tác giả LLM4IAS để tạo cột tác giả ở bảng 112.
- ZIP mới SHA-256: 6d241f202f720c2906154aaffe43437123a2664260d76ad6eddfab93f63c1d34.

File máy đọc: `paired_sop112.csv`, `comparison.json`. Bản ZIP gốc và audit mới: `../../sop_aircraft/reviews/112_qwen35_9b_9bd6d1/`. Báo cáo 22 chi tiết: `../../llm4ias/reviews/20261029T160312Z_3dbdf4/nhan_xet_va_so_sanh_22test.md`.
