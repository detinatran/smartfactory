# Chạy lại 22 test LLM4IAS bằng Qwen2 và chấm theo rubric v1

Notebook: [kaggle_qwen2_llm4ias22_rubric_v1.ipynb](../experiments/llm4ias/kaggle_qwen2_llm4ias22_rubric_v1.ipynb).

Đây là 22 test Inspection Station của bài 2409, gồm 11 SOP và 11 Unexpected Events; **không phải 112 test SOP aircraft**. Runner do mình xây dựng từ artifact công khai, không phải code gốc đầy đủ của tác giả. Kết quả chỉ đại diện cho tập con này, không thay thế kết quả 100 test trong paper.

## 1. Mở notebook và cấu hình Kaggle

1. Import notebook trên vào Kaggle, tạo session riêng cho Qwen2.
2. Bật Internet, chọn GPU T4 ×2.
3. Chạy lần lượt cell 1–4: cài thư viện, kiểm tra GPU, tải artifact và chuẩn bị 22 prompt. Không cài requirements Qwen3.5 vào session này.
4. Cell 4 phải báo 22 cases, 11 SOP và 11 Unexpected Events. Dữ liệu tự tải từ repo công khai, không cần upload dataset hoặc cung cấp API key.

## 2. Nạp model và chạy 22 test

1. Chạy cell 5 để nạp `Qwen/Qwen2-7B-Instruct`; checkpoint được ghim về revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c` như run trước.
2. Chạy cell 6 để khai báo hàm suy luận và chấm.
3. Chạy cell 7 thử một case, rồi cell 8 chạy đủ 22 case.
4. Chạy cell 9 để tạo bộ review và thống kê tự động.

Cấu hình giữ FP16, không lượng tử hóa mặc định, greedy, seed 42, tối đa 256 token đầu ra/case. Prompt dựng từ `prompt_example.txt` và event log; đáp án tham chiếu, rubric và output tác giả không đưa vào model input. Không áp dụng patch JSON protocol v2 của SOP aircraft vào notebook này.

Giữ prompt gốc giúp so sánh với run trước; những mâu thuẫn giữa hướng dẫn công khai và reference vẫn là hạn chế cần báo cáo. Đây là chạy checkpoint pretrained Instruct công khai, không phải checkpoint fine-tuned riêng của tác giả.

## 3. Đọc và chấm kết quả

Cell 9 tạo **44 phản hồi A/B**: 22 phản hồi mới và 22 raw output tác giả lưu tại sheet `Qwen2-7B-Instruct_1`. Cả hai nguồn dùng cùng rubric [rubric_2409_v1.md](rubric_2409_v1.md), do mình quyết định và đã thống nhất, chưa được tác giả xác nhận.

| File | Công dụng |
| --- | --- |
| `review_blind.md` | Event log, reference và hai phản hồi A/B để đọc chấm |
| `review_scores.csv` | Nhập C/R cuối cùng và ghi chú |
| `auto_suggestions.csv` | Điểm F, AST và gợi ý C; chưa phải nhãn cuối |
| `source_key.csv` | Giải mã nguồn A/B; để sau khi chấm nếu muốn giảm thiên lệch |
| `rubric_summary.json` | Tổng hợp từng nguồn, SOP và Unexpected Events; thể hiện pending/U |

- **F** tự động kiểm tra toàn bộ raw output có đúng JSON với `reason` và `command` dạng chuỗi. Không sửa JSON trước khi chấm.
- **C_final_0_1_U**: 1 đúng nội dung command, 0 sai, U không thể kết luận. Gợi ý tự động chỉ hỗ trợ; thông báo alert khác câu chữ phải đọc ngữ cảnh để duyệt.
- **R_final_1_to_5_U**: chấm lý do theo rubric, độc lập với command. Không tự gán 5 vì command đúng. U dùng khi không thể đánh giá; ô trống là chưa chấm.
- Case `inspection_station_017` có reference mâu thuẫn: giữ C=U cho cả hai nguồn theo rubric đã chốt.
- Điền `reviewer` và `review_notes`. Nếu C khác gợi ý tự động, bắt buộc ghi lý do. Khi raw JSON lỗi nhưng nội dung vẫn đọc được để chấm C/R, điền `evidence_mode=verbatim_intent`; F vẫn bằng 0.

Mở CSV bằng Excel hoặc trình soạn thảo rồi lưu UTF-8, giữ nguyên `item_id`. Chấm thủ công có thể thực hiện sau khi tải ZIP về. Không bắt buộc giữ GPU chạy trong thời gian chấm. Một người chấm không tạo ra chỉ số đồng thuận giữa nhiều người chấm.

Cell 10 tổng hợp CSV đã điền. Mặc định đọc `RUN_DIR / "review_scores.csv"`; nếu upload bản đã chấm lên Kaggle, đổi `REVIEW_CSV_PATH` thành đường dẫn file đó. Khi chưa điền nhãn, có thể bỏ qua cell 10 và tải ZIP trước. Không báo tỷ lệ C cuối cùng từ `C_suggested`, và không coi ô trống/U là sai. Báo số đúng/sai/U, độ bao phủ, khoảng tỷ lệ C và số lý do thực sự đã chấm.

## 4. Tải kết quả

Chạy cell 11, bấm link ZIP hoặc tìm trong Output/Files. Thư mục mỗi run có ID riêng dưới:

```text
/kaggle/working/llm4ias22_qwen2_rubric_v1/runs/
```

ZIP chứa cấu hình, prompt, raw output mới, raw output tác giả trong bộ review, rubric snapshot và bảng chấm; không chứa trọng số model. Tải ZIP trước khi kết thúc session. Nếu đã chỉnh CSV sau khi đóng gói, chạy lại cell 10 và 11 để cập nhật ZIP.

Đổi rubric chỉ thay cách chấm: nếu còn raw output của run 22 test cũ thì có thể chấm lại mà không cần sinh mới. Notebook này phục vụ yêu cầu chạy lại, đồng thời giữ cấu hình và bằng chứng để đối chiếu.
