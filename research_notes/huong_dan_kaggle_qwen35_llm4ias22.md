# Qwen3.5-9B trên 22 test LLM4IAS

Upload [kaggle_qwen35_9b_llm4ias22_rubric_v1.ipynb](../experiments/llm4ias/kaggle_qwen35_9b_llm4ias22_rubric_v1.ipynb) lên Kaggle. Đây là **22 test gốc Inspection Station**, không phải notebook 112 test SOP aircraft.

## Chạy trên Kaggle

1. Tạo notebook/session mới, bật Internet và GPU T4 ×2. Không dùng chung kernel đang chạy Qwen2.
2. Cell 1 cài môi trường Qwen3.5. Giữ PyTorch/CUDA có sẵn, không chạy cell cài thư viện Qwen2. Nếu đã import HF/Transformers trước đó, restart session trước khi cài.
3. Cell 2 kiểm tra hai GPU, model class và cấu hình. Mặc định FP16, không lượng tử hóa.
4. Cell 3 tải Excel và prompt gốc theo commit, kiểm tra hash.
5. Cell 4 phải báo 22 cases: 11 SOP và 11 Unexpected Events.
6. Cell 5 tải model `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Dừng kiểm tra nếu thiếu/mismatch trọng số hoặc model bị offload CPU/disk.
7. Cell 6 khai báo suy luận và chấm; cell 7 thử một case. JSON hợp lệ chưa đồng nghĩa command đúng; không sửa prompt theo đáp án sau smoke test.
8. Cell 8 chạy đủ 22 case. Case thử đã lưu được dùng lại trong cùng run.
9. Cell 9 tạo bộ review 44 phản hồi. Cell 10 dành cho CSV đã chấm C/R; có thể để sau khi tải ZIP.
10. Cell 11 đóng ZIP, tải bằng link hoặc Output/Files trước khi kết thúc session.

Không cần API key. Dataset tự tải; không cần gắn thêm Kaggle dataset. Nếu có lỗi DNS `pypi.org`, kiểm tra Internet trước khi kết luận package không tồn tại. Cảnh báo kernel fallback có thể khiến inference chậm; không tự cài kernel khác giữa run.

## Điều kiện so sánh

Giữ prompt của 22 case, system message, rubric F/C/R, greedy, seed 42 và tối đa **256 token đầu ra mỗi case**, giống notebook Qwen2 22 test. Qwen3.5 dùng `enable_thinking=False` qua chat template chính thức. Thư viện và template khác theo model, được ghi metadata; không coi đây là hai môi trường hoàn toàn giống nhau hoặc setting tối ưu riêng từng model.

Model nạp kiến trúc checkpoint đầy đủ nhưng chỉ nhận text. Không dùng vòng agent 12 lượt, giới hạn 384 token hay JSON protocol patch của SOP aircraft. Không bóc khối think hoặc sửa JSON đầu ra để tăng điểm F.

Review pack gồm **22 output Qwen3.5 mới + 22 raw output Qwen2 tác giả lưu sẵn** ở sheet `Qwen2-7B-Instruct_1`. Paper không dùng Qwen3.5. Tên `author_saved_run` chỉ nguồn Qwen2 cũ; không có nghĩa tác giả đã chạy Qwen3.5.

- F và AST tự động.
- C_suggested chỉ hỗ trợ duyệt; C_final cần xác nhận nội dung.
- R cần chấm theo [rubric v1](rubric_2409_v1.md), không tự lấy điểm H cũ.
- Case 017 giữ C=U cho cả hai nguồn vì reference mâu thuẫn.
- `review_scores.csv` trống là đang chờ chấm, không phải accuracy 0%.

Đọc `review_blind.md`, điền `review_scores.csv` rồi dùng cell 10 tổng hợp. Khi CSV chấm ở ngoài Kaggle, upload file và sửa `REVIEW_CSV_PATH`. Cell 11 cần chạy lại sau khi cập nhật nhãn để ZIP có kết quả mới nhất. Chi tiết điền cột và U: [hướng dẫn rubric 22 test](huong_dan_kaggle_llm4ias22_rubric_v1.md).

Thư mục output riêng, không ghi đè Qwen2:

```text
/kaggle/working/llm4ias22_qwen35_9b_rubric_v1/runs/<run_id>/
```

Giữ cả ZIP Qwen2 và ZIP Qwen3.5 để đối chiếu từng case theo cùng rubric. Chạy thay model là baseline so sánh, chưa phải đóng góp phương pháp mới và không thay thế evaluation đầy đủ 100 test của paper.
