# Chạy tiếp Qwen3.5 SOP aircraft từ ZIP đã lưu

Áp dụng cho 112 test SOP aircraft, không phải 22 test LLM4IAS.

1. Trên Kaggle, tạo một Dataset riêng tư chứa **file ZIP backup 41 test**. Giữ nguyên ZIP, không chỉ upload bảng CSV. Gắn dataset đó vào notebook bằng Add Input. Không đưa kết quả vào model prompt.
2. Dùng notebook `experiments/sop_aircraft/kaggle_qwen35_9b_sop_aircraft.ipynb`, bật Internet và GPU T4 ×2.
3. Session mới: chạy cell 1–6. Cần tải/nạp lại model vì session cũ đã tắt. Giữ FP16, thinking=False, greedy, 384 token, 12 lượt; không dùng code cài Qwen2.
4. **Chưa chạy cell 7 hoặc 8.** Cell 5 tạo RUN_DIR mới; cần khôi phục kết quả vào run mới này.
5. Thêm một code cell sau cell 6, dán toàn bộ [kaggle_restore_qwen35_partial.py](kaggle_restore_qwen35_partial.py).
6. Đổi dòng BACKUP_ZIP ở cuối thành đường dẫn ZIP trong Input. Có thể tìm bằng cell `print([str(p) for p in Path('/kaggle/input').rglob('*.zip')])`. Nếu để sai đường dẫn, cell khôi phục cũng liệt kê các ZIP tìm thấy rồi dừng để bạn sửa.
7. Chạy cell khôi phục. Kỳ vọng `Restored: 41 cases` và `Remaining without saved results: 71`. Nếu ZIP có số file khác thì dùng số thực tế, không ép thành 41.
8. Bỏ qua smoke test, chạy cell 8 với RUN_COUNT=112. Các dòng đầu xuất hiện nhanh vì đọc file cũ; chỉ case chưa có mới suy luận. Nếu file đã ghi nhận max_turns_reached, giữ nguyên như kết quả lỗi, không chạy lại để chọn kết quả đẹp hơn.
9. Xong thì chạy cell 9 và 10 để tính điểm trên tổng 112 case và tải ZIP mới.

Cell khôi phục kiểm tra model revision, precision, prompt/template, dataset, giới hạn sinh/lượt, môi trường và hash prompt từng case trước khi copy. Không ghi đè metadata mới bằng metadata cũ; lưu liên kết backup trong resume_manifest.json và resume_parent_metadata.json. Không chạy mã hoặc extract tùy ý từ ZIP.

Nếu báo `Settings/environment differ`, gửi nguyên danh sách khác biệt để kiểm tra. Không bỏ assert hoặc trộn kết quả FP16 với NF4. Gián đoạn session không đồng nghĩa mọi test lỗi: các file hợp lệ đã lưu vẫn dùng được, không cần sinh lại.

Nếu chỉ refresh giao diện nhưng kernel và RUN_DIR cũ vẫn còn, không cần cell khôi phục; chạy lại cell 8 sẽ đọc file đang có. Quy trình trên dành cho trường hợp **session mới**.
