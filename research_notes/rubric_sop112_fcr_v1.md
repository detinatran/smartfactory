# Áp dụng F/C/R cho SOP aircraft 112 test — bản triển khai để duyệt

Ngày: 30/10/2026. Kế thừa định nghĩa F và thang R từ `rubric_2409_v1.md`, không sửa rubric 22 test. Đây là hướng dẫn vận hành cho nhiệm vụ nhiều bước, không phải rubric tác giả công bố và chưa phải một bản đã được người dùng duyệt riêng. Được xây sau khi đã thấy output, cần ghi nhận tính exploratory.

## Đơn vị chấm

Chấm **từng phản hồi model** trong lịch sử thực tế của mỗi task, gồm lượt gọi tool thất bại và lượt nộp kết quả. Không chỉ chấm lượt cuối hoặc chỉ các task thành công. Một phản hồi ở 22 test cũng là một đơn vị, nhưng không gộp các tỷ lệ giữa hai dataset.

Người chấm đọc input, system prompt/SOP/tool specification và các phản hồi/observation **trước lượt đang chấm**. Không dùng observation xuất hiện sau lượt đó để hợp thức hóa reason. Observation của tool ở lượt đang chấm được lưu riêng để kiểm toán kết quả thực thi, không phải bằng chứng model đã biết trước.

## Ba tiêu chí

| Tiêu chí | Quy tắc |
| --- | --- |
| F (0/1) | Toàn bộ raw response parse thành JSON object, có reason và command dạng chuỗi. Không sửa JSON, không bóc một đoạn JSON khỏi văn bản. Giữ y định nghĩa ở bộ 22. |
| C (0/1/U) | Đúng hành động được phép ở trạng thái hiện tại, đúng hàm và tham số/aircraft, thỏa điều kiện trước theo SOP và observation đã có. Không đòi khớp một chuỗi cố định nếu có nhiều hành động hợp lệ. |
| R (1–5/U) | Dùng nguyên mốc điểm của rubric 22; xét sự kiện và SOP, không xét độ dài văn bản và không tự lấy C làm điểm R. |

**C=1:** có bằng chứng đủ để xác nhận hành động hợp lệ. **C=0:** vi phạm xác định được, như gọi sai aircraft, tự bịa trạng thái làm tham số, bỏ điều kiện bắt buộc hoặc báo success khi đã biết bước đó thất bại. **C=U:** không xác định được ý định, hoặc SOP/tool specification/reference mâu thuẫn làm quyết định không thể phân xử. Ghi đoạn quy tắc và bằng chứng cho quyết định C.

Với F=0 mà đọc được ý định nguyên văn, có thể chấm C/R, ghi `evidence_mode=verbatim_intent`; F vẫn 0. Không sửa raw để thực thi. R vẫn có thể cao khi model hiểu lỗi nhưng chọn sai hành động; reason dài không tự nhận điểm cao.

Lượt finish_inspection được chấm như một hành động: các trạng thái báo cáo phải phản ánh evidence và nhánh SOP. Giá trị None có thể đúng khi không áp dụng; không mặc định thiếu observation là sai nếu bước thực sự không áp dụng. Không coi CSV final output là đáp án tuyệt đối về toàn bộ trình tự.

## Nguồn tham chiếu và xung đột

Bộ 22 có next-command reference cho từng test. Bộ 112 chỉ có final reference, không có đáp án next-command cho mọi lượt. Vì vậy C ở đây dựa trên SOP/tool specification và trace, phải ghi rõ khác biệt nguồn tham chiếu này. Đây là chung khung F/C/R, không phải cùng một bộ nhãn next-command đã có sẵn.

Không tự tạo chuỗi ground truth bằng cách chạy mock tool: tool CSV có thể nhận tham số không thỏa trình tự SOP, hoặc từ chối None dù SOP nói chỉ report khi áp dụng. Riêng reporting phải phân biệt: (a) báo cáo được yêu cầu theo nhánh nào; (b) chữ ký tool yêu cầu gì; (c) CSV ghi gì. Khi mâu thuẫn thực chất, dùng C=U và `reference_status=conflict`, không sửa policy để tăng điểm cho model nào.

Nếu có nhiều hành động hợp lệ, không đánh sai chỉ vì model không chọn một thứ tự duy nhất không được SOP bắt buộc. Các cờ thứ tự/dependency trước đây chỉ hỗ trợ tìm điểm cần đọc, không tự biến thành nhãn C.

## Tổng hợp

- F: số phản hồi hợp lệ / tổng phản hồi, kèm số đếm.
- C: số đúng k, sai w, U u và pending p. Tổng N=k+w+u+p. Báo coverage (k+w)/N và khoảng [k/N, (k+u+p)/N]. Khi chấm xong p=0, vẫn giữ U riêng.
- R: trung bình trên phản hồi có điểm 1–5, phân bố 1–5 và số U/pending. Không gán 0 cho missing.
- C/R chính báo cáo theo lượt, kèm số lượt mỗi model; hai model có độ dài trace khác nhau. Có thể bổ sung trung bình theo task khi quy tắc xử lý task không có điểm được định nghĩa trước, không thay mẫu số âm thầm.
- All là toàn bộ task/lượt trong dataset, không phải tổng F+C+R. Time là chỉ số chi phí riêng. Final-field match và thứ tự tool là phụ trợ, không thay C.

## Quy trình

Dùng review pack che nguồn A/B, chấm theo lịch sử của từng model (hai lịch sử có thể khác). Không ghép turn thứ i của hai model thành cùng một trạng thái. Che nguồn không bảo đảm không đoán được phong cách. Single-rater hoặc AI-assisted phải ghi đúng vai trò, không gọi nhãn AI là nhãn con người. Nếu nhờ AI hỗ trợ thì giữ nhãn đề xuất riêng và ghi người xác nhận; không tự tạo điểm R cuối bằng regex/AST.
