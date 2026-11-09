# Rubric 2409 v1 — đã được người dùng chấp nhận, chưa phải rubric tác giả xác nhận

Ngày: 29/10/2026. Paper duy nhất làm gốc: **arXiv:2409.18009v2**.

Trạng thái: người dùng chấp nhận v1 ngày 29/10/2026. Giữ nguyên các quy tắc F/C/R, U và cách báo cáo; mọi chỉnh sửa nội dung sau đây cần phiên bản mới. Sự chấp nhận này không có nghĩa tác giả đã xác nhận rubric.

## 1. Phạm vi bài toán

Đầu vào gồm mô tả agent/thiết bị, hàm điều khiển, SOP, nhiệm vụ và event log. Đầu ra là lý do cùng lệnh điều khiển tiếp theo. Tách SOP và Unexpected Events. Đây là điều khiển công nghiệp theo sự kiện, không phải dự đoán tuổi thọ thiết bị, QA chung hoặc lập lịch job-shop.

Mục V.B của [paper](https://arxiv.org/pdf/2409.18009v2) nêu hai chỉ số: Correctness Rate so command với reference và Reason Plausibility do con người chấm 1–5. Paper còn nói command có thể sai dù reason hợp lý. Giữ hai trục riêng, không ép điểm reason phải bằng độ đúng command.

Những gì dưới đây về tương đương ngữ nghĩa, mốc điểm và cách xử lý JSON là **quy tắc mình đề xuất để vận hành phép chấm**, không gán cho tác giả. Bản này được soạn sau khi đã xem kết quả 22 test, nên có tính exploratory; không tuyên bố đã preregister. Đóng băng phiên bản trước các lần chạy mới và áp dụng đối xứng cho hai nguồn output.

## 2. Ba cột chấm độc lập

### F — định dạng (0/1, tự động)

F=1 khi toàn bộ response sau bỏ khoảng trắng đầu/cuối parse được thành JSON object, có reason và command kiểu string. Không bóc JSON từ văn xuôi, không sửa dấu nháy, không thực thi command. Kiểm tra cấu trúc lệnh/hàm hợp lệ riêng nếu cần; JSON hợp lệ chưa chứng minh điều khiển đúng.

F=0 nếu sai điều kiện trên. Không sửa output gốc. Dấu code fence ngoài JSON vẫn là lỗi theo yêu cầu Only output a JSON structure của prompt; nếu thử parser linh hoạt phải báo cáo thành biến thể.

### C — đúng câu lệnh theo reference (0/1/U, duyệt nội dung)

- C=1: đúng hàm/hành động kế tiếp, đúng thiết bị và mọi tham số vận hành cần thiết theo reference cùng event cuối; không thêm hành động ngoài yêu cầu.
- C=0: sai hàm, thiết bị, hướng, thời lượng, thứ tự; bỏ sót hành động cần; thêm hành động không được phép; hoặc xử lý sự cố khác policy tham chiếu.
- C=U: reference mâu thuẫn với event/SOP, hoặc không xác định được command/ý định từ response. Phải ghi lý do, không tự biến U thành đúng hoặc bỏ khỏi bảng.

Quy tắc tương đương:

1. Bỏ khác biệt khoảng trắng, dấu nháy và cách biểu diễn số khi giá trị/nghĩa tương đương; không đổi hướng, thời lượng hoặc thiết bị.
2. Với alert_to_supervisor(message), cho phép diễn đạt lại **cùng sự cố trên cùng thành phần**. Thông báo chung chung hoặc sai thiết bị không tự được chấp nhận.
3. Nếu lỗi ngữ cảnh chỉ có một, tên thành phần có thể được lược nếu message vẫn định danh rõ sự cố qua ngữ cảnh; ghi chú quyết định để áp dụng cùng cách cho mọi output. Không dùng từ khóa alert đơn thuần làm đủ điều kiện.
4. Không chấp nhận emergency_stop thay alert hoặc retry thay alert chỉ vì có vẻ hợp lý. Đó là hành động khác; nếu muốn policy rộng hơn phải định nghĩa trước như một phân tích riêng.
5. Chuỗi nhiều lệnh chỉ đúng nếu toàn bộ chuỗi được specification cho phép; chứa lệnh đúng kèm lệnh sai không được tính đúng.
6. Với F=0 nhưng người chấm vẫn đọc được command nguyên văn không mơ hồ, có thể chấm C về **ý định** và đánh dấu evidence_mode=verbatim_intent. Không chỉnh sửa hoặc gọi đó là output thực thi được. Nếu cần đo khả năng tiêu thụ tự động, dùng E ở dưới.

Chuẩn tham chiếu giữ nguyên. Nếu phát hiện lỗi nhãn, đặt reference_status=conflict, C=U cho cả hai output cùng test. Đề xuất sửa nhãn ở phiên bản khác kèm lý do; không sửa trực tiếp Excel.

### R — Reason Plausibility (1–5/U, duyệt độc lập)

Đánh giá reason dựa vào sự kiện/điều kiện/SOP, không dựa vào độ dài hoặc văn phong. Nếu reason đọc được trực tiếp trong response lỗi JSON, được đánh giá nội dung và ghi evidence_mode; F vẫn bằng 0. Không lấy reason từ cột ref_reason để thay reason mô hình.

| R | Mốc chấm do mình đề xuất |
|---|---|
| 1 | Mâu thuẫn rõ với sự kiện, bịa trạng thái quyết định hoặc lập luận không liên quan |
| 2 | Nhận ra một phần ngữ cảnh nhưng sai/thiếu sự kiện then chốt, lập luận khó hỗ trợ quyết định |
| 3 | Có liên quan và không mâu thuẫn lớn, nhưng quá chung hoặc thiếu liên hệ với trạng thái mới nhất |
| 4 | Bám đúng sự kiện quan trọng và giải thích hợp lý quyết định, còn thiếu chi tiết nhỏ |
| 5 | Giải thích ngắn gọn, rõ nguyên nhân/điều kiện quyết định theo log và SOP/policy, không có giả định vô căn cứ |
| U | Không có hoặc không xác định được reason; báo cáo thiếu riêng |

Reason đúng sự cố vẫn có thể R cao khi command chọn sai hàm. Không chấm reason=1 chỉ vì C=0. Ngược lại command khớp nhưng reason chung chung không tự nhận R=5.

## 3. Chỉ số và mẫu số

Giữ bảng all/SOP/Unexpected cho từng nguồn output, cùng tập test và cùng tiêu chí.

- **AST match**: baseline kỹ thuật đã có, 4/22 mới và 5/22 tác giả. Giữ nguyên; không đổi tên nó thành semantic accuracy.
- **Semantic command correctness**: C chấm như trên. Ghi số đúng k, sai w, chưa xác định u, N=k+w+u. Báo cáo khoảng [k/N, (k+u)/N], cùng tỷ lệ trên các test đã xác định k/(N-u) nếu N>u; phải in rõ coverage và mẫu số. Không trình bày một tỷ lệ đã loại U như thể đủ 22 test.
- **Reason plausibility**: trung bình R trên các reason có điểm, kèm số reason có điểm/tổng và phân bố 1–5; không bỏ qua số missing/U. Không tự coi điểm H của tác giả là cùng mốc R này.
- **Format validity**: tổng F/N.
- **E=F×1[C=1]**: chỉ số bổ sung về tính hợp lệ định dạng + command đúng. C=U thì E chưa xác định nếu F=1, E=0 nếu F=0. Đây không phải metric paper công bố và không chứng minh thực thi đúng trên phần cứng.

Không gộp R vào accuracy, không dùng H=5 thay C, không so tỷ lệ 22 test với 65% trên 100 test. Chấm lại cả hai nguồn trước khi kết luận chênh lệch.

## 4. Trình tự duyệt để hạn chế thiên lệch

1. Đọc SOP, event và reference để lập case specification trước khi chấm output. Vì đã thấy output của run này, ghi nhận giới hạn đó.
2. Đánh dấu case conflict dựa vào dữ liệu, áp dụng cho cả hai nguồn. Test 017 cần giữ U ở phép so dựa reference hiện tại.
3. Dùng review_blind.jsonl; mỗi test có hai response mã A/B, không hiện model/source, điểm G/H hoặc kết quả AST. Sự khác biệt phong cách vẫn có thể làm người đọc đoán nguồn; không khẳng định blind tuyệt đối.
4. Điền review_scores.csv: C, R, evidence_mode, ghi chú và quyết định case conflict. Không tự tính điểm semantic bằng regex hoặc LLM judge mà gọi là nhãn con người.
5. Duyệt lại các ca không rõ, ghi rationale; một người chấm phải ghi là single-rater, không tuyên bố độ đồng thuận liên giám khảo.
6. Chỉ mở answer_key.csv sau khi chấm xong. Tính các chỉ số cùng rubric/version. Nếu đổi quy tắc sau khi thấy nguồn, chấm lại cả hai bộ và tăng version.

## 5. Những chỗ đang cần tác giả xác nhận

- G, H và I trong workbook phản ánh các khái niệm nào; chỉ số nào dùng cho command accuracy.
- Thông báo alert được chấp nhận theo ngữ nghĩa hay khớp chuỗi.
- JSON lỗi có được tính đúng ý định và/hoặc đúng command hay không.
- Policy cho emergency_stop/retry khi reference yêu cầu alert.
- Reference của case 017 và chi tiết SOP C1/C2 ở BG56.

Kiểm tra toàn bộ ori_response của 22 dòng Qwen2 trong workbook: 008 là văn bản event log thay JSON; 014 và 016 có JSON lỗi dấu nháy. Vì vậy F của tác giả theo quy tắc strict đề xuất là 19/22. Đây là chỉ số mình tính lại từ workbook, không phải con số tác giả công bố. F của run mới đã kiểm tra trước đó là 22/22; không suy ra từ đó C hoặc R cao hơn.

Không gọi rubric này là nguyên bản tác giả. Tên phù hợp: **rubric tái dựng theo hai chỉ số của LLM4IAS, phiên bản v1**.
