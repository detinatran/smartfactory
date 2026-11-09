# Bộ đã chọn: SOP-Bench / aircraft_inspection trên Kaggle

Ngày: 29/10/2026. Rubric 2409 v1 đã được người dùng chấp nhận, chưa phải rubric được tác giả xác nhận.

### Cập nhật sau smoke test đầu tiên: transport v2

Run đầu `a_00123` đã gọi được 7 tool nhưng sau đó sinh bare `finish_inspection(...)` thay vì JSON. Lượt 8–12 bị parser từ chối, hết 12 lượt, chưa có final result. Đây là failure hợp lệ của run v1, phải giữ lại; không sửa output hoặc parser để biến thành đúng. Reason nhiều lượt chỉ lặp tên hàm, và lượt 2 truyền kết quả kiểm tra chưa quan sát được. Tool không báo lỗi không chứng minh tuân thủ SOP.

Notebook hiện tại có **transport v2**: thêm ví dụ finish được bọc JSON rõ ràng và nhắc dùng quan sát thật, không thêm đáp án. Model, scoring, giới hạn lượt và tool giữ nguyên. Bản notebook v1 được giữ dưới tên `kaggle_qwen2_sop_aircraft.protocol_v1.ipynb`. Đây là biến thể prompt sau khi đã thấy smoke test, không tuyên bố preregister hoặc tái lập setting gốc. Không tăng MAX_TURNS chỉ để cứu test này.

Nếu model còn đang nạp trong session Kaggle cũ: mở `research_notes/kaggle_patch_sop_aircraft_protocol_v2.py`, copy **toàn bộ code** vào một cell mới sau cell 6, chạy đúng một lần, rồi chạy lại cell 7. Patch tự ZIP run cũ, tạo run mới và ghi metadata; không tải lại model hoặc ghi đè output cũ. Nếu dùng notebook mới từ đầu, không chạy patch vì v2 đã được tích hợp.

Patch xử lý hướng dẫn định dạng, không bảo đảm model sẽ trả đúng hoặc tuân thủ SOP. Nếu vẫn lỗi, giữ trace như kết quả; không chấp nhận trực tiếp bare call thành JSON, không điền field thiếu bằng reference.

## 1. Vì sao chọn và giới hạn

Nguồn dataset: https://huggingface.co/datasets/amazon/sop-bench

Nguồn framework: https://github.com/amazon-science/SOP-Bench

Bộ con aircraft_inspection có SOP, mô tả 7 tool, input trạng thái thiết bị/kiểm tra và đáp án báo cáo cuối. Hợp hơn QA chung ở chỗ đòi hỏi làm quy trình inspection, dùng công cụ và xử lý bất thường. Không cần model nhìn ảnh; Qwen2-7B-Instruct text-only dùng được.

Khác biệt phải giữ rõ: đây là kiểm tra trước chuyến bay theo workflow nhiều bước, **không phải trạm sản xuất hoặc event-log next-command đúng như bài 2409**. Dùng để đánh giá bổ sung khả năng tuân thủ SOP/tool-use, không thay LLM4IAS làm benchmark chính, không gọi đây là tái lập bài 2409 trên dataset gốc.

Các tool là mock API tra CSV, không phải thiết bị thật hoặc simulator vật lý. Một số tool trả về giá trị đã lưu trong CSV; đó là thiết kế mock của dataset. Runner chỉ cung cấp phản hồi khi model gọi tool qua API giới hạn, không đưa nguyên CSV đáp án/source tool vào prompt. Không diễn giải kết quả thành chứng minh kiểm chứng an toàn vật lý.

## 2. Dữ liệu thực sự dùng

Đã cố định HF revision:

```text
92633eae202fbf66cceb374c6ad55da41ca693b6
```

- `test_set_with_outputs.csv`: **112 mẫu có nhãn**, dùng để chấm offline.
- `test_set_without_outputs.csv`: 38 mẫu không có nhãn; không đưa vào mẫu số khi chưa có evaluator/ground truth.
- SOP, metadata, tool specs và tools.py: đã tải và đọc trực tiếp.
- License dữ liệu/tools: CC-BY-NC-4.0; lưu trích dẫn, tuân thủ giấy phép.

Giới thiệu có thể nêu tổng 150, nhưng notebook không báo đã chấm 150. 112 mẫu được xếp theo aircraft_id, pilot lấy 10 mẫu đầu trước khi xem kết quả model; không chọn mẫu theo tỷ lệ đúng.

## 3. Notebook đã chuẩn bị

Upload file này lên Kaggle:

```text
experiments/sop_aircraft/kaggle_qwen2_sop_aircraft.ipynb
```

Adapter nguồn để đọc/sửa có phiên bản: `research_notes/sop_aircraft_runtime.py`.
Script tạo notebook: `research_notes/build_kaggle_sop_aircraft.py`.

Notebook đã nhúng adapter nên **không cần upload các file Python khác**. Không cần clone repo hoặc cài framework Bedrock. File dataset/tool công khai được tải và kiểm tra hash trong notebook. Không cần API key OpenAI hoặc GitHub PAT.

## 4. Các bước chạy Kaggle

1. Import Notebook từ file ipynb.
2. Bật Internet và GPU. Ưu tiên T4 x2 nếu có, giống lần chạy Qwen2 trước. Không chọn TPU vì runner này dùng CUDA.
3. Chạy cell 1 cài thư viện. Giữ torch/CUDA của Kaggle. Nếu đã import Transformers bản khác trước đó, restart kernel rồi chạy từ đầu.
4. Chạy cell 2. Mặc định `LOAD_IN_4BIT=False`; dùng FP16 và cùng SHA Qwen2 của run LLM4IAS trước. Nếu không đủ GPU trống, chọn phần cứng khác hoặc bật 4-bit trước khi nạp model và báo cáo biến thể rõ ràng.
5. Chạy cell 3 định nghĩa adapter.
6. Chạy cell 4 tải 5 file và chuẩn bị dữ liệu. Kỳ vọng `Labeled cases: 112` và danh sách 7 tool. Không cần download ảnh hoặc toàn bộ benchmark.
7. Chạy cell 5 tải model. Không chạy lại cell này nhiều lần khi model đang chiếm GPU.
8. Chạy cell 6 định nghĩa hàm generate.
9. Chạy cell 7 smoke test một task. Task có nhiều lượt gọi tool, nên lâu hơn một prompt LLM4IAS. Một câu trả lời sai không phải lỗi phần mềm; giữ lại.
10. Chạy cell 8 với `RUN_COUNT=10` để pilot. Task đầu đã lưu được tái sử dụng. Mỗi task độc lập, tối đa 12 lượt. Lỗi tool được trả cho model như observation, giữ trong trace; lỗi kỹ thuật/OOM dừng và lưu hồ sơ.
11. Chạy cell 9 tổng hợp và cell 10 tạo ZIP. Tải ZIP trong Output/Files.
12. Nếu pipeline hoạt động, đổi `RUN_COUNT=112` ở cell 8, chạy lại **cell 8, 9, 10**, không cần nạp lại model. Giữ cùng prompt/model/settings; kết quả 10 task đầu được giữ, chạy tiếp các task còn lại.

Nếu đổi prompt, model, precision hoặc các giới hạn, phải tạo run mới. Không retry task chỉ để tăng accuracy. Nếu hết lượt, task vẫn nằm trong mẫu số của tập đã chọn. Chưa có ước tính chắc chắn thời gian cho 112 task; chạy pilot trước để đo theo GPU/runtime thực tế.

## 5. Giao thức của adapter

Mỗi lượt model sinh JSON có reason và command string, ví dụ:

```json
{"reason":"Check the aircraft identity and clearance first.","command":"VerifyAircraftClearance(aircraft_id='a_00123', tail_number='N12345', maintenance_record_id='mr_00001', expected_departure_time='2025-04-18T10:00:00Z')"}
```

Ví dụ chỉ minh họa schema, không phải dữ liệu test hay đáp án.

Runner parse AST, chỉ cho gọi đúng whitelist mock tool bằng các keyword/literal. Không eval/exec model text. Model không được truy cập aircraft_id khác. Sau mỗi tool call sẽ có observation hoặc lỗi. `finish_inspection(result={...})` là hàm kết thúc do mình thêm cho transport, thay wrapper báo cáo XML của SOP bằng giao thức JSON nhất quán với rubric. SOP còn lại giữ nguyên.

Đây là **adaptation được công bố**, chưa phải cách chạy nguyên framework SOP-Bench. Không so trực tiếp với leaderboard SOP-Bench. Ghi cả revision dataset, model SHA, wrapper và generation settings khi báo cáo.

## 6. Cách đọc điểm, giữ rubric 2409 v1

Rubric F/C/R giữ nguyên cho LLM4IAS. Với dataset bổ sung:

- **F**: chấm JSON reason/command hợp lệ của từng lượt; parse được JSON chưa đủ để tool call hợp lệ.
- **C theo từng hành động**: dùng SOP + inputs + observations để duyệt trace; bộ này không có một reference command duy nhất cho mỗi lượt, nên để C manual trống. Không tự gán điểm từ việc tool chạy được.
- **R**: vẫn chấm riêng 1–5 bằng rubric v1, để trống đến khi người chấm duyệt.
- **Final-field match**: chỉ số task bổ sung tự động, kiểm tra 7 trường kết quả cuối cùng với CSV. Chuẩn hóa hoa/thường và null có khai báo; thiếu một key vẫn không được tính đúng. Báo số đúng/n task đã chọn.
- **Initial-three order check**: kiểm tra ba tool đầu theo thứ tự định danh → cơ khí → điện. Chỉ là chẩn đoán một phần, không phải kiểm chứng đầy đủ các nhánh SOP.

Không gọi final-field match là C của bài 2409, không trộn tỷ lệ của 112 workflow với 22 next-command test. Kết quả field cuối đúng vẫn có thể đi kèm quy trình sai hoặc reason kém; cần trace.

## 7. Tool có giới hạn đã biết

Audit offline gọi cứng 7 tool trên 112 dòng ghi nhận 7 lỗi ReportCrossCheck khi trạng thái tiền đề rỗng. Audit này **không phải model run**, cũng không khẳng định mọi task bắt buộc phải gọi cả 7 tool: SOP có nhánh có điều kiện. Notebook giữ nguyên tool gốc; không điền trạng thái giả, không bỏ mẫu hoặc sửa đáp án để làm điểm cao hơn.

Tool lookup không xác minh mọi tham số có đúng vật lý hay không; việc nó trả status không chứng minh model gọi với đủ ý nghĩa. Đó là lý do phải giữ semantic review C và không gọi benchmark này là simulator factory.

## 8. File kết quả

Trong ZIP:

- `metadata.json`, `pip_freeze.txt`, `system_prompt.txt`.
- `prompts/`: messages đầu vào mỗi task, không có đáp án.
- `responses/`: toàn bộ raw output, lệnh, đối số, observations, lỗi và số token/thời gian theo lượt.
- `task_scores.csv`: kết quả task và các trường match.
- `rubric_review.csv`: chỗ chấm C/R thủ công theo lượt.
- `summary.json`: số task được chọn, kết thúc, đúng field, lỗi tool, lượt đúng định dạng và thời gian generate.

ZIP không chứa trọng số model. Lưu kết quả về project trước khi đóng session Kaggle.

## 9. Trạng thái kiểm tra

Đã kiểm tra hash raw các file, schema 112 mẫu, tách input khỏi nhãn và số tool. Đã kiểm tra cú pháp toàn bộ notebook và một luồng 8 lượt bằng response kịch bản để thử adapter/chấm. Response kịch bản chỉ là kiểm tra phần mềm, **không có accuracy Qwen2 mới**. Chưa chạy model trên Kaggle cho bộ này; có thể còn lỗi môi trường hoặc bộ nhớ cần xử lý sau smoke test.
