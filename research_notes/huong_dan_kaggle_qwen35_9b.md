# Qwen3.5-9B — phiên bản Kaggle so sánh với Qwen2

Ngày chuẩn bị: 29/10/2026. Model: **Qwen/Qwen3.5-9B**, không phải Qwen3.8 hay bản 27B.

## 1. File dùng để chạy

```text
experiments/sop_aircraft/kaggle_qwen35_9b_sop_aircraft.ipynb
```

Đây là notebook độc lập cho **SOP-Bench / aircraft_inspection, 112 task có nhãn**, không bao gồm 22 test LLM4IAS. Runner của mình, không phải code gốc của bài 2409 hoặc SOP-Bench. Không fine-tune và không cần API key. Dữ liệu, tool và SOP tự tải như bản Qwen2.

Giữ nguyên notebook/session Qwen2 đang chạy. Người thực hiện notebook Qwen3.5 dùng một phiên mới với GPU của chính họ; chia sẻ file notebook và kết quả, không chia sẻ thông tin đăng nhập. Dùng Internet và quota GPU theo tài khoản Kaggle.

## 2. Những gì giữ nguyên để so sánh

| Thành phần | Qwen2 run v2 | Qwen3.5 notebook này |
|---|---|---|
| Dataset revision | 92633eae202fbf66cceb374c6ad55da41ca693b6 | Giống |
| Labeled cases và thứ tự | 112, sort aircraft_id | Giống |
| SOP, tool specs, mock tools | Bản đã tải/kiểm hash | Giống |
| System prompt/protocol | json_transport_v2 | Giống nội dung/hash |
| Rubric | 2409 v1, C/R manual | Giống |
| Decoding | Greedy, 384 output token, tối đa 12 lượt | Giống các giới hạn này |
| Precision mặc định | FP16 | FP16, nếu phần cứng chạy được |
| Chat wrapping | Qwen2 template | Template Qwen3.5, enable_thinking=False |
| Thư viện | Transformers 4.51.3 | Transformers 5.21.1 |

System prompt đã kiểm tra khớp run pilot Qwen2 của bạn:

```text
50a6dce6cddaa49f4c43434ebe48697bb33a6c67b47d79bcf3421c0379fc445e
```

Khác tokenizer/kiến trúc và thư viện là khác biệt phải ghi nhận; cùng 384 token không đồng nghĩa lượng ký tự/cost giống nhau. Greedy non-thinking được chọn làm controlled baseline, không khẳng định đây là cấu hình tối ưu riêng Qwen3.5. Model card có các khuyến nghị sampling khác; nếu muốn thử phải tạo run riêng, không đổi giữa chừng.

## 3. Cách chạy

1. Tạo notebook Kaggle mới, Import file ipynb ở trên.
2. Settings: Internet ON, GPU T4 x2 nếu có. Không chọn TPU cho notebook CUDA này.
3. Cell 1: cài **Transformers 5.21.1, Accelerate 1.15.0** và phụ thuộc. Không dùng cell cài môi trường Qwen2, không pin huggingface_hub<1 ở đây. Giữ torch/CUDA của Kaggle. Python >=3.11.
4. Cell 2: kiểm tra GPU và import đúng lớp Qwen3.5. Kỳ vọng hai T4 còn trống, dòng `Model class import OK`. Giữ `LOAD_IN_4BIT=False`, `ENABLE_THINKING=False`.
5. Cell 3: định nghĩa adapter. Không có output cũng bình thường. Đã có protocol v2, **không chạy bản vá Kaggle cũ**.
6. Cell 4: tải dữ liệu, kiểm tra hash, in 112 mẫu và 7 mock tool.
7. Cell 5: tải model. Model SHA cố định `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Kiểm tra không có missing/mismatched weights; device map phải trên GPU. Đợi `Model loaded. Run: ...`.
8. Cell 6: định nghĩa generate non-thinking.
9. Cell 7: một task smoke test. Giữ cả output sai. Nếu lỗi runtime, lưu traceback trước khi chạy rộng.
10. Cell 8: pilot `RUN_COUNT=10`, sau đó cell 9 chấm và cell 10 ZIP.
11. Nếu pipeline hoạt động và không đổi setting, đặt `RUN_COUNT=112`, chạy lại cell 8/9/10; không tải lại model. Các task đã có trong run được dùng lại, không chọn lại output đúng hơn.

Output nằm trong:

```text
/kaggle/working/sop_aircraft_qwen35_9b/runs/<run-id>/
```

Khác thư mục Qwen2, không ghi đè run cũ. Đổi tên ZIP tải về thành tên mô tả model, precision và pilot/full để dễ đối chiếu. Metadata, raw responses, observations và bảng điểm đều được lưu.

## 4. Những điểm kỹ thuật phải biết

- Checkpoint Qwen3.5-9B công khai là kiến trúc multimodal. Notebook nạp đúng lớp **Qwen3_5ForConditionalGeneration** nhưng chỉ truyền text, không dùng hình ảnh. Vision weights vẫn chiếm bộ nhớ. Không tự ép dùng lớp Qwen2 hoặc text-only config làm lệch prefix trọng số.
- T4 không có BF16 native như các GPU mới; notebook chọn FP16. Ngưỡng 24 GiB trống tổng chỉ là kiểm tra chuẩn bị, không đảm bảo không OOM khi generate.
- Qwen3.5 dùng hybrid attention; thiếu kernel tối ưu có thể đi theo PyTorch fallback chậm hơn/tốn bộ nhớ. Không tự cài FlashAttention/FLA trên T4 chỉ vì thấy warning. Gửi nguyên log nếu có lỗi thực sự.
- Chat template thực được gọi với enable_thinking=False; không dùng /nothink thay tham số. Không strip think blocks hoặc sửa JSON của model trước khi chấm F.
- Không có FlashAttention/compile hay forced JSON decoding riêng cho Qwen3.5: tránh thêm một lợi thế khác với baseline Qwen2.
- Model SHA, loading info, template hash, template text, phiên bản thư viện, precision và chế độ thinking được lưu trong ZIP.

## 5. Nếu gặp lỗi

| Lỗi | Xử lý |
|---|---|
| Cell 1 báo HF đã import | Restart session/kernel rồi cài từ đầu; dùng notebook mới |
| Lỗi import Qwen3_5 hoặc KeyError qwen3_5 | Kiểm tra phiên bản Transformers thực tế và restart sau cài; không hạ về 4.51.3 |
| OOM khi tải/sinh | Giữ log; restart, thử LOAD_IN_4BIT=True nếu cần và ghi rõ NF4. Không nối kết quả FP16 và NF4 vào cùng run |
| Missing/mismatched weights | Dừng, không đánh giá model khởi tạo thiếu trọng số; gửi loading info |
| CPU/disk offload | Chọn GPU đủ bộ nhớ hoặc restart thử NF4; không báo latency như GPU-only |
| Cảnh báo fallback kernel | Theo dõi tiến độ/tài nguyên; có thể chậm nhưng không tự kết luận model trả sai |
| Model sai JSON/command hoặc hết 12 lượt | Giữ làm kết quả lỗi mô hình/protocol; không retry chọn output đẹp hơn |

Nếu buộc dùng Qwen3.5 NF4 còn Qwen2 FP16, vẫn có thể báo cáo hai cấu hình triển khai nhưng chưa quy hết chênh lệch cho riêng model. Muốn tách ảnh hưởng lượng tử hóa cần một ablation riêng về precision.

## 6. Trạng thái kiểm tra và nguồn

Đã kiểm tra cú pháp mọi code cell; các cell tải dữ liệu, chạy task và chấm dùng chung với Qwen2. Đã kiểm tra hash system prompt khớp run Qwen2 pilot, kiểm tra suffix non-thinking bằng template chính thức, xác nhận config kiến trúc và lớp loader trong mã Transformers 5.21.1.

**Chưa tải trọng số hoặc chạy inference Qwen3.5 trên GPU/Kaggle.** Không có kết quả accuracy/latency mới, chưa bảo đảm môi trường GPU cụ thể không cần chỉnh phụ thuộc. Chạy smoke test trước.

Nguồn: [model card](https://huggingface.co/Qwen/Qwen3.5-9B), [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [dataset](https://huggingface.co/datasets/amazon/sop-bench), [Transformers loader source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
