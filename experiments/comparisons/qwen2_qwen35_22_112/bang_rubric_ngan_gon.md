# So sánh theo khung F/C/R — đánh giá AI hỗ trợ sơ bộ

**Trạng thái:** F được kiểm tra tự động; C/R là nhãn đề xuất có AI hỗ trợ, chưa được người chấm độc lập xác nhận. Không giới thiệu đây là số liệu nguyên bản của tác giả hoặc kết quả rubric đã xác nhận.

Cả hai bộ chấm từng phản hồi: F kiểm tra JSON, C xét hành động/điều kiện/tham số, R xét lý do theo thang 1–5. Nguồn tham chiếu khác: bộ 22 có next-command reference và event log; bộ 112 dùng SOP/tool specification/observation. Hỗ trợ chấm cũng khác: bộ 22 được assistant đọc từng case; bộ 112 dùng quy tắc C và mã hóa reason theo nhóm cộng kiểm tra ngữ cảnh. Không coi hai cách triển khai có độ tin cậy giống hệt nhau; chỉ so các model trong từng bộ, không gộp tỷ lệ hai bộ.

## 1. LLM4IAS — 22 test

| Chỉ số — All | Qwen2-7B chạy lại | Qwen3.5-9B | Qwen2 tác giả lưu |
| --- | ---: | ---: | ---: |
| Số task / số phản hồi | 22 / 22 | 22 / 22 | 22 / 22 |
| F — JSON hợp lệ | 22/22 (100,00%) | 22/22 (100,00%) | 19/22 (86,36%) |
| C — Đúng, theo đánh giá sơ bộ | 6/22 (27,27%) | 13/22 (59,09%) | 10/22 (45,45%) |
| C — Sai | 14/22 | 7/22 | 9/22 |
| C — Chưa phân xử (U) | 2/22 | 2/22 | 3/22 |
| C — Khoảng giữ U | 27,27%–36,36% | 59,09%–68,18% | 45,45%–59,09% |
| R — Trung bình đề xuất (1–5) | 3,00/5 | 4,45/5 | 3,24/5 |
| Số phản hồi có điểm R | 22/22 | 22/22 | 21/22 |
| Time — Tổng suy luận | 85,72 giây | 189,33 giây | Không có log |

Case 001 (SOP ghi C2, reference ghi C1) và 017 (switch/C2) được giữ C=U cho cả ba nguồn. Raw tác giả ở case 008 không có một command/reason duy nhất nên C/R=U. Hai raw JSON lỗi 014/016 vẫn đọc được ý định nguyên văn: C/R có điểm nhưng F=0; không sửa JSON. Đây là chấm lại raw Qwen2 tác giả, không lấy điểm H cũ.

Nguồn dữ liệu gốc: [LLM4IAS — evaluation_data.xlsx và prompt_example.txt](https://github.com/YuchenXia/LLM4IAS). Bộ 22 là Inspection Station subset, không phải đủ 100 test trong paper.

## 2. SOP-Bench / aircraft_inspection — 112 test

| Chỉ số — All | Qwen2-7B chạy lại | Qwen3.5-9B |
| --- | ---: | ---: |
| Số task / số phản hồi | 112 / 862 | 112 / 891 |
| F — JSON hợp lệ | 862/862 (100,00%) | 805/891 (90,35%) |
| C — Đúng, theo đánh giá sơ bộ | 379/862 (43,97%) | 611/891 (68,57%) |
| C — Sai | 315/862 | 86/891 |
| C — Chưa phân xử (U) | 168/862 | 194/891 |
| C — Khoảng giữ U | 43,97%–63,46% | 68,57%–90,35% |
| R — Trung bình đề xuất (1–5) | 3,05/5 | 4,28/5 |
| Số phản hồi có điểm R | 862/862 | 805/891 |
| Time — Tổng suy luận | 2 giờ 19 phút | 4 giờ 15 phút |

Không có run của tác giả LLM4IAS trên SOP aircraft trong artifact đã thu được nên không thêm cột tác giả ở bảng 112. Mẫu số F/C/R là số lượt phản hồi (862 và 891), không phải 112 task. Có 86 raw lỗi/đa đoạn của Qwen3.5 không được chọn một reason duy nhất để chấm R, nên phải đọc R cùng độ bao phủ.

**Link dataset 112:** [amazon/sop-bench trên Hugging Face](https://huggingface.co/datasets/amazon/sop-bench), subset **aircraft_inspection**, split **test**, 112 dòng có đáp án. Artifact đã chạy được cố định ở revision `92633eae202fbf66cceb374c6ad55da41ca693b6`; file nhãn là `data/aircraft_inspection/test_set_with_outputs.csv`. [Repository SOP-Bench](https://github.com/amazon-science/SOP-Bench). Đây là workflow với mock tool dựa CSV, không phải mô phỏng vật lý máy bay.

## Cách đọc

- All là toàn bộ test/lượt trong từng bộ, không phải tổng F+C+R.
- U là chưa phân xử; không đổi thành sai. Khoảng C là [đúng/N, (đúng+U)/N], không phải khoảng tin cậy và không bao gồm sai số bộ chấm.
- R trung bình chỉ tính trên phản hồi có điểm. Đây là đề xuất của AI, không phải human rating do tác giả xác nhận. Các quy tắc C và kiểm tra ngữ cảnh R có thể sai hoặc thiếu; giữ căn cứ từng lượt để kiểm tra.
- Time là tổng latency model được lưu, không gồm tải trọng số/cài thư viện. Thư viện và kiến trúc hai model khác nhau.

Chi tiết chấm: [22 test](../../llm4ias/fcr_assisted_22/bao_cao_fcr_22.md) và [112 test](../../sop_aircraft/fcr_review_112/ai_assisted_v1/bao_cao_fcr_ai_so_bo.md). Các chỉ số AST, khớp báo cáo cuối và thứ tự tool vẫn giữ ở [báo cáo tự động](bao_cao_so_sanh.md) / [bảng 112 chi tiết](bang_112_chi_tiet.md), không thay C/R.
