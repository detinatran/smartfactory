# EXT47: predictor C-MAPSS lớn hơn (hidden 128, 3 lớp, tối đa 100 epoch); thăm dò; seed 162–171; 80 ensemble

## RMSE của predictor
RMSE trên mọi window đánh giá: 15.55 với predictor mặc định, 15.39 với predictor lớn (8/10 seed thấp hơn). Model lớn hơn gần như không cải thiện RMSE. Nút thắt có vẻ không nằm ở dung lượng model.

## Lợi ích của RPT-D dưới 6 lỗi
| So sánh | Predictor mặc định | Predictor lớn | Seed cải thiện (cả hai) |
|---|---|---|---|
| RPT-D − OSC | −0.61 | −0.62 | 10/10 |
| RPT-D − Gaussian có bias | −1.04 | −1.09 | 10/10 |
| RPT-D − hồi quy lỗi có dấu | −1.45 | −1.42 | 10/10 |

## Kết luận
Lợi ích không teo lại với model lớn hơn. Tuy vậy, thí nghiệm này chưa trả lời được câu hỏi "predictor ở mức SOTA", vì RMSE gần như không đổi.

## Ghi chú vận hành
Lần chạy đầu với 6 shard làm đầy 24 GB GPU và bị treo. Đã chạy lại với 3 shard. Các CSV đã ghi không bị ảnh hưởng.
