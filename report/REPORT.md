# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** 3.13.15 / 2.11.0+cu128 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không. Chạy toàn bộ notebook `Run all` từ đầu đến cuối, không sửa mã nguồn, checkpoint hay threshold mặc định (`DETECTION_SCORE_THRESHOLD = 0.35`, `SEGMENTATION_SCORE_THRESHOLD = 0.35`).

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    ```json
    {
        "class_id": 468,
        "class_name": "cab",
        "rank": 1,
        "score": 0.510915,
        "taxonomy_name": "ImageNet-1K"
    }
    ```
- **Record này mô tả toàn ảnh như thế nào?**

    Ảnh được gán một nhãn duy nhất, và không có vị trí hay số lượng vật thể. Model chọn `cab` vì đó là lớp có score cao nhất, dù ảnh `traffic` có nhiều xe và người đi bộ khác.

- **Ai định nghĩa class list mà checkpoint có thể dự đoán?**

    Danh sách 1000 lớp ImageNet-1K đã được cố định từ lúc huấn luyện checkpoint `yolo11n-cls.pt`. Model không tự tạo lớp mới được.

- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**

    `class_id` dùng để cho model đối chiếu ổn định; `class_name` cho để con người đọc hiểu; `taxonomy_name` để biết đang dùng bộ nhãn nào (ImageNet-1K), tránh nhầm với taxonomy khác như COCO-80.

- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**

    Guideline cần quy tắc rõ ràng để chọn một chủ thể chính (ví dụ chủ thể to nhất), tránh mỗi người chọn một kiểu.

- **Vì sao model score không phải ground truth?**

    Score chỉ là độ tự tin của model, không phải sự thật hay xác nhận của con người. Trong nhiều trường hợp, model có thể cho dự đoán sai. Ở những trường hợp model score gần nhau, sẽ vẫn cần người kiểm tra lại.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    ```json
    {
        "class_name": "person",
        "score": 0.912625,
        "bbox_xyxy": [385.33, 69.24, 498.92, 348.92],
        "bbox_width": 113.58,
        "bbox_height": 279.68
    }
    ```
- **Diễn giải vị trí box bằng lời:**

    Box bắt đầu ở `(385.33, 69.24)` và kết thúc ở `(498.92, 348.92)` - rộng ~114px, cao ~280px, nằm ở nửa phải khung hình. Đây là người, score ~0.91.

- **So sánh số prediction ở hai threshold:**

    Ở threshold 0.35 có 11 vật thể. Hạ xuống 0.20 sẽ có thêm box score thấp; tăng lên 0.60 sẽ mất các box dưới 0.60, ví dụ `bowl` 0.381, `cup` 0.382, `bowl` 0.465, `bowl` 0.500.

- **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?**

    Threshold thấp bắt nhiều vật thể hơn nhưng reviewer phải lọc nhiều dự đoán sai hơn; threshold cao đỡ việc cho reviewer nhưng dễ bỏ sót vật thể mờ/nhỏ/bị che.

- **Đề xuất một quy tắc box chặt:**

    Box ôm sát mép vật thể ở cả 4 phía, không dư nền, không cắt mất chi tiết nhô ra.

- **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**

    Guideline cần định nghĩa: có gán nhãn phần bị che không, ngưỡng % nhìn thấy tối thiểu, và khi nào phải hỏi reviewer thay vì tự quyết - ví dụ `person` bị cắt sát mép trái (bbox `[0.12, 263.18, 61.55, 310.78]`).

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    ```json
    {
        "instance_id": "kitchen-001",
        "class_name": "person",
        "score": 0.899318,
        "polygon_point_count": 348,
        "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], "..."]
    }
    ```
- **Polygon bổ sung chi tiết gì so với box?**

    Box là hình chữ nhật bao quanh, polygon (348 điểm ở đây) bám theo đúng đường viền thật, kể cả chỗ cong
    lồi lõm - nên tách được vật thể khỏi nền/vật thể khác trong cùng box.

- **`instance_id` dùng để làm gì và không phải loại ID nào?**

    Phân biệt từng vật thể riêng lẻ dù cùng lớp (2 `person` là `kitchen-001` và `kitchen-009`). Không phải
    `class_id` (dùng chung cho cả lớp) và không phải tracking ID xuyên nhiều ảnh/video.

- **Đề xuất một quy tắc biên mask:**

    Biên mask bám sát cạnh thật, không lấn vào thân vật thể, không tràn ra nền quá 1-2 pixel.

- **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**

  Khi vật thể che/dính nhau (như `spoon` kitchen-011 chỉ còn 37 điểm polygon so với 348 của người) thì khó
  biết biên thật. Guideline cần quy định mask dừng ở đâu, và khi nào phải báo reviewer thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_id`/`class_name` duy nhất cho cả ảnh, theo taxonomy ImageNet-1K | Ảnh `traffic` có nhiều xe nhưng chỉ được gán một lớp (`cab`, score 0.51); lớp đứng thứ 2 là `minibus` với 0.16, khá gần nghĩa, nên ranh giới hơi mơ hồ | Chọn một lớp duy nhất theo đúng quy tắc guideline (ví dụ chủ thể lớn nhất), không nghe theo gợi ý của model | Đối chiếu với ảnh gốc xem lớp được chọn có đúng quy tắc không |
| Phát hiện vật thể | Danh sách các cặp `(class_name, bbox_xyxy)`, số lượng box tùy ảnh | Vật thể bị cắt sát mép ảnh (`person` góc trái, `x_min` gần bằng 0) và mấy box có score ở khoảng 0.35–0.50 dễ mất/xuất hiện tùy threshold chọn | Vẽ box ôm sát từng vật thể theo taxonomy COCO-80, kể cả những vật thể model bỏ sót | So ảnh gốc với box xem có khít không, có bỏ sót vật thể nào không, có box nào trùng hoặc sai lớp không |
| Instance segmentation | Polygon theo biên pixel cho từng instance, có `instance_id` riêng trong ảnh | Instance bị che có ít điểm polygon hẳn (`spoon` kitchen-011 chỉ 37 điểm so với 348 điểm của người), khó biết biên thật khi vật thể xếp sát nhau (mấy cái `bowl`) | Vẽ polygon bám sát biên thật từng instance, xử lý phần bị che theo đúng guideline | Kiểm tra polygon có bám đúng biên không, các instance cùng lớp có tách nhau rõ ràng không, phần bị che có xử lý đúng cách không |

## 5. An toàn dữ liệu

- **Một quy tắc bảo vệ dữ liệu:**

    Chỉ dùng ảnh công khai rõ nguồn gốc và license (COCO 2017, CC BY 2.0, có checksum SHA-256). Không tải
    ảnh cá nhân, dữ liệu nội bộ lên Colab/GitHub, không ghi họ tên/MSSV/email/SĐT vào báo cáo.

- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**

    Giảng viên hoặc Lab Coach của lớp, dừng xử lý cho tới khi có hướng dẫn.

## 6. Danh sách bằng chứng
- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
