# R1 Label and Image-Content Contract

## Trạng thái

- Contract: `r1_label_contract_v1`
- Trạng thái: `proposed_for_pilot_review`
- Mục tiêu: phân loại lớp của sản phẩm chiếm ưu thế và nhìn thấy được trong ảnh
- Quyết định hợp lệ: `accept`, `quarantine`, `relabel`
- Xóa hoặc ghi đè ảnh nguồn: không được phép
- Sử dụng split hoặc model output khi review: không được phép
- Đánh giá Final Test: không được phép

Machine-readable authority nằm tại `configs/r1_label_contract.json`.

## Ground Truth

Ground truth được xác định từ nội dung sản phẩm nhìn thấy trong ảnh. Query crawl,
tên thư mục, tên file và metadata listing chỉ là bằng chứng hỗ trợ. Chúng không
được ghi đè nội dung hình ảnh.

Một ảnh chỉ được `accept` khi có một sản phẩm chính hoặc nhiều biến thể cùng lớp,
đồng thời bao bì, hình dạng hoặc chữ sản phẩm cung cấp đủ bằng chứng cho nhãn hiện
tại. Template Tiki, watermark, màu nền và banner khuyến mãi không phải bằng chứng
phân lớp.

## Decision Contract

### Accept

`accept` được dùng khi sản phẩm chính khớp nhãn hiện tại. `proposed_label` phải
bằng `original_label`. Reason code phải bắt đầu bằng `ACCEPT_`.

### Relabel

`relabel` chỉ được dùng khi ảnh có đúng một lớp khác trong mười lớp mục tiêu và
bằng chứng đủ rõ. `proposed_label` phải khác `original_label`. Reason code phải
bắt đầu bằng `RELABEL_`.

### Quarantine

`quarantine` được dùng khi ảnh mơ hồ, nhiều lớp, hybrid, ngoài ontology, không có
sản phẩm chính hoặc không đủ thông tin. `proposed_label` phải là `null`. Reason
code phải bắt đầu bằng `QUARANTINE_`.

Quarantine là quyết định không phá hủy dữ liệu. File ảnh nguồn được giữ nguyên.

## Image Rules

### Bundle

Bundle chỉ được accept khi mọi sản phẩm nhìn thấy đều thuộc cùng một lớp và cùng
mục đích sử dụng. Bundle gồm nhiều lớp mục tiêu phải quarantine.

### Lifestyle và before-after

Ảnh người mẫu, lifestyle hoặc before-after chỉ được accept khi sản phẩm vẫn là
chủ thể chính và đủ bằng chứng phân lớp. Nếu sản phẩm không rõ, ảnh phải
quarantine.

### Banner và text-only

Ảnh text-only phải quarantine. Ảnh có banner chỉ được accept khi sản phẩm vẫn rõ
và nhãn không phụ thuộc vào template quảng cáo. Banner hoặc watermark lặp lại sẽ
được audit tiếp ở R2 và R3.

### Hybrid product

Sản phẩm có hai chức năng thuộc hai lớp mục tiêu, như shampoo-body wash hoặc
moisturizer-sunscreen, phải quarantine nếu không có chức năng chính rõ ràng.

### Generated image

Ảnh có hậu tố augmentation hoặc derivative offline không được đưa vào pilot,
full semantic review, split, model selection hoặc evaluation.

## Class Contract

### body_wash

Sản phẩm làm sạch cơ thể và rửa trôi khi tắm. Body wash, shower gel và shower
cream được accept. Shampoo, facial cleanser, hand wash, lotion và sản phẩm
shampoo-body wash không có chức năng chính rõ phải loại khỏi lớp này.

### face_mask

Sản phẩm chăm sóc da mặt được bán rõ là sheet mask, wash-off mask, clay mask hoặc
sleeping mask. Khẩu trang, hair mask, dụng cụ đắp mask và ảnh khuôn mặt không có
sản phẩm phải quarantine.

### facial_cleanser

Sản phẩm làm sạch da mặt và rửa lại với nước, gồm face wash, cleansing foam, gel
hoặc cream cleanser. Micellar water, cleansing oil, toner, body wash và ảnh người
đang rửa mặt không có sản phẩm không thuộc lớp này.

### lipstick

Mỹ phẩm có chức năng chính tạo màu môi, gồm lipstick, liquid lipstick và lip
tint. Son dưỡng không màu, chì kẻ môi, lip gloss trong suốt và ảnh môi không có
sản phẩm không thuộc lớp này.

### moisturizer

Sản phẩm leave-on có chức năng chính dưỡng ẩm da dưới dạng cream, lotion hoặc
gel. Serum, toner, sunscreen và sleeping mask được định danh rõ không thuộc lớp
này. Moisturizer có SPF ngang vai trò dưỡng ẩm phải quarantine như hybrid.

### perfume

Sản phẩm tạo hương cho cơ thể như perfume, fragrance, EDP, EDT hoặc body mist.
Tinh dầu phòng, diffuser, nước hoa xe, deodorant và toner nước hoa hồng không
thuộc lớp này.

### serum

Sản phẩm leave-on chăm sóc da cô đặc được định danh là serum hoặc ampoule. Toner,
essence không định danh serum, hair serum và chai dropper không đủ chữ phải loại
khỏi lớp hoặc quarantine.

### shampoo

Sản phẩm làm sạch tóc hoặc da đầu và rửa trôi. Conditioner, hair mask, hair
serum, body wash và shampoo-body wash không có chức năng chính rõ không thuộc lớp
này.

### sunscreen

Sản phẩm có chức năng chính bảo vệ da khỏi tia UV và thể hiện rõ sunscreen, SPF
hoặc PA. Foundation, moisturizer và lip product chỉ có SPF như lợi ích phụ không
thuộc lớp này. Sản phẩm hybrid không có chức năng chính rõ phải quarantine.

### toner

Dung dịch leave-on dùng sau làm sạch để cân bằng hoặc chuẩn bị da mặt, được định
danh là toner hoặc nước cân bằng. Nước hoa, nước hoa hồng thực phẩm, chất dưỡng
hoa, micellar water, essence và chai dung dịch không đủ bằng chứng không thuộc
lớp này.

## Pilot Review

Pilot `r1_semantic_pilot_s42_v1` gồm 100 ảnh gốc, 10 ảnh mỗi lớp. Ảnh được chọn
bằng SHA-256 ranking với seed 42. Pilot không sử dụng split và model output.

Hai reviewer độc lập phải hoàn thành toàn bộ 100 quyết định. Reviewer không được
trao đổi quyết định trước khi nộp review đầu tiên.

Ngưỡng vượt qua R1:

- exact agreement ít nhất `0.85`;
- Cohen's kappa ít nhất `0.80`;
- agreement của từng lớp ít nhất `0.70`;
- không còn policy case chưa giải quyết;
- hai reviewer có identifier khác nhau;
- contract version và pilot SHA-256 khớp tuyệt đối.

Conflict phải được adjudicate bằng chính contract này. Nếu conflict cho thấy rule
chưa đủ rõ, contract phải tăng version và cả hai review phải chạy lại trên cùng
pilot trước full review.

## R1 Exit Gate

R1 chỉ pass khi contract được chủ project duyệt, hai review độc lập vượt toàn bộ
ngưỡng và không còn policy case chưa giải quyết. Trước thời điểm đó, R2 không
được sử dụng quyết định pilot như ground truth chính thức.
