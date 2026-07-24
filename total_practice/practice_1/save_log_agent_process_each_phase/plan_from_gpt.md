Implementation Plan — PyTorch FashionMNIST Classification

Tôi đã đối chiếu đề bài trong slide, pipeline Deep Learning đã thống nhất, nội dung ANN–backpropagation–MLP trong Chapter 2, phần DataLoader và optimization trong Chapter 4, cùng chuỗi tutorial chính thức của PyTorch. Đề yêu cầu đầy đủ từ Dataset/DataLoader → Transform → Model → Autograd → Optimization → Evaluation → Saving/Loading, sau đó thực nghiệm hyperparameter và trực quan hóa kết quả.

Notebook không nên chỉ là một đoạn code train model. Cấu trúc phù hợp nhất là một mini Deep Learning experiment, có baseline, validation set, log thực nghiệm, chọn model và chỉ dùng test set ở cuối.

1. Phân tích đề bài
1.1. Bài toán

Đây là bài toán:

Input  : Ảnh quần áo grayscale 28 × 28
Output : Một trong 10 lớp
Task   : Supervised multi-class image classification
Model  : Artificial Neural Network bằng PyTorch
Metric : Accuracy và Cross-Entropy Loss

FashionMNIST chính thức gồm 60.000 ảnh training và 10.000 ảnh test; mỗi ảnh là ảnh grayscale kích thước 28 × 28, thuộc một trong 10 lớp.

1.2. Deliverables cần tạo
Deliverable	Cách thể hiện
Python code	Toàn bộ trong một notebook .ipynb
Brief report	Các Markdown cell trong notebook
Loss graphs	Lưu thành ảnh và hiển thị trong notebook
Predicted vs actual images	Grid ảnh kèm nhãn thật và nhãn dự đoán
Saved model	File .pth
Experiment result	Bảng cấu hình và kết quả từng experiment
1.3. Phạm vi phù hợp

Model chính nên là MLP – Multilayer Perceptron, vì bài nằm trong Chapter 2 về Artificial Neural Networks và slide trình bày kiến trúc đầu vào 784 neuron, hidden layers và 10 output neurons. CNN chỉ nên là phần mở rộng, không nên để nó che mất mục tiêu học Flatten, Linear, ReLU, Autograd và training loop.

2. Quyết định thiết kế quan trọng
2.1. Không dùng test set để tuning

Tập dữ liệu sẽ được tổ chức:

60.000 training images
        │
        ├── 54.000 train
        └──  6.000 validation

10.000 official test images
        └── Chỉ mở sau khi model và hyperparameters đã được khóa

Vai trò:

Train set: cập nhật weights.
Validation set: chọn architecture, learning rate, dropout và checkpoint.
Test set: đánh giá cuối cùng đúng một lần.

Cách này tuân thủ nguyên tắc chống leakage trong pipeline đã gửi: split trước, tuning bằng validation, test chỉ dùng ở cuối.

Có thể dùng torch.utils.data.random_split() với seed cố định. Sau khi split cần kiểm tra lại phân phối lớp của train và validation. Nếu giảng viên yêu cầu stratified split tuyệt đối, có thể tạo indices bằng StratifiedShuffleSplit, nhưng không cần đưa dependency đó vào baseline đầu tiên.

2.2. Transform

Transform mặc định:

PIL Image
    ↓
ToImage()
    ↓
ToDtype(torch.float32, scale=True)
    ↓
Tensor shape [1, 28, 28], pixel trong [0, 1]

API torchvision.transforms.v2 hiện dùng ToImage() và ToDtype(..., scale=True) để chuyển ảnh thành tensor float32 và scale pixel về [0,1].

Quyết định về label

Không one-hot encode label trong notebook chính.

Label giữ ở dạng:

0, 1, 2, ..., 9
dtype = torch.int64

Lý do là model dùng nn.CrossEntropyLoss(), nhận raw logits và class index. Tutorial optimization chính thức của PyTorch cũng dùng CrossEntropyLoss trực tiếp với label class index.

2.3. Model không chứa Softmax ở output

Model trả về:

logits.shape = [batch_size, 10]

Không thêm Softmax vào cuối model khi training:

Linear(..., 10) → raw logits → CrossEntropyLoss

Chỉ dùng softmax() khi cần hiển thị confidence dự đoán. Tutorial xây dựng model chính thức cũng cho MLP trả về logits từ layer cuối.

2.4. Baseline model

Baseline nên đủ đơn giản để debug:

Input: [B, 1, 28, 28]
        ↓
Flatten
        ↓
Linear(784, 128)
        ↓
ReLU
        ↓
Linear(128, 10)
        ↓
Logits: [B, 10]

Cấu hình baseline đề xuất:

Thành phần	Giá trị
Architecture	784 → 128 → 10
Activation	ReLU
Loss	CrossEntropyLoss
Optimizer	SGD
Learning rate	0.01
Batch size	64
Epochs	10
Seed	42

Model chính thức trong PyTorch tutorial cũng được xây bằng cách subclass nn.Module, sử dụng Flatten, Linear, ReLU và output 10 logits cho FashionMNIST.

3. Cấu trúc thư mục dự kiến
fashion_mnist_project/
│
├── fashion_mnist_classification.ipynb
│
├── data/
│   └── FashionMNIST/
│
├── models/
│   └── best_fashion_mnist_mlp.pth
│
├── outputs/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── experiment_comparison.png
│   ├── predictions_correct_wrong.png
│   └── confusion_matrix.png
│
└── results/
    ├── experiment_results.csv
    └── model_config.json

Brief report sẽ đặt trong notebook nên không bắt buộc tạo thêm file Word hoặc PDF.

4. Cấu trúc notebook chuẩn
Section 0 — Title và mục tiêu
Markdown cell
PyTorch FashionMNIST Classification

Objectives:
1. Load and explore FashionMNIST.
2. Build an MLP classifier.
3. Implement training and evaluation loops.
4. Experiment with architecture and hyperparameters.
5. Visualize training performance and predictions.
6. Save and reload the trained model.

Ghi:

Tên sinh viên.
MSSV.
Môn học.
Ngày thực hiện.
PyTorch version.
Section 1 — Problem Definition

Markdown giải thích ngắn:

What:
Phân loại ảnh thời trang thành 10 lớp.

Why:
Thực hành toàn bộ PyTorch workflow.

Input:
Tensor ảnh [1, 28, 28].

Output:
10 logits tương ứng với 10 lớp.

Primary metric:
Validation/Test Accuracy.

Secondary metric:
Cross-Entropy Loss.

Thêm nguyên tắc:

Validation dùng để lựa chọn model; official test set chỉ dùng sau khi khóa cấu hình.

Section 2 — Imports và reproducibility
Packages
torch
torch.nn
torch.optim
torchvision
matplotlib
numpy
pandas
random
pathlib
Việc cần thực hiện
In phiên bản Python.
In phiên bản PyTorch.
In phiên bản TorchVision.
Set seed cho:
Python.
NumPy.
PyTorch.
CUDA nếu có.
Tạo generator cho DataLoader/split.
Tạo các thư mục output.

PyTorch lưu ý rằng kết quả không được đảm bảo giống tuyệt đối giữa các phiên bản, thiết bị hoặc nền tảng, nhưng seed và cấu hình deterministic có thể giảm nguồn biến động.

Section 3 — Device configuration

Notebook phải tự động nhận diện:

CUDA → NVIDIA GPU
MPS  → Apple Silicon GPU
CPU  → fallback

Ưu tiên:

if CUDA available:
    device = "cuda"
elif MPS available:
    device = "mps"
else:
    device = "cpu"

Vì đang sử dụng MacBook, nhánh MPS cần được hỗ trợ.

Notebook phải in:

Using device: mps

hoặc thiết bị thực tế tương ứng.

Section 4 — Configuration

Không rải hyperparameter ở nhiều cell. Tất cả đặt trong một cấu hình trung tâm:

CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "hidden_dims": [128],
    "dropout": 0.0,
}

Lợi ích:

Dễ chạy lại.
Dễ log.
Dễ so sánh experiments.
Tránh thay nhầm tham số ở nhiều nơi.
Section 5 — Class names

Khai báo đúng thứ tự 10 class:

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

Class mapping này cần được dùng thống nhất trong:

Visualization.
Prediction.
Confusion matrix.
Saved metadata.
Section 6 — Define transforms
Transform chính
ToImage
→ ToDtype(float32, scale=True)

Sau khi transform phải kiểm tra:

Shape : [1, 28, 28]
Dtype : torch.float32
Min   : 0.0
Max   : 1.0
Label : integer 0–9
Normalization

Để giữ notebook dễ hiểu, baseline có thể chỉ scale [0,1].

Một experiment nâng cao có thể thêm:

Normalize(mean_train, std_train)

Mean/std phải được tính từ training set, không lấy từ validation hoặc test.

Section 7 — Load dataset

Tạo hai dataset gốc:

full_train_dataset: train=True
test_dataset      : train=False

In ra:

Kích thước dataset.
Shape một sample.
Label một sample.
Class name.
Giá trị min/max pixel.

Dataset lưu sample và label, còn DataLoader tạo iterable theo batch, shuffle và hỗ trợ quá trình training.

Section 8 — Train/validation split

Tách:

Train      = 54.000
Validation =  6.000
Test       = 10.000

Yêu cầu:

Dùng seed cố định.
Không thay đổi official test set.
Kiểm tra tổng số sample.
Kiểm tra không overlap indices.
Vẽ bảng phân phối class train/validation/test.

Validation split phải được tạo trước khi thực nghiệm.

Section 9 — DataLoaders

Cấu hình:

Loader	Shuffle	Vai trò
Train	True	Thay đổi batch order mỗi epoch
Validation	False	Đánh giá ổn định
Test	False	Đánh giá cuối
Prediction loader	False	Hiển thị đúng thứ tự

Khởi đầu nên dùng:

batch_size = 64
num_workers = 0

num_workers=0 giúp notebook chạy ổn định trên macOS. Sau khi pipeline chạy đúng mới cân nhắc tăng số worker.

Sanity check một batch:

images.shape = [64, 1, 28, 28]
labels.shape = [64]
images.dtype = float32
labels.dtype = int64
Section 10 — Data exploration

EDA vừa đủ, không biến bài thành một project EDA lớn.

Các kết quả cần hiển thị
Bảng kích thước các tập.
Class distribution.
Grid 20 ảnh ngẫu nhiên.
Một ảnh của mỗi class.
Shape và range tensor.

Mục tiêu là xác nhận:

Label mapping đúng.
Ảnh không bị xoay hoặc biến dạng.
Tensor range đúng.
Train/validation không bị lỗi split.
Dữ liệu có đủ 10 class.
Section 11 — Build model class

Tạo class có thể tái sử dụng:

FashionMLP(
    hidden_dims,
    dropout
)

Không hard-code chỉ một architecture.

Logic:

Flatten
→ Linear
→ ReLU
→ optional Dropout
→ ...
→ Linear(last_hidden, 10)

Như vậy cùng một model class có thể tạo:

[128]
[256, 128]
[512, 256, 128]
Section 12 — Model sanity checks

Trước khi train toàn bộ:

Test 1 — Forward pass

Cho một batch qua model:

Input shape  : [64, 1, 28, 28]
Output shape : [64, 10]
Test 2 — Loss computation

Kiểm tra:

loss is finite
loss.requires_grad = True
Test 3 — Backward pass

Chạy một bước:

optimizer.zero_grad()
logits = model(images)
loss = loss_fn(logits, labels)
loss.backward()

Kiểm tra ít nhất một parameter có gradient.

Test 4 — Small-batch overfit

Lấy một vài batch nhỏ và train thử nhiều iteration.

Mục tiêu không phải báo cáo kết quả này, mà để phát hiện:

Label mapping sai.
Loss sai.
Model không cập nhật.
Learning rate lỗi.
Forward shape sai.
Gradient không chảy.

Chỉ sau khi các test này đạt mới train toàn dataset.

5. Training pipeline
5.1. Hàm train_one_epoch

Thứ tự mỗi batch:

1. model.train()
2. Move images và labels sang device
3. optimizer.zero_grad()
4. logits = model(images)
5. loss = criterion(logits, labels)
6. loss.backward()
7. optimizer.step()
8. Accumulate loss và correct predictions

PyTorch Autograd tự xây computational graph và tính gradient qua loss.backward(). Trong training loop, gradient cần được reset, sau đó backward và optimizer step để cập nhật parameters.

Cách tính epoch loss đúng

Không lấy trung bình đơn giản giữa các batch nếu batch cuối nhỏ hơn.

Nên cộng:

total_loss += loss.item() × batch_size_current

Sau đó:

epoch_loss = total_loss / number_of_samples
5.2. Hàm evaluate

Thứ tự:

1. model.eval()
2. torch.inference_mode()
3. Forward pass
4. Accumulate loss
5. Count correct predictions
6. Không backward
7. Không optimizer.step

model.eval() chuyển Dropout và BatchNorm sang evaluation behavior; tắt gradient trong evaluation giúp giảm computation và memory.

Hàm trả về:

{
    "loss": ...,
    "accuracy": ...,
    "predictions": ...,
    "targets": ...,
    "probabilities": ...
}
5.3. Hàm fit

Mỗi epoch:

Train one epoch
       ↓
Evaluate validation
       ↓
Append history
       ↓
Compare validation accuracy
       ↓
Save best checkpoint
       ↓
Print epoch summary

Log mỗi epoch:

Epoch	Train Loss	Train Acc	Val Loss	Val Acc	LR

Không dùng test loader trong hàm fit().

6. Kế hoạch thực nghiệm

Không đổi nhiều yếu tố cùng lúc. Thực nghiệm được chia theo từng phase.

Phase A — Baseline
ID	Architecture	Optimizer	LR	Batch	Dropout
E0	[128]	SGD	0.01	64	0.0

Mục tiêu:

Kiểm tra pipeline.
Có mốc so sánh.
Tạo loss curve đầu tiên.
Phase B — Learning-rate experiment

Giữ nguyên mọi yếu tố, chỉ đổi learning rate:

ID	Hidden dims	Optimizer	LR	Batch
E1	[128]	SGD	0.001	64
E0	[128]	SGD	0.01	64
E2	[128]	SGD	0.1	64

So sánh:

Convergence speed.
Final validation loss.
Final validation accuracy.
Stability của loss curve.

Chọn learning rate tốt nhất cho phase tiếp theo.

Phase C — Network architecture

Giữ optimizer, batch size và learning rate tốt nhất:

ID	Architecture	Thay đổi
E-best-LR	[128]	Baseline architecture
E3	[256, 128]	Thêm capacity

Câu hỏi nghiên cứu:

Thêm một hidden layer có cải thiện validation accuracy đủ rõ so với chi phí tăng thêm hay không?

Ghi thêm:

Số parameters.
Thời gian train.
Validation accuracy delta.
Phase D — Regularization

Giữ cấu hình E3, chỉ thêm:

Dropout = 0.2
ID	Architecture	Dropout
E3	[256,128]	0.0
E4	[256,128]	0.2

Câu hỏi:

Dropout có giảm khoảng cách train–validation và cải thiện generalization không?

Phase E — Optimizer comparison, phần mở rộng

So sánh:

ID	Optimizer policy
E4	SGD với learning rate tốt nhất
E5	Adam với lr=0.001

Đây được xem là thay đổi optimizer policy, vì Adam và SGD cần learning rate scale khác nhau.

Chapter 4 cũng nêu optimizer, learning rate, batch size và số epoch là các thành phần chính của training algorithm; SGD, Adam và RMSprop là các lựa chọn phổ biến trong PyTorch.

Phase F — Repeated seeds cho hai model tốt nhất

Nếu thời gian chạy cho phép, lấy hai cấu hình tốt nhất và chạy lại với:

seed = 42
seed = 123
seed = 2026

Báo cáo:

Validation accuracy = mean ± standard deviation

Điều này giúp tránh kết luận dựa trên một lần khởi tạo weights ngẫu nhiên.

7. Bảng log experiment

Notebook nên tự tạo DataFrame:

Experiment	Architecture	Optimizer	LR	Batch	Dropout	Best Epoch	Best Val Loss	Best Val Acc	Train Time

Mỗi experiment phải lưu:

{
    "experiment_id": ...,
    "hypothesis": ...,
    "config": ...,
    "history": ...,
    "best_epoch": ...,
    "best_val_loss": ...,
    "best_val_accuracy": ...,
    "parameter_count": ...,
    "training_time": ...
}

Sau đó export:

results/experiment_results.csv
8. Visualization plan
8.1. Data sample grid

Hiển thị 20 ảnh:

Image
Actual label
8.2. Class distribution

Bar chart cho:

Train.
Validation.
Test.
8.3. Loss curve

Đồ thị bắt buộc:

X-axis: Epoch
Y-axis: Cross-Entropy Loss

Line 1: Train loss
Line 2: Validation loss

Phân tích:

Cả hai cùng giảm: model đang học.
Train giảm, validation tăng: overfitting.
Cả hai cao: underfitting.
Loss dao động: learning rate có thể quá cao.
8.4. Accuracy curve
Train accuracy
Validation accuracy

Không bắt buộc trong đề nhưng nên có vì hỗ trợ giải thích loss.

8.5. Experiment comparison

Bar chart:

X-axis: Experiment ID
Y-axis: Best validation accuracy

Có thể kèm bảng số parameters và thời gian train.

8.6. Predicted vs actual images

Hiển thị khoảng 12–16 ảnh test:

Actual: Shirt
Predicted: Coat
Confidence: 71.4%

Quy ước:

Dự đoán đúng: tiêu đề màu xanh.
Dự đoán sai: tiêu đề màu đỏ.

Nên có hai hình:

Một grid gồm cả đúng và sai.
Một grid chỉ chứa các dự đoán sai có confidence cao.

Hình thứ hai có giá trị error analysis tốt hơn hình chọn ngẫu nhiên.

8.7. Confusion matrix

Không bắt buộc nhưng nên có.

Mục tiêu:

Xem class nào dễ nhầm.
Phân tích các cặp như:
Shirt và T-shirt/top.
Pullover và Coat.
Sandal và Sneaker.

Chỉ kết luận những cặp nhầm thực tế sau khi chạy model, không ghi trước số liệu.

9. Model selection và test evaluation
9.1. Quy tắc chọn model

Chọn model theo thứ tự:

Validation accuracy cao.
Validation loss thấp.
Khoảng cách train–validation hợp lý.
Kết quả ổn định.
Nếu hai model gần nhau, ưu tiên model đơn giản hơn.

Không chọn theo test accuracy.

9.2. Final test

Sau khi khóa cấu hình:

Load best validation checkpoint
        ↓
Run test loader một lần
        ↓
Report test loss và test accuracy
        ↓
Generate confusion matrix
        ↓
Generate prediction images

Báo cáo:

Test accuracy.
Test cross-entropy loss.
Per-class accuracy.
Confusion matrix.
Một số lỗi đại diện.

Không đặt trước con số accuracy kỳ vọng; chỉ báo cáo kết quả chạy thực tế.

10. Save và load model
10.1. Save best model

Không chỉ save last epoch.

Save checkpoint tốt nhất theo validation accuracy:

checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": config,
    "class_names": class_names,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
}

File:

models/best_fashion_mnist_mlp.pth

PyTorch khuyến nghị lưu state_dict; khi load cần khởi tạo lại cùng model architecture rồi gọi load_state_dict().

10.2. Load verification

Không chỉ viết code load mà phải chứng minh model load thành công:

1. Tạo model object mới
2. Load checkpoint
3. model.eval()
4. Chạy cùng một batch
5. So sánh logits/predictions trước và sau load
6. Đánh giá lại accuracy

Acceptance check:

Predictions before load == Predictions after load

Trước inference phải gọi model.eval(), đặc biệt khi model có Dropout hoặc BatchNorm.

11. Brief report đặt cuối notebook
11.1. Objective

Một đoạn ngắn:

This exercise implements a complete PyTorch workflow for classifying FashionMNIST images using a multilayer perceptron.

11.2. Dataset

Nêu:

28×28 grayscale images.
10 classes.
Train/validation/test split.
Transform đã áp dụng.
11.3. Model

Nêu:

Flatten.
Hidden layers.
ReLU.
Dropout nếu có.
10 output logits.
11.4. Training setup

Nêu:

Loss.
Optimizer.
Learning rate.
Batch size.
Epochs.
Device.
Seed.
11.5. Experiments

Bảng toàn bộ experiments và giả thuyết.

11.6. Results

Nêu đúng số liệu chạy được:

Best validation accuracy.
Test accuracy.
Test loss.
Best epoch.
Training time.
11.7. Error analysis

Trả lời:

Những class nào bị nhầm nhiều nhất?
Model sai trong loại ảnh nào?
Sai số có liên quan đến hình dáng tương tự không?
Model có dự đoán sai nhưng confidence cao không?
11.8. Conclusion

Kết luận:

Model tốt nhất là cấu hình nào?
Nó cải thiện bao nhiêu so với baseline?
Thay đổi nào hiệu quả?
Thay đổi nào không hiệu quả?
Hướng tiếp theo có thể là CNN hoặc data augmentation.
12. Mapping từ đề bài sang notebook
Yêu cầu đề	Notebook section
Load FashionMNIST	Sections 6–8
Apply transforms	Section 6
Build neural network	Sections 11–12
Forward pass	train_one_epoch()
Compute loss	CrossEntropyLoss
Backward pass	loss.backward()
Optimize	optimizer.step()
Evaluate accuracy	evaluate() và final test
Save model	Save best checkpoint
Load model	Fresh model reload verification
Experiment network	Architecture phase
Experiment hyperparameters	Learning-rate/dropout/optimizer phases
Visualize loss	Loss and accuracy curves
Predicted vs actual	Prediction grid
Brief report	Final Markdown sections
Python code	Một notebook chạy từ đầu đến cuối

Chuỗi này cũng khớp với cấu trúc tutorial chính thức của PyTorch: Tensors, Datasets/DataLoaders, Transforms, Build Model, Autograd, Optimization và Save/Load.

13. Rủi ro và cách kiểm soát
Rủi ro	Cách xử lý
Test leakage	Test loader không xuất hiện trong training/experiment runner
Shape mismatch	Sanity check [B,1,28,28] → [B,10]
Softmax dùng sai	Model trả logits; softmax chỉ dùng khi hiển thị
Label one-hot gây lỗi	Giữ label class index int64
Gradient tích lũy	optimizer.zero_grad() mỗi batch
Quên train/eval mode	Gọi model.train() và model.eval() đúng chỗ
MPS không hỗ trợ một operation	Fallback CPU và ghi rõ device
Notebook chạy lại cho kết quả khác	Seed và log environment
Experiment không công bằng	Giữ nguyên các yếu tố ngoài biến đang kiểm tra
Save nhầm last model	Save best validation checkpoint
Load đúng code nhưng sai model	Verify predictions trước/sau load
Notebook quá dài	Dùng helper functions và config dictionary
Kết quả bị bịa/ghi thủ công	DataFrame lấy trực tiếp từ training history
14. Definition of Done

Notebook chỉ được xem là hoàn thành khi đạt toàn bộ điều kiện:

Chạy từ cell đầu đến cell cuối không lỗi.
Tải FashionMNIST tự động.
In đúng dataset sizes và tensor shapes.
Có train/validation/test separation.
Có MLP subclass từ nn.Module.
Có forward, loss, backward và optimizer step.
Có train loss và validation loss theo epoch.
Có ít nhất một baseline và ba experiment có kiểm soát.
Có bảng so sánh experiment.
Có loss curve.
Có predicted vs actual images.
Có error-analysis images.
Có final test accuracy.
Có file .pth.
Load vào model object mới thành công.
Predictions trước và sau load nhất quán.
Có brief report bằng Markdown.
Có references tới PyTorch tutorials.
Không dùng test set để tuning.
Không ghi bất kỳ metric nào chưa thực sự chạy.
Kết luận plan

Pipeline thực thi sẽ là:

Define problem
    ↓
Environment + reproducibility
    ↓
Load FashionMNIST
    ↓
Split train/validation
    ↓
Transform + DataLoader
    ↓
EDA và sanity checks
    ↓
Build MLP baseline
    ↓
Verify forward/backward
    ↓
Train baseline
    ↓
Controlled experiments
    ↓
Compare validation results
    ↓
Select and lock best configuration
    ↓
Evaluate official test once
    ↓
Visualize predictions and errors
    ↓
Save model
    ↓
Load and verify model
    ↓
Write brief report and conclusion

Plan này giữ đúng phạm vi bài tập, đồng thời áp dụng đầy đủ các nguyên tắc Deep Learning đã thống nhất: baseline trước, test set được niêm phong, thực nghiệm chỉ thay đổi một yếu tố, log đầy đủ và không báo cáo kết quả chưa chạy. Sếp xác nhận plan này, bước tiếp theo tôi sẽ triển khai notebook theo đúng cấu trúc trên.