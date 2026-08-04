import torch
import torch.nn as nn

def get_loss_normalizer(criterion: nn.Module, labels: torch.Tensor) -> float:
    """
    Tính toán đúng mẫu số (normalizer) cho một batch.
    Khắc phục lỗi epoch aggregation của V1 (nhân với batch_size một cách sai lệch khi có weights).
    """
    weight = getattr(criterion, "weight", None)
    if weight is None:
        # Nếu không dùng class weights, hàm CE mặc định tính trung bình của batch (unweighted mean)
        return float(labels.numel())
    else:
        # Nếu dùng class weights, hàm CE trả về weighted mean.
        # Mẫu số đúng để khôi phục lại tổng loss của batch là tổng trọng số của các phần tử trong batch.
        return float(weight[labels].sum().item())

class EpochLossAggregator:
    """
    Helper class chuẩn hóa việc tính toán Epoch Loss cho Train/Val/Test.
    """
    def __init__(self):
        self.running_loss = 0.0
        self.total_normalizer = 0.0
        
    def update(self, batch_loss_item: float, criterion: nn.Module, labels: torch.Tensor):
        normalizer = get_loss_normalizer(criterion, labels)
        # batch_loss_item là giá trị trung bình (weighted hoặc unweighted) do PyTorch trả về.
        # Ta cần nhân ngược với normalizer để tính lại tổng loss tuyệt đối của batch.
        self.running_loss += batch_loss_item * normalizer
        self.total_normalizer += normalizer
        
    def get_epoch_loss(self) -> float:
        if self.total_normalizer == 0:
            return 0.0
        return self.running_loss / self.total_normalizer
