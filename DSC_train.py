import os
import math
import csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    confusion_matrix
)

class SamplingScheduler:
    """
    Dynamic curriculum sampling scheduler.
    - D_train: 초기 클래스 비율 [#non_seizure / #seizure, 1]
    - SF(n):   epoch마다 1 -> 0으로 감소하는 cos 스케줄
    - D_target(n) = D_train ** SF(n)
      이후 batch_size에 맞게 정규화해서 각 클래스 샘플 수를 만듬.
    """
    def __init__(self, num_epochs: int, batch_size: int, class_counts: np.ndarray):
        """
        :param num_epochs: 총 학습 epoch 수 (N)
        :param batch_size: 한 iteration에서 쓸 총 배치 사이즈
        :param class_counts: 각 클래스(0: non-seizure, 1: seizure) 샘플 수
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # D_train = [#C_NS / #C_S, 1]
        # class 0: non-seizure, class 1: seizure
        non_seizure = class_counts[0]
        seizure = class_counts[1]
        self.D_train = np.array([non_seizure / seizure, 1.0], dtype=np.float32)

    def scheduler_function(self, epoch: int) -> float:
        """
        SF(n) = cos( n/N * pi/2 )
        epoch은 0-based라 n = epoch+1 으로 사용.
        """
        n = epoch + 1
        N = self.num_epochs
        return math.cos((n / N) * (math.pi / 2.0))

    def get_class_batch_sizes(self, epoch: int) -> Tuple[int, int]:
        """
        현재 epoch에서 각 클래스별 배치 크기 반환.
        :return: (batch_non_seizure, batch_seizure)
        """
        g_l = self.scheduler_function(epoch)  # SF(n)
        # D_target = D_train ** g_l
        D_target = self.D_train ** g_l
        # 배치 사이즈 합이 batch_size가 되도록 정규화
        D_target = D_target / D_target.sum() * self.batch_size

        batch_non_seizure = max(1, int(D_target[0]))
        batch_seizure = max(1, int(D_target[1]))
        # 합이 정확히 batch_size가 되도록 보정
        diff = self.batch_size - (batch_non_seizure + batch_seizure)
        if diff > 0:
            batch_non_seizure += diff
        elif diff < 0:
            batch_non_seizure = max(1, batch_non_seizure + diff)

        return batch_non_seizure, batch_seizure


# ============================================================
# 2. DANN 기반 EEGNet 모델
#    - EEGNet encoder (feature) + Label Classifier + Domain Classifier
#    - Gradient Reversal Layer (GRL) 포함
# ============================================================

class GradReverseFn(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2016)
    forward: identity
    backward: gradient * (-lambda_)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradReverse(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradReverseFn.apply(x, self.lambda_)


class EEGNetEncoder(nn.Module):
    """
    아주 간단하게 구현한 EEGNet-like encoder.
    (Lawhern et al., 2018 구조를 축약한 버전)
    논문에서는 EEGNet [20]을 encoder로 사용. :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, n_chans: int, n_times: int, latent_dim: int = 64):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (n_chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        # feature dimension 계산을 위해 dummy forward
        with torch.no_grad():
            x_dummy = torch.zeros(1, 1, n_chans, n_times)
            feat = self._forward_features(x_dummy)
            feat_dim = feat.shape[1]

        self.fc = nn.Linear(feat_dim, latent_dim)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        x: (B, C, T)
        """
        # EEGNet은 (B, 1, C, T)를 입력으로 사용
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = self.fc(x)
        return x  # (B, latent_dim)


class DANN_EEGNet(nn.Module):
    """
    EEGNet encoder + Task(Class) classifier + Domain classifier
    - Task: Seizure vs Non-seizure
    - Domain: Patient ID
    """
    def __init__(self, n_chans: int, n_times: int, n_classes: int, n_domains: int,
                 latent_dim: int = 64, lambda_domain: float = 1.0):
        super().__init__()
        self.encoder = EEGNetEncoder(n_chans=n_chans, n_times=n_times, latent_dim=latent_dim)

        self.label_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        self.grl = GradReverse(lambda_=lambda_domain)

        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains)
        )

    def forward(self, x):
        """
        :param x: (B, C, T)
        :return: class_logits, domain_logits
        """
        feat = self.encoder(x)  # (B, latent_dim)
        class_logits = self.label_classifier(feat)

        rev_feat = self.grl(feat)
        domain_logits = self.domain_classifier(rev_feat)

        return class_logits, domain_logits


# ============================================================
# 3. CHB-MIT 전처리 결과 로딩 유틸
#    - 가정: 환자별로 seizure / non-seizure 세그먼트가 .npy로 저장되어 있음
#    - 각 세그먼트: shape = (num_segments, n_chans, n_times)
# ============================================================

def load_patient_segments(root: str, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    예시용 로더 (본인 환경에 맞게 파일 이름/구조 수정 필요)
    - root/
        ├─ seizure/{patient_id}_seizure.npy
        └─ nonseizure/{patient_id}_nonseizure.npy

    return:
        X: (N, C, T), y: (N,)  (0=non-seizure, 1=seizure)
    """
    seizure_path = os.path.join(root, "seizure", f"{patient_id}_seizure.npy")
    non_path = os.path.join(root, "nonseizure", f"{patient_id}_nonseizure.npy")

    seizure_data = np.load(seizure_path)   # label 1
    non_data = np.load(non_path)          # label 0

    X = np.concatenate([non_data, seizure_data], axis=0)
    y = np.concatenate([
        np.zeros(len(non_data), dtype=np.int64),
        np.ones(len(seizure_data), dtype=np.int64)
    ], axis=0)

    return X, y


def build_chbmit_dataset(root: str,
                         all_patients: List[str],
                         target_patient: str):
    """
    Leave-one-patient-out 셋업:
      - target_patient: 테스트용 환자
      - 나머지 환자: 학습용 (domain label = patient index)
    """
    train_X_list, train_y_list, train_domain_list = [], [], []
    test_X, test_y = None, None

    for domain_idx, pid in enumerate(all_patients):
        X, y = load_patient_segments(root, pid)

        domain_labels = np.full_like(y, fill_value=domain_idx)

        if pid == target_patient:
            test_X, test_y = X, y
        else:
            train_X_list.append(X)
            train_y_list.append(y)
            train_domain_list.append(domain_labels)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    train_domain = np.concatenate(train_domain_list, axis=0)

    return train_X, train_y, train_domain, test_X, test_y


# ============================================================
# 4. Metric / Train / Evaluation
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = recall_score(y_true, y_pred)           # sensitivity
    specificity = tn / (tn + fp + 1e-8)
    precision = precision_score(y_true, y_pred)
    bacc = (recall + specificity) / 2.0
    gmean = math.sqrt(recall * specificity)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "bacc": bacc,
        "gmean": gmean,
        "auc": auc
    }


def train_one_epoch(model, train_dataset, train_y_np, num_epochs, epoch,
                    batch_size, device,
                    lambda_domain=1.0):
    """
    Sampling Scheduler를 epoch마다 적용해서
    seizure / non-seizure 비율을 점진적으로 맞춰가는 학습.
    """
    model.train()
    criterion_task = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # 클래스 분포로부터 scheduler 생성 (epoch마다 새로 만들어도 되지만, 여기선 간단히)
    class_counts = np.bincount(train_y_np, minlength=2)
    scheduler = SamplingScheduler(num_epochs=num_epochs,
                                  batch_size=batch_size,
                                  class_counts=class_counts)

    batch_non_seizure, batch_seizure = scheduler.get_class_batch_sizes(epoch)

    # 클래스 index
    idx_non = np.where(train_y_np == 0)[0]
    idx_seiz = np.where(train_y_np == 1)[0]

    sampler_non = SubsetRandomSampler(idx_non)
    sampler_seiz = SubsetRandomSampler(idx_seiz)

    loader_non = DataLoader(train_dataset, batch_size=batch_non_seizure,
                            sampler=sampler_non, drop_last=True)
    loader_seiz = DataLoader(train_dataset, batch_size=batch_seizure,
                             sampler=sampler_seiz, drop_last=True)

    optimizer = model.optimizer  # 모델에 optimizer를 미리 붙여두었다고 가정

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # 두 클래스 로더를 동시에 순회
    for (x_non, y_non, d_non), (x_seiz, y_seiz, d_seiz) in zip(loader_non, loader_seiz):
        x = torch.cat([x_non, x_seiz], dim=0).to(device)
        y = torch.cat([y_non, y_seiz], dim=0).to(device)
        d = torch.cat([d_non, d_seiz], dim=0).to(device)

        class_logits, domain_logits = model(x)

        loss_task = criterion_task(class_logits, y)
        loss_domain = criterion_domain(domain_logits, d)
        loss = loss_task - lambda_domain * loss_domain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = class_logits.argmax(dim=1)
        running_loss += loss.item() * x.size(0)
        running_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device):
    model.eval()
    criterion_task = nn.CrossEntropyLoss()

    all_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            class_logits, _ = model(x)
            loss = criterion_task(class_logits, y)
            all_loss += loss.item() * x.size(0)
            total += x.size(0)

            probs = torch.softmax(class_logits, dim=1)[:, 1]   # seizure class prob
            preds = class_logits.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = all_loss / total
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return avg_loss, metrics


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    # ----- 설정 -----
    gpu_num = 0
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print("Current cuda device:", torch.cuda.current_device())

    data_root = "/path/to/your/preprocessed_CHBMIT"  # 수정 필요
    results_dir = "./results/chbmit"
    os.makedirs(results_dir, exist_ok=True)

    num_epochs = 300
    batch_size = 1024
    lambda_domain = 1.0
    learning_rate = 1e-4
    weight_decay = 1e-3

    # CHB-MIT 환자 ID 예시 (실제와 다를 수 있음, 환경에 맞게 수정)
    all_patients = [f"chb{idx:02d}" for idx in range(1, 24)]

    # 예시: chb01을 test로 두고 나머지로 학습
    target_patient = "chb01"

    # ----- 데이터 로딩 -----
    train_X, train_y, train_domain, test_X, test_y = build_chbmit_dataset(
        data_root, all_patients, target_patient
    )

    print("Train:", train_X.shape, train_y.shape, train_domain.shape)
    print("Test :", test_X.shape, test_y.shape)

    n_chans = train_X.shape[1]
    n_times = train_X.shape[2]
    n_classes = 2
    n_domains = len(all_patients)

    # TensorDataset (float32, long)
    train_dataset = TensorDataset(
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_y).long(),
        torch.from_numpy(train_domain).long()
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_X).float(),
        torch.from_numpy(test_y).long()
    )

    test_loader = DataLoader(test_dataset, batch_size=1024,
                             shuffle=False, drop_last=False)

    # ----- 모델 & 옵티마이저 -----
    model = DANN_EEGNet(
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        n_domains=n_domains,
        latent_dim=64,
        lambda_domain=lambda_domain
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay)
    # train 함수에서 쓰기 위해 모델에 붙여둠
    model.optimizer = optimizer

    # ----- 학습 루프 -----
    best_bacc = 0.0
    csv_path = os.path.join(results_dir, f"DANN_EEGNet_scheduler_{target_patient}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "TrainLoss", "TrainAcc",
            "TestLoss", "bACC", "Recall", "Specificity",
            "Precision", "Gmean", "AUC"
        ])

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_dataset=train_dataset,
                train_y_np=train_y,
                num_epochs=num_epochs,
                epoch=epoch,
                batch_size=batch_size,
                device=device,
                lambda_domain=lambda_domain
            )

            test_loss, metrics = evaluate(model, test_loader, device)

            print(
                f"[Epoch {epoch+1:03d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | bACC: {metrics['bacc']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f} | "
                f"AUC: {metrics['auc']:.4f}"
            )

            writer.writerow([
                epoch + 1,
                train_loss, train_acc,
                test_loss,
                metrics["bacc"], metrics["recall"], metrics["specificity"],
                metrics["precision"], metrics["gmean"], metrics["auc"]
            ])

            # 논문에 맞게 bACC 기준으로 best 모델 저장 같은 것도 가능
            if metrics["bacc"] > best_bacc:
                best_bacc = metrics["bacc"]
                torch.save(model.state_dict(),
                           os.path.join(results_dir, f"best_model_{target_patient}.pt"))
