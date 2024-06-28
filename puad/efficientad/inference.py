import os
from typing import Dict, Tuple, TypedDict, Union

import numpy as np
from puad.networks import AutoEncoder, PDN_M, PDN_S
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Normalize


class QuantileDict(TypedDict):
    q_a_student: str
    q_b_student: str
    q_a_autoencoder: str
    q_b_autoencoder: str


class EfficientADInference:
    def __init__(
        self,
        teacher: Union[PDN_S, PDN_M],
        student: Union[PDN_S, PDN_M],
        autoencoder: AutoEncoder,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        quantile: QuantileDict,
        img_size: int = 256,
        device: str = "cuda",
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.autoencoder = autoencoder.to(device)

        self.normalization = Normalize(mu, sigma)
        self.q_a_student = quantile["q_a_student"].to(device)
        self.q_b_student = quantile["q_b_student"].to(device)
        self.q_a_autoencoder = quantile["q_a_autoencoder"].to(device)
        self.q_b_autoencoder = quantile["q_b_autoencoder"].to(device)
        self.img_size = img_size
        self.device = device

        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()

    def run(
        self, img: torch.Tensor, is_for_puad: bool = False
    ) -> Union[np.float64, Tuple[np.float64, torch.Tensor, torch.Tensor]]:
        img = img.to(self.device).unsqueeze(0)
        with torch.no_grad():
            # This implementation followed `Algorithm 2` in the original EfficentAD paper:
            # https://arxiv.org/abs/2303.14535
            teacher_output = self.teacher(img)
            student_output = self.student(img)
            autoencoder_output = self.autoencoder(img)
            normalized_teacher_output = self.normalization(teacher_output)
            student_output_st = student_output[:, : student_output.shape[1] // 2, :, :]
            student_output_ae = student_output[:, student_output.shape[1] // 2 :, :, :]
            diff_teacher_student = (normalized_teacher_output - student_output_st) ** 2
            diff_student_ae = (autoencoder_output - student_output_ae) ** 2
            student_map = torch.mean(diff_teacher_student, dim=1, keepdim=True)
            ae_map = torch.mean(diff_student_ae, dim=1, keepdim=True)
            resized_student_map = F.interpolate(student_map, size=(self.img_size, self.img_size), mode="bilinear")
            resized_ae_map = F.interpolate(ae_map, size=(self.img_size, self.img_size), mode="bilinear")
            resized_student_map = resized_student_map.squeeze()
            resized_ae_map = resized_ae_map.squeeze()
            resized_student_map = 0.1 * (resized_student_map - self.q_a_student) / (self.q_b_student - self.q_a_student)
            resized_ae_map = (
                0.1 * (resized_ae_map - self.q_a_autoencoder) / (self.q_b_autoencoder - self.q_a_autoencoder)
            )
            anomaly_map = 0.5 * resized_student_map + 0.5 * resized_ae_map
            anomaly_score = torch.max(anomaly_map).cpu().numpy()
            if is_for_puad:
                return anomaly_score, normalized_teacher_output, student_output_st
            return anomaly_score

    def auroc(self, dataset: torchvision.datasets.ImageFolder) -> np.float64:
        anomaly_scores = []
        targets = []
        for img, label in dataset:
            img = img.to(self.device)
            anomaly_scores.append(self.run(img))
            targets.append(0 if label == dataset.class_to_idx["good"] else 1)

        return roc_auc_score(targets, anomaly_scores)

    def auroc_for_anomalies(self, dataset: torchvision.datasets.ImageFolder) -> Dict[str, np.float64]:
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

        good_scores = []
        good_targets = []
        anomaly_scores = {}
        anomaly_targets = {}
        for img, label in dataset:
            class_name = idx_to_class[label]
            img = img.to(self.device)
            pred = self.run(img)
            if class_name == "good":
                good_scores.append(pred)
                good_targets.append(0)
            else:
                if class_name not in anomaly_scores:
                    anomaly_scores[class_name] = []
                    anomaly_targets[class_name] = []
                anomaly_scores[class_name].append(pred)
                anomaly_targets[class_name].append(1)

        scores = good_scores + sum(anomaly_scores.values(), [])
        targets = good_targets + sum(anomaly_targets.values(), [])
        auroc_score = roc_auc_score(targets, scores)

        auroc_scores = {}
        for anomaly_class in anomaly_scores.keys():
            scores = good_scores + anomaly_scores[anomaly_class]
            targets = good_targets + anomaly_targets[anomaly_class]
            auroc_scores[anomaly_class] = roc_auc_score(targets, scores)

        return auroc_score, auroc_scores


def load_efficient_ad(
    model_dir_path: str,
    size: str,
    dataset_name: str,
    category: str,
    out_channels: int = 384,
    img_size: int = 256,
    device: str = "cuda",
) -> EfficientADInference:
    efficient_ad_model_path = os.path.join(model_dir_path, f"{size}_size", dataset_name, category)

    teacher = (PDN_S(out_channels=out_channels) if size == "s" else PDN_M(out_channels=out_channels)).to(device)
    student = (PDN_S(out_channels=out_channels * 2) if size == "s" else PDN_M(out_channels=out_channels * 2)).to(device)
    autoencoder = (AutoEncoder(out_channels=out_channels, img_size=img_size, padding=False)).to(device)

    teacher.load_state_dict(torch.load(os.path.join(model_dir_path, f"{size}_size", "teacher", "teacher.pt")))
    student.load_state_dict(torch.load(os.path.join(efficient_ad_model_path, "student.pt")))
    autoencoder.load_state_dict(torch.load(os.path.join(efficient_ad_model_path, "autoencoder.pt")))

    quantile = torch.load(os.path.join(efficient_ad_model_path, "quantile.pt"))

    return EfficientADInference(
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        mu=torch.load(os.path.join(efficient_ad_model_path, "mu.pt")),
        sigma=torch.load(os.path.join(efficient_ad_model_path, "sigma.pt")),
        quantile=quantile,
        img_size=img_size,
        device=device,
    )
