from typing import Dict, Union

import numpy as np
from puad.dataset import NormalDataset
from puad.efficientad.inference import EfficientADInference
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor
import torchvision


class PUAD:
    def __init__(self, feature_extractor: str = "student") -> None:
        self.feature_extractor: str = feature_extractor  # "student" or "teacher"

    def load_efficient_ad(self, efficient_ad_inference: EfficientADInference) -> None:
        self.efficient_ad_inference = efficient_ad_inference

    def train(self, dataset: Union[NormalDataset, torch.utils.data.Subset]) -> None:
        feature_vectors = []
        for img in dataset:
            _, teacher_output, student_output = self.efficient_ad_inference.run(img, is_for_puad=True)
            if self.feature_extractor == "student":
                feature_vector = torch.mean(student_output, dim=(0, 2, 3)).detach().cpu().numpy()
            else:
                feature_vector = torch.mean(teacher_output, dim=(0, 2, 3)).detach().cpu().numpy()
            feature_vectors.append(feature_vector)
        feature_vectors = np.array(feature_vectors)
        self.feature_vectors_mean = np.mean(feature_vectors, axis=0)
        cov = LedoitWolf().fit(feature_vectors).covariance_
        self.feature_vectors_covinv = np.linalg.pinv(cov)

    def mahalanobis(self, img: Tensor) -> np.float32:
        _, teacher_output, student_output = self.efficient_ad_inference.run(img, is_for_puad=True)
        if self.feature_extractor == "student":
            feature_vector = torch.mean(student_output, dim=(0, 2, 3)).detach().cpu().numpy()
        else:
            feature_vector = torch.mean(teacher_output, dim=(0, 2, 3)).detach().cpu().numpy()
        centered_feature_vector = feature_vector - self.feature_vectors_mean
        mahalanobis_distance = np.sqrt(
            max(
                0,
                np.dot(
                    np.dot(centered_feature_vector, self.feature_vectors_covinv),
                    centered_feature_vector,
                ),
            )
        )
        return mahalanobis_distance

    def valid(self, dataset: Union[NormalDataset, torch.utils.data.Subset]) -> None:
        efficient_ad_scores = []
        mahalanobis_distances = []
        for img in dataset:
            efficient_ad_scores.append(self.efficient_ad_inference.run(img))
            mahalanobis_distances.append(self.mahalanobis(img))
        self.efficient_ad_mean = np.mean(efficient_ad_scores)
        self.efficient_ad_std = np.std(efficient_ad_scores)
        self.mahalanobis_mean = np.mean(mahalanobis_distances)
        self.mahalanobis_std = np.std(mahalanobis_distances)

    def test(self, img: Tensor) -> np.float32:
        mahalanobis_distance = self.mahalanobis(img)
        efficient_ad_score = self.efficient_ad_inference.run(img)
        normalized_picturable_anomaly_score = (efficient_ad_score - self.efficient_ad_mean) / self.efficient_ad_std
        normalized_unpicturable_anomaly_score = (mahalanobis_distance - self.mahalanobis_mean) / self.mahalanobis_std
        return normalized_picturable_anomaly_score + normalized_unpicturable_anomaly_score

    def auroc(self, dataset: torchvision.datasets.ImageFolder) -> np.float64:
        anomaly_scores = []
        targets = []
        for img, label in dataset:
            anomaly_scores.append(self.test(img))
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
            pred = self.test(img)
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
