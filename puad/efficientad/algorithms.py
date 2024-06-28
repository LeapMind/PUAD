import os
from typing import Tuple, Union

from puad.common import build_imagenet_normalization
from puad.dataset import NormalDataset, RandomAugment, split_dataset
from puad.efficientad.inference import EfficientADInference, QuantileDict
from puad.efficientad.pretrained_feature_extractor import PretrainedFeatureExtractor
from puad.networks import AutoEncoder, PDN_M, PDN_S
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm


def train_student_and_autoencoder(
    model_dir: str,
    dataset_path: str,
    imagenet_path: str,
    teacher_model_path: str,
    pdn_size: str = "s",
    out_channels: int = 384,
    img_size: int = 256,
    device: str = "cuda",
) -> Tuple[Union[PDN_S, PDN_M], Union[PDN_S, PDN_M], AutoEncoder, torch.Tensor, torch.Tensor, QuantileDict,]:
    """
    Training student network and autoencoder for EfficientAD.

    This implementation followed `Algorithm 1` in the original EfficentAD paper:
    https://arxiv.org/abs/2303.14535
    """
    if not (pdn_size == "s" or pdn_size == "m"):
        raise ValueError("pdn_size must be `s` or `m`.")

    if pdn_size == "s":
        teacher = PDN_S(out_channels=out_channels, padding=False).to(device)
    else:
        teacher = PDN_M(out_channels=out_channels, padding=False).to(device)
    teacher.load_state_dict(
        torch.load(
            teacher_model_path,
            map_location=device,
        )
    )
    teacher.eval()

    if pdn_size == "s":
        student = PDN_S(out_channels=out_channels * 2, padding=False).to(device)
    else:
        student = PDN_M(out_channels=out_channels * 2, padding=False).to(device)
    student.train()

    autoencoder = AutoEncoder(out_channels=out_channels, img_size=img_size, padding=False).to(device)
    autoencoder.train()

    random_augment = RandomAugment()
    base_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    normalize = build_imagenet_normalization()

    def train_transform(img):
        return (
            normalize(base_transform(img)),
            normalize(random_augment(base_transform(img))),
        )

    def valid_transform(img):
        return normalize(base_transform(img))

    train_img_dir = os.path.join(dataset_path, "train", "good")
    valid_img_dir = os.path.join(dataset_path, "validation", "good")
    if os.path.exists(valid_img_dir):
        train_dataset = NormalDataset(train_img_dir, train_transform)
        valid_dataset = NormalDataset(valid_img_dir, valid_transform)
    else:
        train_dataset, valid_dataset = split_dataset(
            train_img_dir, split_ratio=0.15, transform_1=train_transform, transform_2=valid_transform
        )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    imagenet_dataset = torchvision.datasets.ImageFolder(
        root=imagenet_path,
        transform=transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomGrayscale(p=0.3),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
                build_imagenet_normalization(),
            ]
        ),
    )
    imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        teacher_outputs = []
        for i in tqdm(range(len(train_dataset)), desc="Calculating mean and std for train dataset"):
            img_train, _ = train_dataset[i]
            img_train = img_train.to(device)
            teacher_output = teacher(img_train).squeeze()
            teacher_outputs.append(teacher_output)
        stacked_teacher_outputs = torch.stack(teacher_outputs, dim=0)
        mu = stacked_teacher_outputs.mean(dim=(0, 2, 3))
        sigma = stacked_teacher_outputs.std(dim=(0, 2, 3))

    plist = nn.ParameterList()
    plist.extend(autoencoder.parameters())
    plist.extend(student.parameters())
    optimizer = Adam(plist, lr=1e-4, weight_decay=1e-5)

    n_training_loop = 70000

    teacher_output_normalization = transforms.Normalize(mu, sigma)

    with tqdm(range(n_training_loop), "Training student network and autoencoder") as pbar:
        for it in pbar:
            img_train, img_aug = next(iter(train_dataloader))
            img_train = img_train.to(device)
            img_aug = img_aug.to(device)
            with torch.no_grad():
                teacher_output = teacher(img_train)
                normalized_teacher_output = teacher_output_normalization(teacher_output)
            student_output = student(img_train)
            student_output_st = student_output[:, :out_channels, :, :]
            diff_teacher_student = (normalized_teacher_output - student_output_st) ** 2
            threshold = torch.quantile(input=diff_teacher_student, q=0.999)
            loss_hard = torch.mean(diff_teacher_student[torch.where(diff_teacher_student >= threshold)])
            imagenet_img, _ = next(iter(imagenet_dataloader))
            imagenet_img = imagenet_img.to(device)
            loss_student = loss_hard + torch.mean(student(imagenet_img)[:, :out_channels, :, :] ** 2)
            autoencoder_output = autoencoder(img_aug)
            with torch.no_grad():
                teacher_output = teacher(img_aug)
                normalized_teacher_output = teacher_output_normalization(teacher_output)
            student_output = student(img_aug)
            student_output_ae = student_output[:, out_channels:, :, :]
            diff_teacher_ae = (normalized_teacher_output - autoencoder_output) ** 2
            diff_student_ae = (autoencoder_output - student_output_ae) ** 2
            loss_ae = torch.mean(diff_teacher_ae)
            loss_student_ae = torch.mean(diff_student_ae)
            loss_total = loss_student + loss_ae + loss_student_ae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss_total.item())

            if it == 66500:
                optimizer.param_groups[0]["lr"] = 1e-5

    student_maps = []
    ae_maps = []
    student.eval()
    autoencoder.eval()
    with torch.no_grad():
        with tqdm(valid_dataloader, "Calculating parameters") as pbar:
            for img_val in pbar:
                img_val = img_val.to(device)
                teacher_output = teacher(img_val)
                student_output = student(img_val)
                autoencoder_output = autoencoder(img_val)
                normalized_teacher_output = teacher_output_normalization(teacher_output)
                student_output_st = student_output[:, :out_channels, :, :]
                student_output_ae = student_output[:, out_channels:, :, :]
                diff_teacher_student = (normalized_teacher_output - student_output_st) ** 2
                diff_student_ae = (autoencoder_output - student_output_ae) ** 2
                student_map = torch.mean(diff_teacher_student, dim=1, keepdim=True)
                ae_map = torch.mean(diff_student_ae, dim=1, keepdim=True)
                resized_student_map = F.interpolate(student_map, size=(img_size, img_size), mode="bilinear")
                resized_ae_map = F.interpolate(ae_map, size=(img_size, img_size), mode="bilinear")
                resized_student_map = resized_student_map.squeeze()
                resized_ae_map = resized_ae_map.squeeze()
                student_maps.append(resized_student_map)
                ae_maps.append(resized_ae_map)

    stacked_student_maps = torch.stack(student_maps, dim=0)
    stacked_ae_maps = torch.stack(ae_maps, dim=0)
    q_a_student = torch.quantile(input=stacked_student_maps, q=0.9)
    q_b_student = torch.quantile(input=stacked_student_maps, q=0.995)
    q_a_autoencoder = torch.quantile(input=stacked_ae_maps, q=0.9)
    q_b_autoencoder = torch.quantile(input=stacked_ae_maps, q=0.995)

    torch.save(student.state_dict(), os.path.join(model_dir, "student.pt"))
    torch.save(autoencoder.state_dict(), os.path.join(model_dir, "autoencoder.pt"))
    torch.save(mu, os.path.join(model_dir, "mu.pt"))
    torch.save(sigma, os.path.join(model_dir, "sigma.pt"))

    quantile: QuantileDict = {
        "q_a_student": q_a_student,
        "q_b_student": q_b_student,
        "q_a_autoencoder": q_a_autoencoder,
        "q_b_autoencoder": q_b_autoencoder,
    }
    torch.save(quantile, os.path.join(model_dir, "quantile.pt"))

    return teacher, student, autoencoder, mu, sigma, quantile


def inference(
    teacher: Union[PDN_S, PDN_M],
    student: Union[PDN_S, PDN_M],
    autoencoder: AutoEncoder,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    quantile: QuantileDict,
    model_dir: str,
    dataset_path: str,
    img_size: int = 256,
    device: str = "cuda",
) -> None:
    """
    Inference trained EfficientAD.

    This function outputs auroc scores for each anomaly type.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            build_imagenet_normalization(),
        ]
    )
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)

    efficient_ad_inference = EfficientADInference(
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        mu=mu,
        sigma=sigma,
        quantile=quantile,
        img_size=img_size,
        device=device,
    )

    auroc, auroc_for_anomalies = efficient_ad_inference.auroc_for_anomalies(test_dataset)

    print(f"auroc score = {auroc}")
    with open(os.path.join(model_dir, "results.txt"), "w") as f:
        print(f"auroc score = {auroc}", file=f)

    for anomaly_class, auroc_score in auroc_for_anomalies.items():
        print(f"{anomaly_class} auroc score = {auroc_score}")
        with open(os.path.join(model_dir, "results.txt"), "a") as f:
            print(f"{anomaly_class} auroc score = {auroc_score}", file=f)


def _load_or_compute_mu_and_sigma(
    model_dir: str,
    imagenet_dataset: torchvision.datasets.ImageFolder,
    feature_extractor: PretrainedFeatureExtractor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_samples_for_mean_and_variance = 10000

    mu_path = os.path.join(model_dir, "mu.pt")
    sigma_path = os.path.join(model_dir, "sigma.pt")

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    sample_idxs = torch.randint(
        len(imagenet_dataset),
        size=(n_samples_for_mean_and_variance,),
        generator=generator,
        device=device,
    )

    if not os.path.exists(mu_path):
        with torch.no_grad():
            spatial_means = []
            for idx in tqdm(sample_idxs, desc="Calculating mean"):
                [imagenet_img, _], _ = imagenet_dataset[idx]
                imagenet_img = imagenet_img.to(device)
                imagenet_img = imagenet_img[None, :]
                fe_output = feature_extractor(imagenet_img).to("cpu")
                fe_output = fe_output.squeeze()
                spatial_mean = fe_output.mean(dim=(1, 2))
                spatial_means.append(spatial_mean)
            stacked_spatial_means = torch.stack(spatial_means, dim=0)
            mu = stacked_spatial_means.mean(dim=0)
            torch.save(mu, mu_path)
    else:
        mu = torch.load(mu_path)
        print(f"Loaded mu from `{mu_path}`.")

    if not os.path.exists(sigma_path):
        with torch.no_grad():
            spatial_vars = []
            for idx in tqdm(sample_idxs, desc="Calculating std"):
                [imagenet_img, _], _ = imagenet_dataset[idx]
                imagenet_img = imagenet_img.to(device)
                imagenet_img = imagenet_img[None, :]
                fe_output = feature_extractor(imagenet_img).to("cpu")
                fe_output = fe_output.squeeze()
                tmp = (fe_output - mu[:, None, None]) ** 2
                spatial_var = tmp.mean(dim=(1, 2))
                spatial_vars.append(spatial_var)
            stacked_spatial_vars = torch.stack(spatial_vars, dim=0)
            var = stacked_spatial_vars.mean(dim=0)
            sigma = torch.sqrt(var)
            torch.save(sigma, sigma_path)
    else:
        sigma = torch.load(sigma_path)
        print(f"Loaded sigma from `{sigma_path}`.")

    return mu, sigma


def train_teacher(
    model_dir: str,
    imagenet_path: str,
    pdn_size: str = "s",
    out_channels: int = 384,
    img_size: int = 256,
    device: str = "cuda",
) -> None:
    """
    Training the teacher network for distillation for EfficientAD.

    This implementation followed `Algorithm 3` in the original EfficentAD paper:
    https://arxiv.org/abs/2303.14535
    """

    if not (pdn_size == "s" or pdn_size == "m"):
        raise ValueError("pdn_size must be `s` or `m`.")

    feature_extractor = PretrainedFeatureExtractor(out_channels=out_channels).to(device)
    feature_extractor.eval()

    grayscale_transform = transforms.RandomGrayscale(p=0.1)
    extractor_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            build_imagenet_normalization(),
        ]
    )
    pdn_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            build_imagenet_normalization(),
        ]
    )

    def train_transform(img):
        img = grayscale_transform(img)
        return extractor_transform(img), pdn_transform(img)

    imagenet_dataset = torchvision.datasets.ImageFolder(
        root=imagenet_path,
        transform=train_transform,
    )

    if pdn_size == "s":
        teacher = PDN_S(out_channels=out_channels, padding=True).to(device)
    else:
        teacher = PDN_M(out_channels=out_channels, padding=True).to(device)

    mu, sigma = _load_or_compute_mu_and_sigma(model_dir, imagenet_dataset, feature_extractor, device)

    optimizer = Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    dataloader = DataLoader(imagenet_dataset, batch_size=16, shuffle=True)

    n_iteration = 60000

    teacher.train()
    with tqdm(range(n_iteration), "Training teacher network") as pbar:
        for it in pbar:
            [imagenet_img_fe, imagenet_img_teacher], _ = next(iter(dataloader))
            imagenet_img_fe = imagenet_img_fe.to(device)
            imagenet_img_teacher = imagenet_img_teacher.to(device)
            with torch.no_grad():
                fe_output = feature_extractor(imagenet_img_fe)
                normalized_fe_output = transforms.Normalize(mu, sigma)(fe_output)
            teacher_output = teacher(imagenet_img_teacher)
            loss_batch = loss_fn(normalized_fe_output, teacher_output)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss_batch.item())

    torch.save(teacher.state_dict(), os.path.join(model_dir, "teacher.pt"))
