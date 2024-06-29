import argparse
import os

from puad.dataset import build_dataset
from puad.efficientad.inference import load_efficient_ad
from puad.puad import PUAD
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PUAD")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory containing `train` and `test` (and `validation` in MVTec LOCO AD Dataset)",
    )
    parser.add_argument(
        "model_dir_path",
        type=str,
        help="Path to directory containing pretrained models",
    )
    parser.add_argument(
        "--size",
        choices=["s", "m"],
        type=str,
        default="s",
        help=(
            "Specify the size of EfficientAD used for Picturable anomaly detection "
            "and feature extraction for Unpicturable anomaly detection in either `s` or `m`"
        ),
    )
    parser.add_argument(
        "--feature_extractor",
        choices=["student", "teacher"],
        type=str,
        default="student",
        help=(
            "Specify the network in EfficientAD used for feature extraction for Unpicturable anomaly detection "
            "in either `teacher` or `student`"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir, category = os.path.split(os.path.abspath(args.dataset_path))
    dataset_name = os.path.split(dataset_dir)[1]
    if not (
        os.path.exists(os.path.join(args.dataset_path, "train"))
        and os.path.exists(os.path.join(args.dataset_path, "test"))
    ):
        raise ValueError("The dataset specified in `dataset_path` must contain `train` and `test` directories.")
    print(f"dataset name : {dataset_name}")
    print(f"category : {category}")
    print(f"size : {args.size}")
    print(f"feature extractor : {args.feature_extractor}")

    # load EfficientAD
    efficient_ad_inference = load_efficient_ad(args.model_dir_path, args.size, dataset_name, category)

    # build dataset
    train_dataset, valid_dataset, test_dataset = build_dataset(args.dataset_path)

    # EfficientAD
    efficient_ad_auroc = efficient_ad_inference.auroc(test_dataset)
    print(f"efficient_ad auroc : {efficient_ad_auroc}")

    # PUAD
    puad = PUAD(feature_extractor=args.feature_extractor)
    puad.load_efficient_ad(efficient_ad_inference)
    puad.train(train_dataset)
    puad.valid(valid_dataset)
    puad_auroc = puad.auroc(test_dataset)
    print(f"puad auroc : {puad_auroc}")
