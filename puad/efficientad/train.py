import argparse
import os
import sys

import torch

# HACK: To resolve import path issue
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)

from puad.efficientad.algorithms import inference, train_student_and_autoencoder, train_teacher  # noqa: E261, E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientAD")
    parser.add_argument(
        "target",
        type=str,
        choices=["teacher", "student"],
        help=(
            "Specify the training network in either `teacher` or `student`. "
            "Before training `student`, `teacher` must have been pre-trained."
        ),
    )
    parser.add_argument(
        "imagenet_path",
        type=str,
        help="Path to ImageNet dataset (ILSVRC 2012).",
    )
    parser.add_argument(
        "model_dir_path",
        type=str,
        help="Path to directory to save/load trained model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help=(
            "Path to the directory of the dataset to be trained, "
            "containing `train` (and `validation` in MVTec LOCO AD Dataset). "
            "This must be specified when training `student'."
        ),
    )
    parser.add_argument(
        "--size",
        choices=["s", "m"],
        type=str,
        default="s",
        help="Specify the network size in either `s` or `m`",
    )
    args = parser.parse_args()

    if args.target == "student" and args.dataset_path is None:
        raise ValueError("`--dataset_path` must be specified when training `student'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_model_dir = os.path.join(
        args.model_dir_path,
        f"{args.size}_size",
        "teacher",
    )
    teacher_model_path = os.path.join(teacher_model_dir, "teacher.pt")

    if args.target == "teacher":
        if os.path.exists(teacher_model_path):
            raise FileExistsError(
                f"Trained teacher model already exists in `{teacher_model_dir}`. "
                "Training script have been stopped to avoid unintentional overwriting."
            )

        os.makedirs(teacher_model_dir, exist_ok=True)
        train_teacher(teacher_model_dir, args.imagenet_path, args.size, device=device)

    if args.target == "student":
        if not os.path.exists(teacher_model_path):
            raise ValueError(
                "Before training `student`, `teacher` must have been pre-trained. "
                "If you have pre-trained teacher model files, "
                f"please check that they are placed in `{teacher_model_dir}`."
            )
        dataset_dir, category = os.path.split(os.path.abspath(args.dataset_path))
        dataset_name = os.path.split(dataset_dir)[1]
        model_dir = os.path.join(
            args.model_dir_path,
            f"{args.size}_size",
            dataset_name,
            category,
        )
        os.makedirs(model_dir, exist_ok=True)

        res = train_student_and_autoencoder(
            model_dir=model_dir,
            dataset_path=args.dataset_path,
            imagenet_path=args.imagenet_path,
            teacher_model_path=teacher_model_path,
            pdn_size=args.size,
            out_channels=384,
            img_size=256,
            device=device,
        )

        inference(*res, model_dir=model_dir, dataset_path=args.dataset_path, img_size=256, device=device)
