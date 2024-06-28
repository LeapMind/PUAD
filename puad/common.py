import torchvision


def build_imagenet_normalization() -> torchvision.transforms.Normalize:
    # mu and sigma are based on the values used in pytorch official for Wide_ResNet101_2:
    # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet101_2.html
    imagenet_mu = [0.485, 0.456, 0.406]
    imagenet_sigma = [0.229, 0.224, 0.225]
    return torchvision.transforms.Normalize(mean=imagenet_mu, std=imagenet_sigma)
