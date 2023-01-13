from modules.model import Model
import torch
import torch.nn as nn
import torchsummary as summary
if __name__ == '__main__':
    torch.manual_seed(0)
    
    model = Model(num_classes=2, depth=1)
    model.to("cuda")
    x = torch.randn(5, 3, 224, 224).to("cuda")

    model.eval()

    y = model(x)
    print(y.shape)

    summary.summary(model, (3, 224, 224))


    # LocalFeatureExtraction layers:
    # total params: 17_202
    # output shape: torch.Size([5, 32, 224, 224])
    # VisionTransformer layers:
    ## total params: 200_000_000
    ## output shape: torch.Size([-1, 257, 6272])

    # ClassificationHead layers:
    # total params: 16_386
    # output shape: torch.Size([5, 2])
