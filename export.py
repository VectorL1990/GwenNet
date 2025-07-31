import torch
from PytorchNet import Net

def export():
    #model = Net(in_features_num = 200)
    model = Net(in_features_num = 59)
    model.eval()

    #dummyInput = torch.randn(1, 200, 18, 8)
    dummyInput = torch.randn(1, 59, 14, 4)

    tracedModel = torch.jit.trace(model, dummyInput)
    tracedModel.save("model.pt")

if __name__ == "__main__":
    export()

