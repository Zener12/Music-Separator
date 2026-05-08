import torch
from train import load_checkpoint

def main():
    checkpoint_path = './checkpoints/checkpoint_epoch_48.pt'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, _ = load_checkpoint(checkpoint_path, device)

    some_tensor = torch.randn(1,1,1025,130).to(device)

    onnx_path = './checkpoints/model.onnx'

    torch.onnx.export(model, 
                      some_tensor, 
                      onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 
                                    'output': {0: 'batch'}},
                      do_constant_folding=True,
                      verbose = False)

if __name__ == '__main__':
    main()