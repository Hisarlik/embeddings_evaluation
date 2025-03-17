import torch

def test_mps():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("MPS is working properly:")
        print(x)
        print(f"Device: {x.device}")
    else:
        print("MPS device not found.")

if __name__ == "__main__":
    test_mps() 