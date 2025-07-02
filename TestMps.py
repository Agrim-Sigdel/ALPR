import torch
print(torch.backends.mps.is_available())  # Should print True
print(torch.backends.mps.is_built())      # Should print True
