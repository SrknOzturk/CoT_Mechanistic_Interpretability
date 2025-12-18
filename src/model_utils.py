import torch
from transformer_lens import HookedTransformer
from typing import Optional, Union

def load_model(model_name: str, device: Optional[Union[str, torch.device]] = "cuda"):
  """
  Loads a pretrained Transformer-Lens model and returns it in evaluation mode.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Loading model: {model_name} on {device}...")
  
  model= HookedTransformer.from_pretrained(
      model_name,
      device=device,
      dtype=torch.float32
  )
  model.eval()
  device_of_model = next(model.parameters()).device
  print(f"Model loaded successfully. Total layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
  return model