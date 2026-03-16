import torch

from experiments.universal_encoder.models import RowEncoder

checkpoint = torch.load(
    "experiments/universal_encoder/best_model.pt", map_location="cpu", weights_only=False
)
print("Saved keys in checkpoint:", checkpoint.keys())
# Extract the encoder config
encoder_config = checkpoint["row_encoder_config"]
print("Encoder Config:", encoder_config)
# Recreate the model using the saved configurations
loaded_encoder = RowEncoder(**encoder_config)
# Load the weights
loaded_encoder.load_state_dict(checkpoint["row_encoder_state_dict"])
loaded_encoder.eval()
print("Model loaded successfully!")
