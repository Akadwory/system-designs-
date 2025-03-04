import torch
import torch.nn as nn
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerIntent(nn.Module):
    """Production-grade transformer for intent classification."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, n_layers: int = 6, 
                 n_heads: int = 4, dropout: float = 0.1, max_seq_len: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.pos_encoding = self._generate_pos_encoding(max_seq_len, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(input_dim, 10)  # 10 intent classes
        self._init_weights()

    def _generate_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Positional encoding for temporal stability."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with positional encoding."""
        if x.shape[1] > self.pos_encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max {self.pos_encoding.shape[1]}")
        x = x + self.pos_encoding[:, :x.shape[1], :].to(x.device)
        x = self.encoder(x)  # [batch, seq, dim]
        return self.fc(x[:, -1, :])  # Last timestep output

def export_model(model_path: str = "transformer.onnx", batch_size: int = 1):
    """Export to ONNX for TensorRT optimization."""
    try:
        model = TransformerIntent()
        model.eval()
        dummy_input = torch.randn(batch_size, 100, 128)
        torch.onnx.export(
            model, dummy_input, model_path, 
            input_names=['input'], output_names=['output'], 
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13
        )
        logger.info(f"Model exported to {model_path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    export_model()