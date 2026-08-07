"""
ONNX Export Script for ResNet18 Embedding Model.

Exports the trained or pretrained ResNet18 embedder to ONNX format
for use with ONNX Runtime inference in the production pipeline.
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.resnet.embedding_model import create_embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    output_path: str = "models/resnet/resnet18.onnx",
    checkpoint_path: str = None,
    embedding_dim: int = 512,
    input_size: int = 224,
    opset_version: int = 17,
):
    """
    Exports the ResNet18 embedding model to ONNX.

    Args:
        output_path: Path to save the exported ONNX model.
        checkpoint_path: Optional path to a trained .pt checkpoint.
        embedding_dim: Embedding vector dimension.
        input_size: Input image size (square).
        opset_version: ONNX opset version.
    """
    logger.info("Creating ResNet18 embedding model...")
    model = create_embedding_model(pretrained=True, embedding_dim=embedding_dim)

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Verify forward pass
    with torch.no_grad():
        output = model(dummy_input)
    logger.info(f"Model output shape: {output.shape}")
    assert output.shape == (1, embedding_dim), f"Unexpected output shape: {output.shape}"

    # Export
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        dynamo=False,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )

    # Verify exported model
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_output = session.run(None, {"input": dummy_input.numpy()})[0]
    logger.info(f"ONNX verification output shape: {ort_output.shape}")
    logger.info(f"Successfully exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ResNet18 embedder to ONNX")
    parser.add_argument("--output", default="models/resnet/resnet18.onnx", help="Output ONNX path")
    parser.add_argument("--checkpoint", default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension")
    args = parser.parse_args()

    export_to_onnx(output_path=args.output, checkpoint_path=args.checkpoint, embedding_dim=args.dim)
