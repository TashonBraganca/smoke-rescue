"""Optional pretrained-model check compares our shortcut with a full forward pass."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from brain.place import DinoValueExtractor, IMAGE_SIZE


@pytest.mark.filterwarnings("ignore:xFormers is not available:UserWarning")
@pytest.mark.skipif(os.environ.get("SMOKE_RESCUE_MODEL_CHECK") != "1",
                    reason="pretrained weights required; set SMOKE_RESCUE_MODEL_CHECK=1")
def test_late_value_facet_equals_hook_from_full_dinov2_forward():
    root = Path(__file__).resolve().parents[2]
    with pytest.warns(UserWarning, match="MPS is unavailable or CPU was selected"):
        extractor = DinoValueExtractor("dinov2_vitb14", "cpu", root / "data/runs/place_assets/torch_hub")
    path = root / "data/fixtures/tiny_2robots/robot_A/rgb/0.000000.png"
    if not path.exists():
        path = next((root / "data/fixtures/tiny_2robots/robot_A/rgb").glob("*.png"))
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    captured = []
    handle = extractor.model.blocks[extractor.layer].attn.qkv.register_forward_hook(
        lambda _module, _args, output: captured.append(output.detach().clone()))
    tensor = torch.from_numpy(cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        extractor.model((tensor - extractor.mean) / extractor.std)
    handle.remove()
    expected = torch.nn.functional.normalize(captured[0][:, 1:, 2 * extractor.model.embed_dim:], dim=-1)[0].numpy()
    actual = extractor(rgb)
    assert actual.shape == (391, 768)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
