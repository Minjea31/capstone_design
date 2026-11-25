"""Exercise both pruning implementations without modifying installed PyTorch."""

from pathlib import Path
import subprocess
import sys
import unittest


TRAINING_DIR = Path(__file__).resolve().parents[1]
CHECK = """
import inspect
import sys
import torch
import torch.nn.utils.prune as installed_prune

sys.path.insert(0, sys.argv[1])
from compress import GM, prune
from compress.Compress import PruneHandler

assert 'module_name' not in inspect.signature(installed_prune.l1_unstructured).parameters
torch.manual_seed(7)
for channels in (8, 64, 160):
    conv = torch.nn.Conv2d(3, channels, 3, bias=False)
    GM.gm_structured(conv, name='weight', module_name='test.conv', amount=0.5, dim=0)
    assert (conv.weight_mask.flatten(1).sum(1) > 0).sum().item() == channels // 2
    assert conv(torch.ones(1, 3, 8, 8)).shape == (1, channels, 6, 6)
    prune.remove(conv, 'weight')
    assert not hasattr(conv, 'weight_mask')

bn = torch.nn.BatchNorm2d(4)
prune.l1_unstructured(bn, name='weight', module_name='test.bn', amount=0.5)
assert bn.weight_mask.sum().item() == 2
prune.remove(bn, 'weight')
assert 'module_name' not in inspect.signature(installed_prune.l1_unstructured).parameters
"""


class PruningTests(unittest.TestCase):
    def check_implementation(self, folder):
        # Each experiment imports a package called compress; isolate the imports.
        result = subprocess.run(
            [sys.executable, "-c", CHECK, str(TRAINING_DIR / folder / "yolov12")],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_weather_training_pruning(self):
        self.check_implementation("yolov12")

    def test_ga_training_pruning(self):
        self.check_implementation("yolov12_pruning")


if __name__ == "__main__":
    unittest.main()
