from pathlib import Path
import unittest

import numpy as np

from cistardist_pytorch.config import StarDist2DConfig, load_thresholds
from cistardist_pytorch.converter import convert_h5_to_state_dict
from cistardist_pytorch.model import StarDist2D
from cistardist_pytorch.net import StarDist2DNet


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "SD_Nuclei_Versatile"


class V1ModelTests(unittest.TestCase):
    def test_load_config_and_thresholds(self):
        config = StarDist2DConfig.from_json(MODEL_DIR / "config.json")
        self.assertEqual(config.n_rays, 32)
        self.assertEqual(config.grid, (2, 2))
        self.assertEqual(config.n_channel_in, 1)
        self.assertEqual(config.axes_div_by(), (16, 16))

        thresholds = load_thresholds(MODEL_DIR / "thresholds.json")
        self.assertAlmostEqual(thresholds["prob"], 0.479071463157368)
        self.assertAlmostEqual(thresholds["nms"], 0.3)

    def test_build_network_output_shape(self):
        config = StarDist2DConfig.from_json(MODEL_DIR / "config.json")
        net = StarDist2DNet(config)
        self.assertIn("prob", net.keras_layer_names)
        self.assertIn("dist", net.keras_layer_names)
        self.assertEqual(net.layers["dist"].out_channels, 32)

        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        x = torch.zeros(1, 1, 256, 256)
        prob, dist = net(x)
        self.assertEqual(tuple(prob.shape), (1, 1, 128, 128))
        self.assertEqual(tuple(dist.shape), (1, 32, 128, 128))

    def test_convert_h5_consumes_expected_layers(self):
        try:
            import h5py  # noqa: F401
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"{exc.name} is not installed")

        config = StarDist2DConfig.from_json(MODEL_DIR / "config.json")
        state_dict, report = convert_h5_to_state_dict(MODEL_DIR / "weights_best.h5", config)
        self.assertEqual(report["n_layers"], len(StarDist2DNet(config).keras_layer_names))
        self.assertIn("layers.prob.weight", state_dict)
        self.assertIn("layers.dist.bias", state_dict)

    def test_predict_shapes_on_synthetic_image(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"{exc.name} is not installed")

        config = StarDist2DConfig.from_json(MODEL_DIR / "config.json")
        net = StarDist2DNet(config)
        model = StarDist2D(net, config)
        image = np.zeros((65, 66), dtype=np.float32)
        prob, dist = model.predict(image, normalize=False)
        self.assertEqual(prob.shape, (33, 33))
        self.assertEqual(dist.shape, (33, 33, 32))

    def test_compiled_nms_suppresses_identical_polygon(self):
        try:
            from cistardist_pytorch._c_nms import c_non_max_suppression_inds
        except ModuleNotFoundError:
            self.skipTest("compiled NMS extension is not built")

        dist = np.full((2, 32), 10, dtype=np.float32)
        points = np.array([[20, 20], [20, 20]], dtype=np.float32)
        keep = c_non_max_suppression_inds(dist, points, 1, 1, 0, np.float32(0.3))
        self.assertEqual(keep.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
