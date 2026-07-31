from __future__ import annotations

import unittest

from silhouette_detector.device import resolve_torch_device, select_onnx_providers


class FakeCuda:
    def __init__(self, available: bool, count: int = 0) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class FakeTorch:
    def __init__(self, available: bool, count: int = 0) -> None:
        self.cuda = FakeCuda(available, count)


class DeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_without_cuda(self) -> None:
        self.assertEqual(resolve_torch_device("auto", FakeTorch(False)), "cpu")

    def test_auto_uses_first_cuda_device(self) -> None:
        self.assertEqual(resolve_torch_device("auto", FakeTorch(True, 2)), "cuda:0")

    def test_explicit_cuda_never_silently_falls_back(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_torch_device("cuda", FakeTorch(False))

    def test_invalid_cuda_index_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_torch_device("cuda:2", FakeTorch(True, 1))

    def test_onnx_auto_prefers_cuda_then_cpu(self) -> None:
        providers = select_onnx_providers(
            "auto", ["CPUExecutionProvider", "CUDAExecutionProvider"]
        )
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_onnx_explicit_cuda_requires_provider(self) -> None:
        with self.assertRaises(RuntimeError):
            select_onnx_providers("cuda", ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
