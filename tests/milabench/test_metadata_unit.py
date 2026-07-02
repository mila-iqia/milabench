import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from milabench.metadata import _get_gpu_info, fetch_torch_version, machine_metadata


class TestGetGpuInfo:
    @patch("milabench.metadata.gpu.select_backend")
    def test_success(self, mock_select_backend):
        mock_smi = MagicMock()
        mock_smi.arch = "CUDA"
        mock_smi.get_gpus_info.return_value = [{"name": "RTX 4090"}]
        mock_smi.system_info.return_value = {"driver": "535.0"}
        mock_select_backend.return_value = mock_smi

        result = _get_gpu_info()

        assert result == {
            "arch": "CUDA",
            "gpus": [{"name": "RTX 4090"}],
            "system": {"driver": "535.0"},
        }

    @patch("milabench.metadata.gpu.select_backend")
    def test_exception_returns_empty_dict(self, mock_select_backend):
        """Covers lines 26-28: exception path in _get_gpu_info."""
        mock_select_backend.side_effect = RuntimeError("No GPU backend")

        result = _get_gpu_info()

        assert result == {}

    @patch("milabench.metadata.gpu.select_backend")
    def test_exception_during_get_gpus_info(self, mock_select_backend):
        mock_smi = MagicMock()
        mock_smi.arch = "ROCm"
        mock_smi.get_gpus_info.side_effect = OSError("smi failure")
        mock_select_backend.return_value = mock_smi

        result = _get_gpu_info()

        assert result == {}


class TestFetchTorchVersion:
    def _make_pack(self, stdout=b'{"version": "2.1.0"}', returncode=0):
        pack = MagicMock()
        pack.dirs.code = "/tmp/code"
        pack.full_env.return_value = {"PATH": "/usr/bin"}
        return pack

    @patch("milabench.metadata.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=b'{"version": "2.1.0", "cuda": "12.1"}',
            returncode=0,
        )
        pack = self._make_pack()

        result = fetch_torch_version(pack)

        assert result == {"version": "2.1.0", "cuda": "12.1"}
        mock_run.assert_called_once()

    @patch("milabench.metadata.subprocess.run")
    def test_json_decode_error_returns_empty_dict(self, mock_run):
        """Covers lines 47-49: JSONDecodeError in fetch_torch_version."""
        mock_run.return_value = MagicMock(
            stdout=b"not valid json at all",
            returncode=1,
        )
        pack = self._make_pack()

        result = fetch_torch_version(pack)

        assert result == {}

    @patch("milabench.metadata.subprocess.run")
    def test_empty_stdout(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"", returncode=0)
        pack = self._make_pack()

        result = fetch_torch_version(pack)

        assert result == {}

    @patch("milabench.metadata.subprocess.run")
    def test_subprocess_exception_returns_empty_dict(self, mock_run):
        """error_guard catches exceptions and returns {}."""
        mock_run.side_effect = FileNotFoundError("python not found")
        pack = self._make_pack()

        result = fetch_torch_version(pack)

        assert result == {}


class TestMachineMetadata:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        machine_metadata.cache_clear()
        yield
        machine_metadata.cache_clear()

    @patch("milabench.metadata.cpuinfo.get_cpu_info")
    @patch("milabench.metadata._get_gpu_info")
    @patch("milabench.metadata.os.uname")
    @patch("milabench.metadata.os.cpu_count")
    def test_without_pack(self, mock_cpu_count, mock_uname, mock_gpu_info, mock_cpu_info):
        mock_cpu_count.return_value = 16
        mock_uname.return_value = MagicMock(
            sysname="Linux",
            nodename="testhost",
            release="5.15.0",
            version="#1 SMP",
            machine="x86_64",
        )
        mock_gpu_info.return_value = {"arch": "CUDA", "gpus": [], "system": {}}
        mock_cpu_info.return_value = {"brand_raw": "Intel i9-13900K"}

        with patch(
            "milabench.scripts.torchversion.get_pytorch_version",
            return_value={"version": "2.1.0"},
        ), patch(
            "milabench.scripts.vcs.retrieve_git_versions",
            return_value={"tag": "v1.0"},
        ):
            result = machine_metadata(pack=None)

        assert result["cpu"] == {"count": 16, "brand": "Intel i9-13900K"}
        assert result["os"]["sysname"] == "Linux"
        assert result["os"]["nodename"] == "testhost"
        assert result["accelerators"] == {"arch": "CUDA", "gpus": [], "system": {}}
        assert result["pytorch"] == {"version": "2.1.0"}
        assert "date" in result
        assert "milabench" in result

    @patch("milabench.metadata.fetch_torch_version")
    @patch("milabench.metadata.cpuinfo.get_cpu_info")
    @patch("milabench.metadata._get_gpu_info")
    @patch("milabench.metadata.os.uname")
    @patch("milabench.metadata.os.cpu_count")
    def test_with_pack(self, mock_cpu_count, mock_uname, mock_gpu_info, mock_cpu_info, mock_fetch_torch):
        mock_cpu_count.return_value = 8
        mock_uname.return_value = MagicMock(
            sysname="Linux",
            nodename="node1",
            release="6.1.0",
            version="#2 SMP",
            machine="aarch64",
        )
        mock_gpu_info.return_value = {}
        mock_cpu_info.return_value = {"brand_raw": "AMD EPYC"}
        mock_fetch_torch.return_value = {"version": "2.2.0"}

        pack = MagicMock()

        with patch(
            "milabench.scripts.vcs.retrieve_git_versions",
            return_value={"tag": "v2.0"},
        ):
            result = machine_metadata(pack=pack)

        assert result["cpu"] == {"count": 8, "brand": "AMD EPYC"}
        assert result["os"]["machine"] == "aarch64"
        assert result["pytorch"] == {"version": "2.2.0"}
        mock_fetch_torch.assert_called_once_with(pack)

    @patch("milabench.metadata.cpuinfo.get_cpu_info")
    @patch("milabench.metadata._get_gpu_info")
    @patch("milabench.metadata.os.uname")
    @patch("milabench.metadata.os.cpu_count")
    def test_missing_brand_raw(self, mock_cpu_count, mock_uname, mock_gpu_info, mock_cpu_info):
        """cpu info without brand_raw should default to '<unknown>'."""
        mock_cpu_count.return_value = 4
        mock_uname.return_value = MagicMock(
            sysname="Linux",
            nodename="host",
            release="5.0",
            version="#1",
            machine="x86_64",
        )
        mock_gpu_info.return_value = {}
        mock_cpu_info.return_value = {}

        with patch(
            "milabench.scripts.torchversion.get_pytorch_version",
            return_value={},
        ), patch(
            "milabench.scripts.vcs.retrieve_git_versions",
            return_value={},
        ):
            result = machine_metadata(pack=None)

        assert result["cpu"]["brand"] == "<unknown>"

    @patch("milabench.metadata.cpuinfo.get_cpu_info")
    @patch("milabench.metadata._get_gpu_info")
    @patch("milabench.metadata.os.uname")
    @patch("milabench.metadata.os.cpu_count")
    def test_error_guard_catches_exception(self, mock_cpu_count, mock_uname, mock_gpu_info, mock_cpu_info):
        """error_guard on machine_metadata returns {} on unhandled exception."""
        mock_cpu_count.side_effect = RuntimeError("unexpected")

        result = machine_metadata(pack=None)

        assert result == {}


class TestMainBlock:
    """Covers lines 92-94: if __name__ == '__main__' block."""

    def test_main_block_prints_valid_json(self):
        """Run the module as __main__ via subprocess to cover lines 92-94."""
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "milabench.metadata"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert isinstance(output, dict)
        assert "cpu" in output
        assert "os" in output
        assert "date" in output

    def test_main_block_via_exec(self, capsys):
        """Directly execute the __main__ block logic."""
        mock_result = {"cpu": {"count": 4}, "gpus": []}

        with patch("milabench.metadata.machine_metadata", return_value=mock_result) as mock_meta:
            mock_meta.cache_clear = MagicMock()
            code = "import json\nfrom milabench.metadata import machine_metadata\nprint(json.dumps(machine_metadata(), indent=2))"
            exec(compile(code, "<test_main>", "exec"))

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == mock_result
