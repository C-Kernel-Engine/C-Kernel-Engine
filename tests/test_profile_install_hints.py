import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VISUALIZERS = (
    ROOT / "version/v7/tools/open_ir_visualizer.py",
    ROOT / "version/v8/tools/open_ir_visualizer_v8.py",
)
ARCH_HINTS = [
    "sudo pacman -S base-devel git perf valgrind",
    "# Intel hosts: sudo pacman -S intel-oneapi-toolkit",
    "# Intel hosts: source /opt/intel/oneapi/setvars.sh",
    "git clone https://github.com/brendangregg/FlameGraph.git",
    "chmod +x FlameGraph/stackcollapse-perf.pl FlameGraph/flamegraph.pl",
    "sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0 kernel.yama.ptrace_scope=0",
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileInstallHintsTests(unittest.TestCase):
    def test_arch_family_hints_are_consistent_across_visualizers(self):
        distro_cases = (
            {"ID": "arch", "ID_LIKE": ""},
            {"ID": "cachyos", "ID_LIKE": "arch"},
            {"ID": "manjaro", "ID_LIKE": "arch"},
            {"ID": "endeavouros", "ID_LIKE": "arch"},
        )

        for path in VISUALIZERS:
            module = _load_module(path)
            for os_release in distro_cases:
                with self.subTest(visualizer=path.name, distro=os_release["ID"]):
                    with patch.object(module, "_read_os_release", return_value=os_release):
                        self.assertEqual(module._profile_install_hints(), ARCH_HINTS)

    def test_v8_likwid_hints_cover_ubuntu_source_and_cachyos_aur(self):
        module = _load_module(VISUALIZERS[1])
        with patch.object(
            module, "_read_os_release", return_value={"ID": "ubuntu", "ID_LIKE": "debian"}
        ):
            hints = module._likwid_install_hints()
        self.assertEqual(hints["distro_family"], "ubuntu")
        self.assertEqual(hints["recommended"], "ubuntu_source")
        self.assertIn("sudo apt-get install -y likwid", hints["commands"]["ubuntu_package"])
        self.assertTrue(
            any("--branch v5.5.1" in command for command in hints["commands"]["ubuntu_source"])
        )

        with patch.object(
            module, "_read_os_release", return_value={"ID": "cachyos", "ID_LIKE": "arch"}
        ):
            hints = module._likwid_install_hints()
        self.assertEqual(hints["distro_family"], "arch")
        self.assertEqual(hints["recommended"], "arch_aur")
        self.assertIn("makepkg -si", hints["commands"]["arch_aur"])
        self.assertIn("likwid-perfctr -a", hints["commands"]["verify"])


if __name__ == "__main__":
    unittest.main()
