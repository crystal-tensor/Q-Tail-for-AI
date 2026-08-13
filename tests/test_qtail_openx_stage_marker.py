import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.qtail_openx_stage_marker import marker_status, write_marker


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class OpenXStageMarkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare_download_marker(self) -> None:
        write_json(
            self.root / "openx_1t_object_manifest.json",
            {"object_count": 1, "total_bytes": 3},
        )
        write_json(self.root / "openx_1t_checksum_manifest.json", {"objects": []})
        write_json(
            self.root / "download_checksum_ledger.json",
            {
                "objects": {
                    "sample": {
                        "official_md5_base64": "abc",
                        "local_md5_base64": "abc",
                    }
                }
            },
        )
        write_json(
            self.root / "download_verification.json",
            {
                "status": "verified",
                "expected_objects": 1,
                "complete_objects": 1,
                "md5_verified_objects": 1,
                "expected_bytes": 3,
                "complete_bytes": 3,
                "missing": [],
                "size_mismatch": [],
                "ledger_mismatch": [],
                "partials": [],
            },
        )
        write_marker(self.root, "download")

    def prepare_training_marker(self) -> Path:
        self.prepare_download_marker()
        training = self.root / "training"
        training.mkdir()
        checkpoint = training / "qtail_allocation_head.pt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write_json(
            training / "training_runtime_status.json",
            {"status": "complete", "returncode": 0},
        )
        write_json(
            training / "openx_demo_training_report.json",
            {
                "status": "complete",
                "steps": 20000,
                "shard_count": 12,
                "model_artifact": {"sha256": checkpoint_sha},
            },
        )
        (training / "openx_shard_training_rows.csv").write_text(
            "path,weight\nsample,1\n", encoding="utf-8"
        )
        write_json(training / "feature_cache_usage.json", {"cached_rows": 12})
        write_json(
            training / "optimizer_progress.json",
            {
                "status": "phase_complete",
                "phase": "qtail",
                "step": 20000,
                "steps_target": 20000,
                "overall_completed_updates": 40000,
                "overall_target_updates": 40000,
            },
        )
        resume = training / "resume_checkpoints"
        resume.mkdir()
        (resume / "source.pt").write_bytes(b"source optimizer checkpoint")
        (resume / "qtail.pt").write_bytes(b"qtail optimizer checkpoint")
        write_marker(self.root, "training")
        return checkpoint

    def test_download_marker_rejects_changed_verification(self) -> None:
        write_json(
            self.root / "openx_1t_object_manifest.json",
            {"object_count": 1, "total_bytes": 3},
        )
        write_json(self.root / "openx_1t_checksum_manifest.json", {"objects": []})
        write_json(
            self.root / "download_checksum_ledger.json",
            {
                "objects": {
                    "sample": {
                        "official_md5_base64": "abc",
                        "local_md5_base64": "abc",
                    }
                }
            },
        )
        verification = {
            "status": "verified",
            "expected_objects": 1,
            "complete_objects": 1,
            "md5_verified_objects": 1,
            "expected_bytes": 3,
            "complete_bytes": 3,
            "missing": [],
            "size_mismatch": [],
            "ledger_mismatch": [],
            "partials": [],
        }
        write_json(self.root / "download_verification.json", verification)
        write_marker(self.root, "download")
        self.assertTrue(marker_status(self.root, "download")["valid"])
        verification["status"] = "incomplete"
        write_json(self.root / "download_verification.json", verification)
        self.assertFalse(marker_status(self.root, "download")["valid"])

    def test_training_marker_rejects_changed_checkpoint(self) -> None:
        checkpoint = self.prepare_training_marker()
        self.assertTrue(marker_status(self.root, "training")["valid"])
        checkpoint.write_bytes(b"changed")
        self.assertFalse(marker_status(self.root, "training")["valid"])

    def test_training_marker_rejects_changed_download_parent(self) -> None:
        self.prepare_training_marker()
        self.assertTrue(marker_status(self.root, "training")["valid"])
        download_marker = self.root / "OPENX_1T_DOWNLOAD_COMPLETE"
        download_marker.write_text(
            download_marker.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        self.assertFalse(marker_status(self.root, "training")["valid"])

    def test_synthesis_marker_rejects_changed_delivery_artifact(self) -> None:
        self.prepare_training_marker()
        synthesis = self.root / "synthesis"
        synthesis.mkdir()
        write_json(
            synthesis / "synthesis_runtime_status.json",
            {"status": "complete", "returncode": 0},
        )
        write_json(
            synthesis / "qtail_service_delivery_report.json",
            {
                "customer_package": {
                    "validation": {"valid": True, "winner": "qtail_synthetic"}
                }
            },
        )
        for name in (
            "qtail_service_synthetic_plan.csv",
            "qtail_synthetic_data.csv",
            "qtail_service_model_card.json",
            "qtail_data_engine_report.json",
            "README_QTAIL_DELIVERY.md",
            "qtail_delivery_package.zip",
        ):
            (synthesis / name).write_bytes(b"artifact")
        write_marker(self.root, "synthesis")
        self.assertTrue(marker_status(self.root, "synthesis")["valid"])
        (synthesis / "qtail_synthetic_data.csv").write_bytes(b"changed")
        self.assertFalse(marker_status(self.root, "synthesis")["valid"])

    def test_synthesis_marker_rejects_changed_training_parent(self) -> None:
        self.prepare_training_marker()
        synthesis = self.root / "synthesis"
        synthesis.mkdir()
        write_json(
            synthesis / "synthesis_runtime_status.json",
            {"status": "complete", "returncode": 0},
        )
        write_json(
            synthesis / "qtail_service_delivery_report.json",
            {
                "customer_package": {
                    "validation": {"valid": True, "winner": "qtail_synthetic"}
                }
            },
        )
        for name in (
            "qtail_service_synthetic_plan.csv",
            "qtail_synthetic_data.csv",
            "qtail_service_model_card.json",
            "qtail_data_engine_report.json",
            "README_QTAIL_DELIVERY.md",
            "qtail_delivery_package.zip",
        ):
            (synthesis / name).write_bytes(b"artifact")
        write_marker(self.root, "synthesis")
        self.assertTrue(marker_status(self.root, "synthesis")["valid"])
        training_marker = self.root / "OPENX_1T_TRAINING_COMPLETE"
        training_marker.write_text(
            training_marker.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        self.assertFalse(marker_status(self.root, "synthesis")["valid"])


if __name__ == "__main__":
    unittest.main()
