from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.manifest import Manifest


def test_manifest_persists_checkpoint(tmp_path):
    settings = Settings(project_root=tmp_path)
    manifest = Manifest(settings)
    run_id = manifest.begin()
    manifest.data["chunks"]["chunk-1"] = "hash"
    manifest.finish(run_id)
    loaded = Manifest(settings)
    assert loaded.data["chunks"]["chunk-1"] == "hash"
    assert loaded.data["runs"][run_id]["status"] == "completed"
