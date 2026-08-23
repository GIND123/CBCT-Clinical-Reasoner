"""Back the competition artifacts up to a private Hugging Face repository.

Everything needed to rebuild or audit the submission goes up: the fitted report
and its gates, the selection trajectory and length curve, the metrics, the
container sources, and the submission tarball itself.

**The repository is private, and this script will not make it public.** The
prototype bank contains sentences taken verbatim from the access-controlled
ToothFairy4 release, so the artifacts are patient-derived text. Nothing here
accepts a ``--public`` flag; making it public is a decision that belongs with
the data-use agreement, not with a backup script.

The token is read from ``hf`` in .env and never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Individual files worth keeping, relative to the repository root.
FILES = (
    "submission/inference.py",
    "submission/report_model.py",
    "submission/Dockerfile.slim",
    "submission/requirements-slim.txt",
    "submission/do_build_slim.sh",
    "submission/do_stress_test.sh",
    "src/cbct_reasoner/decode/adaptive.py",
    "src/cbct_reasoner/decode/redundancy.py",
    "src/cbct_reasoner/ontology.py",
    "src/cbct_reasoner/metrics/official.py",
    "src/cbct_reasoner/metrics/radfact.py",
    "scripts/optimize_final_score.py",
    "scripts/choose_report_length.py",
    "scripts/fit_adaptive_report.py",
    "scripts/build_submission.py",
)

#: Artifacts directory entries to skip: large, regenerable, or both.
SKIP_SUFFIXES = (".npy", ".npz", ".pt", ".pth")


def load_token() -> str:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() in {"hf", "hf_token"} and value.strip():
                return value.strip().strip("\"'")
    raise SystemExit("no Hugging Face token: set HF_TOKEN or add hf=... to .env")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="cbct-clinical-reasoner-odin2026")
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--skip-tarballs",
        action="store_true",
        help="upload sources and artifacts only; the images are ~116 MB each",
    )
    parser.add_argument(
        "--only-tarball",
        default=None,
        help="upload just this one image, by filename",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = load_token()
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami().get("name")
    if not namespace:
        raise SystemExit("could not determine the Hugging Face namespace")
    repo_id = f"{namespace}/{args.repo}"

    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"private repo: {repo_id}")

    uploaded = 0
    for relative in FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            print(f"  skip (absent) {relative}")
            continue
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=relative,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded += 1
        print(f"  {relative}")

    artifacts = REPO_ROOT / "artifacts"
    if artifacts.is_dir():
        for source in sorted(artifacts.rglob("*")):
            if not source.is_file() or source.suffix in SKIP_SUFFIXES:
                continue
            relative = source.relative_to(REPO_ROOT).as_posix()
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=relative,
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded += 1
            print(f"  {relative}")

    if not args.skip_tarballs:
        pattern = args.only_tarball or "*.tar.gz"
        for tarball in sorted((REPO_ROOT / "submission").glob(pattern)):
            size_mb = tarball.stat().st_size / 1_048_576
            print(f"  uploading {tarball.name} ({size_mb:.0f} MB) ...", flush=True)
            api.upload_file(
                path_or_fileobj=str(tarball),
                path_in_repo=f"submission/{tarball.name}",
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded += 1

    print(f"\n{uploaded} files -> https://huggingface.co/{repo_id}  (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
