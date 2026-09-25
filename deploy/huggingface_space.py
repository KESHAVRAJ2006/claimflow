"""Upload the API to a Hugging Face Space: free Docker hosting with 2 vCPU and 16 GB RAM.

The alternative to Render's free plan (render.yaml) for the API. Both run the same production image, which fits in
512 MB; a Space has twenty times the CPU of a free Render instance, so it starts and embeds much faster.

Run on your own machine, from the repository root, once you have logged in with `hf auth login`:

    pip install --upgrade huggingface_hub
    python deploy/huggingface_space.py <your-username>/claimflow-api

What is uploaded is exactly what git has committed under backend/, so .env files, the virtualenv and caches can
never be uploaded, plus the README header that tells Hugging Face to build the Dockerfile and send traffic to
port 8000. The Space builds the Dockerfile's last stage: the same production image render.yaml describes.
Secrets (database URL, API keys) are set in the Space's settings, never here.
"""

import argparse
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

README = """---
title: ClaimFlow API
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Agentic insurance claim triage API (FastAPI + LangGraph)
---

# ClaimFlow API

The API of ClaimFlow, an agentic insurance claim triage system: LLM agents investigate a claim with read-only tools,
a deterministic rules engine applies eligibility and fraud checks, and a person makes the final decision.

Every endpoint except `/api/health` requires the `X-API-Key` header. Interactive documentation: `/api/docs`.
{source}
Recommendations are AI-assisted and require human review; this deployment holds synthetic demo data only.
"""


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def space_url(repo_id: str) -> str:
    """The Space's direct URL, where the API answers.

    Args:
        repo_id: "owner/name".

    Returns:
        https://owner-name.hf.space (lower case, other characters as hyphens), as Hugging Face builds it.
    """
    return "https://" + re.sub(r"[^a-z0-9]+", "-", repo_id.lower().replace("/", "-")).strip("-") + ".hf.space"


def main() -> int:
    """Export backend/ from git, add the Space README, and upload it.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Upload the ClaimFlow API to a Hugging Face Docker Space.")
    parser.add_argument("repo_id", help='the Space, as "username/space-name"; created if it does not exist')
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo_id):
        parser.error('give the Space as "username/space-name"')

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install the Hugging Face client first:  pip install --upgrade huggingface_hub", file=sys.stderr)
        return 2

    if _git("status", "--porcelain", "--", "backend").strip():
        print("backend/ has uncommitted changes. Commit them first: the upload takes what git has committed.")
        return 1
    commit = _git("rev-parse", "--short", "HEAD").decode().strip()
    try:
        origin = _git("remote", "get-url", "origin").decode().strip().removesuffix(".git")
    except subprocess.CalledProcessError:
        origin = ""
    source = f"Source code and documentation: {origin}\n" if origin.startswith("https://") else ""

    api = HfApi()
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", "HEAD", "backend"))) as archive:
            archive.extractall(temporary, filter="data")
        folder = Path(temporary) / "backend"
        (folder / "README.md").write_text(README.format(source=source), encoding="utf-8")
        api.create_repo(args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id, repo_type="space", folder_path=folder, commit_message=f"Deploy ClaimFlow {commit}"
        )

    print(f"Uploaded commit {commit}. The Space now builds the image (about 10 minutes the first time):")
    print(f"  build log and settings: https://huggingface.co/spaces/{args.repo_id}")
    print(f"  API, once running:      {space_url(args.repo_id)}/api/health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
