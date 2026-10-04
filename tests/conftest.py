import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prepare_kb import build_chunks      # noqa: E402
from src.retrieve import ChessRetriever      # noqa: E402
import json                                  # noqa: E402


@pytest.fixture(scope="session")
def chunks_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("kb") / "chunks.json"
    p.write_text(json.dumps(build_chunks(), ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def retriever(chunks_path):
    return ChessRetriever(chunks_path=chunks_path, mode="hybrid")
