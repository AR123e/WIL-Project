from src.prepare_kb import build_chunks, DOCS


def test_every_document_contributes_chunks_and_ids_are_unique():
    chunks = build_chunks()
    assert {c["doc_id"] for c in chunks} == {d["doc_id"] for d in DOCS}
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_carry_citation_and_license_metadata():
    for c in build_chunks():
        assert c["doc_title"] and c["section"] and c["source"] and c["license"]


def test_list_items_stay_with_their_heading():
    chunks = build_chunks()
    lost = next(c for c in chunks if "right to castle has been lost" in c["text"].lower())
    assert "king has already moved" in lost["text"].lower()        # the bullets are not split away


def test_intro_and_howto_tags_exist():
    chunks = build_chunks()
    assert [c for c in chunks if "intro" in c["tags"]]
    assert [c for c in chunks if "howto" in c["tags"]]
    assert any("chess is a two-player strategy board game" in c["text"].lower() for c in chunks if "intro" in c["tags"])


def test_no_empty_or_heading_only_chunks():
    assert all(len(c["text"].split()) >= 6 for c in build_chunks())
