import torch
import vectorizacion


def test_analyze_pdf_sentence_chunking(monkeypatch):
    # Patch PDF extraction to avoid I/O
    sample_text = "First sentence. Second sentence about Python and libraries. Third sentence with more info."
    monkeypatch.setattr(vectorizacion, "extract_text_from_pdf", lambda p: sample_text)

    # Dummy model that returns deterministic embeddings (same dim)
    class DummyModel:
        def __init__(self, dim=16):
            self.dim = dim
        def encode(self, texts, convert_to_tensor=True):
            if isinstance(texts, list):
                return torch.tensor([[0.1] * self.dim for _ in texts])
            return torch.tensor([0.1] * self.dim)

    dummy = DummyModel(dim=16)
    # Create a query embedding using the dummy model
    query_embedding = dummy.encode("Python", convert_to_tensor=True)

    score = vectorizacion.analyze_pdf(
        pdf_path="dummy.pdf",
        search_query="Python",
        query_embedding=query_embedding,
        model=dummy,
        embedding_method='sentence',
        use_chunking=True,
        max_chunks=2,
        model_name=None,
    )

    assert score is not None
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_analyze_pdf_sentence_whole_document(monkeypatch):
    sample_text = "This CV mentions Python and testing. Experience in APIs and Flask."
    monkeypatch.setattr(vectorizacion, "extract_text_from_pdf", lambda p: sample_text)

    class DummyModel:
        def __init__(self, dim=16):
            self.dim = dim
        def encode(self, texts, convert_to_tensor=True):
            if isinstance(texts, list):
                return torch.tensor([[0.2] * self.dim for _ in texts])
            return torch.tensor([0.2] * self.dim)

    dummy = DummyModel(dim=16)
    query_embedding = dummy.encode("Python", convert_to_tensor=True)

    score = vectorizacion.analyze_pdf(
        pdf_path="dummy.pdf",
        search_query="Python",
        query_embedding=query_embedding,
        model=dummy,
        embedding_method='sentence',
        use_chunking=False,
        max_chunks=0,
        model_name=None,
    )

    assert score is not None
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
