import re

from dataclasses import dataclass
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = "corpus/northstar-travel-policy.md"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


def load_markdown_sections(path: Path) -> list[Chunk]:
    content = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", content)[1:]
    chunks = []

    for section in sections:
        lines = section.strip().splitlines()
        heading = lines[0].removesuffix(":").strip()
        body = "\n".join(lines[1:]).strip()
        chunk_id = heading.lower().replace(" ", "-")
        chunks.append(Chunk(chunk_id=chunk_id, text=body))

    if not chunks:
        raise ValueError("Unable to parse the document into chunks")

    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()

        documents = [
            f"{chunk.chunk_id.replace('-', ' ')} {chunk.text}"
            for chunk in chunks
        ]

        self.chunk_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.chunk_vectors,
        )[0]

        top_indexes = scores.argsort()[::-1][:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
            )
            for index in top_indexes
        ]

chunks = load_markdown_sections(
    Path(POLICY_PATH)
)

retriever = TfidfRetriever(chunks)
