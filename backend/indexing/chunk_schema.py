"""
indexing/chunk_schema.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
─────────────────────
Every piece of code we parse — a function, a method, a class — gets turned into
a "chunk" before it is embedded and stored in Qdrant. A chunk is the atomic unit
of this entire system: it is what gets embedded, indexed, retrieved, re-ranked,
and finally shown to the user as a source citation.

Without a strict schema, different parts of the pipeline would attach different
fields to a chunk, spell field names differently, or forget to include critical
metadata. That creates bugs that are hard to trace because they only surface at
query time, not at ingestion time.

This file defines the single source of truth for what a chunk looks like.
Every layer — parsing, embedding, indexing, retrieval, generation — speaks the
same language because they all import from here.

HOW IT FITS INTO THE PIPELINE
───────────────────────────────
    parsing/entity_extractor.py
        │  produces raw dicts (function name, lines, body, etc.)
        │
        ▼
    indexing/metadata_builder.py
        │  enriches the raw dict and constructs a CodeChunk object
        │
        ▼
    embeddings/code_embedder.py
        │  receives a CodeChunk, embeds chunk.embeddable_text
        │  writes the vector back into chunk.semantic_vector
        │
        ▼
    indexing/qdrant_client.py
        │  calls chunk.to_qdrant_payload() to get the dict Qdrant stores
        │  stores (chunk.chunk_id, vectors, payload) as a Qdrant point
        │
        ▼
    retrieval/semantic_retriever.py
        │  receives raw Qdrant hits, calls CodeChunk.from_qdrant_payload()
        │  to reconstruct typed objects for the generation layer
        │
        ▼
    generation/generator.py
           reads chunk.citation_string() for source references in the answer

DESIGN DECISIONS
─────────────────
- Pydantic v2 is used for validation. It raises at object-construction time
  if a required field is missing or the wrong type — failing fast at ingestion
  rather than silently producing broken data.

- Vectors are NOT stored inside the Pydantic model permanently. They are
  attached temporarily during the embedding step and then extracted for Qdrant.
  Keeping large numpy arrays inside Pydantic objects causes serialization
  problems and wastes memory. The Optional[List[float]] fields exist only as
  a staging area during the ingestion pipeline.

- chunk_id is a deterministic hash of (repo_url + file_path + entity_name +
  start_line). This means re-ingesting the same file produces the same IDs,
  which lets Qdrant upsert (overwrite) rather than create duplicates.

- ChunkType is an enum rather than a plain string so that the query classifier
  and retrieval layer can do type-safe comparisons without worrying about
  typos like "Funcion" vs "Function".
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    """
    The kind of code entity this chunk represents.

    Using str as the mixin means ChunkType.FUNCTION == "function" is True,
    which simplifies JSON serialization and Qdrant payload filtering.

    Why these four types?
    - FUNCTION: The primary retrieval unit. Most queries resolve to a function.
    - METHOD: A function inside a class. Kept separate because methods carry
      class context (the class name, inheritance) that standalone functions don't.
    - CLASS: The container entity. Retrieved when a query is about a class's
      overall responsibility, not a specific method inside it.
    - MODULE: Represents an entire file. Used for summarization queries like
      "explain what auth.py does" where no single function is the answer.
    """
    FUNCTION = "function"
    METHOD   = "method"
    CLASS    = "class"
    MODULE   = "module"


class Language(str, Enum):
    """
    Programming language of the source file.

    Stored in the chunk payload so that retrieval can be filtered by language.
    Example: a query like "find all async functions" makes more sense scoped
    to Python/JS/TS than accidentally pulling Java results.
    """
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA       = "java"
    GO         = "go"
    UNKNOWN    = "unknown"


# ── Core Schema ───────────────────────────────────────────────────────────────

class CodeChunk(BaseModel):
    """
    The atomic unit of CodeSense's indexing and retrieval system.

    One CodeChunk = one parsed code entity (function / method / class / module).
    It carries three categories of information:

    1. IDENTITY — what uniquely identifies this chunk (chunk_id, repo, file, name)
    2. CONTENT  — the actual source code and human-readable text for embedding
    3. METADATA — structural signals used for filtered search and ranking

    A chunk starts its life in metadata_builder.py, gets its vectors attached
    by the embedding layer, and is finally stored in Qdrant. At retrieval time
    it is reconstructed from Qdrant's payload via from_qdrant_payload().
    """

    # ── Identity Fields ───────────────────────────────────────────
    # These fields together uniquely identify a chunk across all indexed repos.

    chunk_id: str = Field(
        ...,
        description=(
            "Deterministic SHA-256 hash of (repo_url + file_path + entity_name "
            "+ start_line). Deterministic so re-ingestion produces the same ID, "
            "enabling Qdrant upserts instead of duplicate inserts."
        )
    )

    repo_url: str = Field(
        ...,
        description=(
            "Full GitHub URL of the repository. Example: "
            "'https://github.com/tiangolo/fastapi'. "
            "Used to scope all queries to a single repo."
        )
    )

    repo_owner: str = Field(
        ...,
        description="GitHub username or org that owns the repo. Example: 'tiangolo'."
    )

    repo_name: str = Field(
        ...,
        description="Repository name without the owner prefix. Example: 'fastapi'."
    )

    # ── Location Fields ───────────────────────────────────────────
    # Tells us exactly where in the codebase this chunk lives.

    file_path: str = Field(
        ...,
        description=(
            "Path to the source file relative to the repo root. "
            "Example: 'fastapi/routing.py'. "
            "Used in citation strings and the source viewer."
        )
    )

    language: Language = Field(
        ...,
        description=(
            "Programming language of the source file. "
            "Enables language-scoped retrieval filtering."
        )
    )

    start_line: int = Field(
        ...,
        ge=1,
        description=(
            "1-indexed line number where this entity begins in the source file. "
            "ge=1 means Pydantic rejects line 0 — tree-sitter uses 0-based lines "
            "internally, so metadata_builder.py adds +1 before constructing the chunk."
        )
    )

    end_line: int = Field(
        ...,
        ge=1,
        description=(
            "1-indexed line number where this entity ends. "
            "end_line - start_line + 1 gives the line count of this chunk."
        )
    )

    # ── Entity Fields ─────────────────────────────────────────────
    # Describe the code entity itself.

    chunk_type: ChunkType = Field(
        ...,
        description=(
            "Whether this is a function, method, class, or module-level chunk. "
            "Stored in Qdrant payload for filtered retrieval. "
            "Example: a query asking about 'class structure' filters to ChunkType.CLASS."
        )
    )

    entity_name: str = Field(
        ...,
        description=(
            "The name of the function, method, or class. "
            "For module-level chunks, this is the filename without extension. "
            "Example: 'authenticate_user', 'Router', 'routing'."
        )
    )

    class_name: Optional[str] = Field(
        default=None,
        description=(
            "For METHOD chunks only: the name of the class this method belongs to. "
            "None for standalone functions and top-level classes. "
            "Used to build fully qualified names like 'Router.add_api_route'."
        )
    )

    signature: Optional[str] = Field(
        default=None,
        description=(
            "The function/method signature as a string, extracted from the AST. "
            "Includes parameter names and type annotations where present. "
            "Example: 'def authenticate_user(token: str, db: Session) -> User'. "
            "Stored separately from body because it's the most query-relevant part."
        )
    )

    docstring: Optional[str] = Field(
        default=None,
        description=(
            "The docstring immediately following the function/class definition, "
            "if present. Included in the embeddable text because it is the "
            "clearest natural-language description of what the entity does."
        )
    )

    # ── Source Content ────────────────────────────────────────────
    # The actual source code. Stored in full for the source viewer.

    source_code: str = Field(
        ...,
        description=(
            "The complete source code of this entity, as extracted from the file. "
            "Not truncated. Shown verbatim in the source viewer panel when the "
            "user clicks a citation. Never paraphrased — always the real code."
        )
    )

    # ── Structural Metadata ───────────────────────────────────────
    # Computed from the AST and call graph. Used for ranking and filtering.

    cyclomatic_complexity: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "McCabe cyclomatic complexity of this function/method. "
            "Computed from the AST by counting decision points (if, for, while, "
            "except, and, or, etc.) and adding 1. "
            "Higher = more complex = more likely to be a source of bugs. "
            "Stored in Qdrant payload so queries like 'find complex functions' "
            "can filter by complexity > threshold. "
            "None for CLASS and MODULE chunks (complexity is function-level)."
        )
    )

    in_degree: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Number of other functions that call this function, from the call graph. "
            "High in-degree = widely used = high blast radius if changed. "
            "Used by impact_analysis.py to rank the risk of modifying a function. "
            "None until the call graph is built (after all files in the repo are parsed)."
        )
    )

    out_degree: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Number of functions this function calls, from the call graph. "
            "High out_degree = many dependencies = many things to check when debugging. "
            "None until the call graph is built."
        )
    )

    imports: List[str] = Field(
        default_factory=list,
        description=(
            "For MODULE chunks: the list of module paths imported by this file. "
            "Example: ['fastapi', 'sqlalchemy.orm', '.dependencies']. "
            "Used by the graph retriever to find files that depend on a given module."
        )
    )

    # ── Temporal Metadata ─────────────────────────────────────────

    last_modified_commit: Optional[str] = Field(
        default=None,
        description=(
            "The git commit SHA of the last commit that touched this file. "
            "Used to detect stale embeddings on re-ingestion: if the commit hash "
            "in Redis cache matches this field, we skip re-embedding."
        )
    )

    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description=(
            "UTC timestamp of when this chunk was ingested. "
            "Useful for debugging stale data and for sorting results by recency."
        )
    )

    # ── Embedding Staging Fields ──────────────────────────────────
    # These are NOT stored in Qdrant. They are populated by the embedding
    # layer and then extracted by qdrant_client.py before the chunk object
    # is discarded. They live here so the pipeline can pass one object
    # through instead of passing (chunk, semantic_vector, structural_vector)
    # as three separate things everywhere.

    semantic_vector: Optional[List[float]] = Field(
        default=None,
        exclude=True,   # excluded from model_dump() / JSON serialization
        description=(
            "768-dimensional CodeBERT embedding of embeddable_text. "
            "Populated by embeddings/code_embedder.py. "
            "Excluded from serialization — extracted by qdrant_client.py "
            "before the chunk is stored, then discarded."
        )
    )

    structural_vector: Optional[List[float]] = Field(
        default=None,
        exclude=True,   # excluded from model_dump() / JSON serialization
        description=(
            "128-dimensional node2vec embedding from the call graph. "
            "Populated by embeddings/graph_embedder.py after the call graph "
            "is fully built for the repo. "
            "Excluded from serialization — same reason as semantic_vector."
        )
    )

    # ── Validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_line_range(self) -> "CodeChunk":
        """
        Ensures end_line >= start_line.

        Why: tree-sitter returns byte offsets that are converted to line numbers
        by entity_extractor.py. If that conversion has a bug, this validator
        catches it at chunk construction time rather than storing corrupt data
        in Qdrant. A chunk where start_line > end_line is physically impossible.
        """
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line}) "
                f"for entity '{self.entity_name}' in '{self.file_path}'."
            )
        return self

    @model_validator(mode="after")
    def validate_method_has_class(self) -> "CodeChunk":
        """
        Ensures METHOD chunks always have a class_name.

        Why: A method without a class is a data integrity bug. It means the
        entity_extractor incorrectly typed something as METHOD. Catching this
        here prevents confusing 'None.method_name' citation strings downstream.
        """
        if self.chunk_type == ChunkType.METHOD and not self.class_name:
            raise ValueError(
                f"ChunkType.METHOD requires class_name to be set, "
                f"but got None for entity '{self.entity_name}' in '{self.file_path}'."
            )
        return self

    # ── Computed Properties ───────────────────────────────────────

    @computed_field
    @property
    def line_count(self) -> int:
        """
        Number of lines this entity spans.

        Why a computed field: this is always derivable from start_line and
        end_line. Storing it separately would be redundant and could go out
        of sync. Pydantic's computed_field includes it in model_dump() output
        and Qdrant payload automatically.

        Used in the generation layer to decide whether to show the full source
        or truncate it in the answer (functions > 80 lines get truncated to
        the signature + docstring in the response, but full source is still
        available in the source viewer).
        """
        return self.end_line - self.start_line + 1

    @computed_field
    @property
    def fully_qualified_name(self) -> str:
        """
        Human-readable fully qualified name of this entity.

        Examples:
            FUNCTION → "authenticate_user"
            METHOD   → "Router.add_api_route"
            CLASS    → "Router"
            MODULE   → "fastapi/routing"

        Used in citation strings and as the display name in the source viewer.
        Keeps the generation layer from having to re-derive this from raw fields.
        """
        if self.chunk_type == ChunkType.METHOD and self.class_name:
            return f"{self.class_name}.{self.entity_name}"
        if self.chunk_type == ChunkType.MODULE:
            # Return the file path without extension for module display
            from pathlib import Path
            return str(Path(self.file_path).with_suffix(""))
        return self.entity_name

    @computed_field
    @property
    def embeddable_text(self) -> str:
        """
        The text string that gets passed to the embedding model.

        Why this specific construction:
        - Signature first: the most semantically dense part. The model sees
          the function name, parameter names, and return type before anything.
        - Docstring second: natural language description. CodeBERT was trained
          on code + docstring pairs — including the docstring significantly
          improves embedding quality for functions that have one.
        - Source code last: the full implementation, truncated to 400 chars
          to stay within CodeBERT's 512-token limit after tokenization.
          We keep a suffix to capture return statements that often appear last.

        This ordering is deliberate — the model attends more to earlier tokens.
        """
        parts = []

        # Include the signature if available; fall back to just the entity name
        if self.signature:
            parts.append(self.signature)
        else:
            parts.append(self.entity_name)

        # Include docstring if present — highest signal for semantic meaning
        if self.docstring:
            parts.append(self.docstring.strip())

        # Include a truncated version of the source code body.
        # 1500 chars ≈ 375 tokens, leaving ~135 tokens of headroom for
        # signature + docstring within the embedding model's 512-token
        # limit (see embeddings/code_embedder.py MAX_TOKEN_LENGTH) — safely
        # conservative for typical signature/docstring lengths, and the
        # tokenizer truncates further on its own for any chunk that still
        # overflows rather than erroring.
        #
        # WHY THIS WAS RAISED FROM 400 CHARS (~100 tokens):
        # A 400-char window is short enough to systematically bias ranking
        # toward whichever of two similar files happens to have its most
        # relevant content earliest — observed directly: two files with
        # near-identical MongoDB connection setup (a real app's main.py,
        # and a one-off migration script new.py) ranked the throwaway
        # script's module chunk higher for "where is the data stored?",
        # purely because its connection code sat in the first 400 characters
        # while the real app's was pushed slightly later by more imports.
        # The LLM then confidently described the placeholder/example
        # connection string from the wrong file as if it were the real one.
        # This isn't a one-repo quirk — MODULE chunks in particular (see
        # MetadataBuilder.build_module_chunk) can be many KB of full file
        # content, so a short truncation window means most of any
        # meaningfully-sized file's module chunk is invisible to embedding
        # regardless of which file it is.
        code_preview = self.source_code[:1500]
        if len(self.source_code) > 1500:
            # Keep a suffix to capture return types and final logic that a
            # pure head-truncation would otherwise always miss.
            code_preview += " ... " + self.source_code[-150:]
        parts.append(code_preview)

        return "\n".join(parts)

    # ── Serialization Helpers ─────────────────────────────────────

    def citation_string(self) -> str:
        """
        Produces a compact citation string for use in generated answers.

        Format: "file_path :: FullyQualifiedName (lines start–end)"
        Example: "fastapi/routing.py :: Router.add_api_route (lines 412–447)"

        Why: The generation layer is instructed to cite every claim with a
        specific file and line range. Using a standardized format means the
        frontend can parse it reliably to build source viewer links.
        """
        return (
            f"{self.file_path} :: {self.fully_qualified_name} "
            f"(lines {self.start_line}–{self.end_line})"
        )

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """
        Serializes this chunk into the flat dict stored as Qdrant point payload.

        Why a dedicated method instead of model_dump():
        - Qdrant payloads must be flat dicts of JSON-serializable primitives.
          model_dump() returns nested objects and datetime instances that Qdrant
          cannot store directly.
        - We explicitly control which fields go into the payload. The vectors
          (semantic_vector, structural_vector) are excluded here because Qdrant
          stores them separately in the vector index, not in the payload.
        - datetime is converted to ISO 8601 string.
        - Enum values are converted to their string values for Qdrant filtering.

        The payload is what gets returned alongside search results, so every
        field here is something the retrieval or generation layer will read.
        """
        return {
            # Identity
            "chunk_id":              self.chunk_id,
            "repo_url":              self.repo_url,
            "repo_owner":            self.repo_owner,
            "repo_name":             self.repo_name,
            # Location
            "file_path":             self.file_path,
            "language":              self.language.value,
            "start_line":            self.start_line,
            "end_line":              self.end_line,
            "line_count":            self.line_count,
            # Entity
            "chunk_type":            self.chunk_type.value,
            "entity_name":           self.entity_name,
            "class_name":            self.class_name,
            "fully_qualified_name":  self.fully_qualified_name,
            "signature":             self.signature,
            "docstring":             self.docstring,
            "source_code":           self.source_code,
            # Structural metadata
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "in_degree":             self.in_degree,
            "out_degree":            self.out_degree,
            "imports":               self.imports,
            # Temporal
            "last_modified_commit":  self.last_modified_commit,
            "ingested_at":           self.ingested_at.isoformat(),
        }

    @classmethod
    def from_qdrant_payload(cls, payload: Dict[str, Any]) -> "CodeChunk":
        """
        Reconstructs a CodeChunk from a Qdrant point payload.

        Why this classmethod:
        The retrieval layer receives raw Qdrant ScoredPoint objects. Those have
        a .payload dict, not a CodeChunk. This method bridges that gap so the
        retrieval and generation layers always work with typed CodeChunk objects,
        never raw dicts.

        Note: semantic_vector and structural_vector are NOT restored here — they
        are not stored in the payload (see to_qdrant_payload). The reconstructed
        chunk is for reading and citation purposes only, not for re-embedding.

        Args:
            payload: The dict from a Qdrant ScoredPoint.payload field.

        Returns:
            A CodeChunk instance with all fields populated from the payload.
        """
        return cls(
            chunk_id              = payload["chunk_id"],
            repo_url              = payload["repo_url"],
            repo_owner            = payload["repo_owner"],
            repo_name             = payload["repo_name"],
            file_path             = payload["file_path"],
            language              = Language(payload["language"]),
            start_line            = payload["start_line"],
            end_line              = payload["end_line"],
            chunk_type            = ChunkType(payload["chunk_type"]),
            entity_name           = payload["entity_name"],
            class_name            = payload.get("class_name"),
            signature             = payload.get("signature"),
            docstring             = payload.get("docstring"),
            source_code           = payload["source_code"],
            cyclomatic_complexity = payload.get("cyclomatic_complexity"),
            in_degree             = payload.get("in_degree"),
            out_degree            = payload.get("out_degree"),
            imports               = payload.get("imports", []),
            last_modified_commit  = payload.get("last_modified_commit"),
            ingested_at           = datetime.fromisoformat(payload["ingested_at"]),
        )


# ── ID Generation Utility ─────────────────────────────────────────────────────

def generate_chunk_id(
    repo_url: str,
    file_path: str,
    entity_name: str,
    start_line: int
) -> str:
    """
    Generates a deterministic, stable chunk ID from four identifying fields.

    WHY DETERMINISTIC:
    If we used random UUIDs, re-ingesting the same repo after a small code
    change would create duplicate entries in Qdrant for all the unchanged
    functions — one with the old UUID, one with the new. The collection would
    grow without bound, retrieval would return duplicates, and we'd need a
    separate cleanup job.

    With a deterministic ID based on (repo, file, name, line), re-ingesting
    produces the exact same ID for unchanged code. Qdrant's upsert operation
    then simply overwrites the old point with identical data — no duplicates,
    no cleanup needed.

    WHY SHA-256 TRUNCATED TO 16 CHARS:
    - SHA-256 gives us a collision-resistant hash (probability of collision
      across millions of functions is negligible).
    - We truncate to 16 hex characters (64 bits of entropy) because Qdrant
      point IDs have a size limit and 16 chars is more than sufficient for
      the scale of any single codebase.

    Args:
        repo_url:    Full GitHub URL of the repository.
        file_path:   Relative path to the file within the repo.
        entity_name: Name of the function, method, or class.
        start_line:  1-indexed starting line number.

    Returns:
        A 16-character lowercase hex string. Example: "a3f2c1d8e9b74052"
    """
    raw = f"{repo_url}::{file_path}::{entity_name}::{start_line}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]