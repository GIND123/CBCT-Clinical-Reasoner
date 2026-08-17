"""Dataset discovery, preprocessing, splits, and the report corpus.

``cbct_reasoner.data.dataset`` is not re-exported here because it imports torch;
the Grand Challenge inference container installs no training dependencies.
"""

from cbct_reasoner.data.corpus import (
    CorpusEntry,
    build_corpus,
    corpus_index,
    load_corpus,
    save_corpus,
    select_reference,
)
from cbct_reasoner.data.discovery import (
    LayoutReport,
    discover_cases,
    infer_center,
    inspect_layout,
    normalize_report,
    read_manifest,
    resolve_root,
    write_manifest,
)
from cbct_reasoner.data.preprocess import (
    META_DIM,
    VolumeMeta,
    cache_paths,
    is_cached,
    preprocess_volume,
    read_cache,
    write_cache,
)
from cbct_reasoner.data.splits import Fold, SplitPlan, build_splits, leave_one_center_out

__all__ = [
    "META_DIM",
    "CorpusEntry",
    "Fold",
    "LayoutReport",
    "SplitPlan",
    "VolumeMeta",
    "build_corpus",
    "build_splits",
    "cache_paths",
    "corpus_index",
    "discover_cases",
    "infer_center",
    "inspect_layout",
    "is_cached",
    "leave_one_center_out",
    "load_corpus",
    "normalize_report",
    "preprocess_volume",
    "read_cache",
    "read_manifest",
    "resolve_root",
    "save_corpus",
    "select_reference",
    "write_cache",
    "write_manifest",
]
