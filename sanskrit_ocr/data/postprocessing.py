from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional


@dataclass
class SanskritPostProcessor:
    dictionary_path: Optional[Path] = None
    _dictionary: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.dictionary_path and self.dictionary_path.exists():
            self._dictionary = [
                line.strip()
                for line in self.dictionary_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def remove_repeated_spaces(self, text: str) -> str:
        return " ".join(text.split())

    def dictionary_correct(self, text: str) -> str:
        if not self._dictionary:
            return text
        corrected: List[str] = []
        for token in text.split():
            matches = get_close_matches(token, self._dictionary, n=1, cutoff=0.8)
            corrected.append(matches[0] if matches else token)
        return " ".join(corrected)

    def clean(self, text: str) -> str:
        text = text.strip()
        text = self.remove_repeated_spaces(text)
        text = self.dictionary_correct(text)
        return text
