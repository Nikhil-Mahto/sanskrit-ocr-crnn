from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SanskritCharset:
    blank_token: str = "<BLANK>"
    pad_token: str = "<PAD>"
    space_token: str = " "
    characters: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.characters:
            self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.characters)}
            return

        vowels = list("अआइईउऊऋॠऌॡएऐओऔ")
        vowel_signs = list("ािीुूृॄॢॣेैोौ")
        consonants = [
            "क",
            "ख",
            "ग",
            "घ",
            "ङ",
            "च",
            "छ",
            "ज",
            "झ",
            "ञ",
            "ट",
            "ठ",
            "ड",
            "ढ",
            "ण",
            "त",
            "थ",
            "द",
            "ध",
            "न",
            "प",
            "फ",
            "ब",
            "भ",
            "म",
            "य",
            "र",
            "ल",
            "व",
            "श",
            "ष",
            "स",
            "ह",
            "ळ",
            "क्ष",
            "ज्ञ",
            "श्र",
        ]
        digits = list("०१२३४५६७८९")
        specials = ["ं", "ः", "ँ", "ऽ", "।", "॥", "ॐ", "्", "-", ",", ".", "?"]

        ordered = [self.blank_token, self.pad_token, self.space_token]
        seen = set(ordered)
        for token in vowels + vowel_signs + consonants + digits + specials:
            if token not in seen:
                ordered.append(token)
                seen.add(token)

        self.characters = ordered
        self.char_to_idx: Dict[str, int] = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self.characters)}

    @property
    def blank_index(self) -> int:
        return self.char_to_idx[self.blank_token]

    @property
    def pad_index(self) -> int:
        return self.char_to_idx[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        chars: List[str] = []
        for idx in indices:
            char = self.idx_to_char.get(idx, "")
            if remove_special and char in {self.blank_token, self.pad_token}:
                continue
            chars.append(char)
        return "".join(chars)
