"""BM25 text processing service for optimizing text for keyword search."""

import re
import unicodedata
from typing import List, Tuple


STOPWORDS_ES = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o",
    "este","sí","porque","esta","entre","cuando","muy","sin","sobre","también",
    "me","hasta","hay","donde","quien","desde","todo","nos","durante","todos",
    "uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto",
    "mí","antes","algunos","qué","unos","yo","otro","otras","otra","él","tanto",
    "esa","estos","mucho","quienes","nada","muchos","cual","poco","ella","estar",
    "estas","algunas","algo","nosotros","mi","mis","tú","te","ti","tu","tus",
    "ellas","nosotras","vosotros","vosotras","os","mío","mía","míos","mías"
}


class BM25TextProcessor:
    """Service for processing text to optimize for BM25 keyword search."""

    @staticmethod
    def make_bm25_text(text: str) -> str:
        """
        Create optimized BM25 text by extracting keywords and removing stopwords.

        Process:
        1. Remove markdown syntax (#, *, _, -, etc.)
        2. Convert to lowercase
        3. Remove accents/tildes
        4. Extract only word tokens
        5. Filter out Spanish stopwords
        6. Return space-separated keywords
        """
        clean = re.sub(r'[#*_\-\n\r\t]+', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        clean = clean.lower()

        clean = unicodedata.normalize('NFD', clean)
        clean = ''.join(c for c in clean if unicodedata.category(c) != 'Mn')

        tokens = re.findall(r'\w+', clean)

        filtered = [
            token for token in tokens
            if token not in STOPWORDS_ES and len(token) > 2
        ]

        return " ".join(filtered)

    @staticmethod
    def extract_section_info(text: str) -> Tuple[str, List[str], str]:
        """Extract section title, section path, and BM25 text from chunk text."""
        lines = text.split('\n')
        section_title = ""
        section_path = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                heading_text = stripped.lstrip('#').strip()

                while len(section_path) >= level:
                    section_path.pop()
                section_path.append(heading_text)

                section_title = heading_text

        bm25_text = BM25TextProcessor.make_bm25_text(text)

        return section_title, section_path, bm25_text
