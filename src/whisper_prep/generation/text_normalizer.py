import re
import unicodedata

APOS_TABLE = str.maketrans(
    {
        "’": "'",  # curly
        "‘": "'",  # curly left (rare)
        "ˊ": "'",  # modifier acute
        "ˋ": "'",  # modifier grave (folds to ` with NFKC)
        "ʹ": "'",  # modifier prime
        "＇": "'",  # full-width
        "`": "'",  # optional: treat back-tick as apostrophe
    }
)


def normalize_abbrv(text):
    abbreviation_map = {
        "Std.": "Stunden",
        "Min.": "Minuten",
        "Sek.": "Sekunden",
        "Jr.": "Junior",
        "Tg.": "Tage",
        "m ü. M.": "Meter über Meer",
        "u.a.": "unter anderem",
        "sog.": "sogenannte",
        "bzw.": "beziehungsweise",
        "ca.": "circa",
        "z.B.": "zum Beispiel",
        "d.h.": "das heisst",
        "D.h": "das heisst",
        "etc.": "et cetera",
        "z.T.": "zum Teil",
        "i.d.R.": "in der Regel",
        "u.v.m.": "und vieles mehr",
        "Nr.": "Nummer",
        "b.a.w.": "bis auf weiteres",
        "n.Chr.": "nach Christus",
        "v.Chr.": "vor Christus",
        "ggf.": "gegebenenfalls",
        "evtl.": "eventuell",
        "u.U.": "unter Umständen",
        "v.a.": "vor allem",
        "vgl.": "vergleiche",
        "inkl.": "inklusive",
        "zzgl.": "zuzüglich",
        "u.s.w.": "und so weiter",
        "bzgl.": "bezüglich",
        "s.a.": "siehe auch",
        "bspw.": "beispielsweise",
        "Hbf.": "Hauptbahnhof",
        "Dr.": "Doktor",
        "Dipl.": "Diplom",
        "Univ.": "Universität",
        "km/h": "Kilometer pro Stunde",
        "m/s": "Meter pro Sekunde",
        "Gebr.": "Gebrüder",
        "Mio.": "Millionen",
        "Mia.": "Milliarden",
        "ähm": "",
        "äh": "",
        "hm": "",
        "hm...": "",
        "Öhh...": "",
        "Öhh": "",
    }
    for abbr, full in abbreviation_map.items():
        pattern = re.compile(rf"(^|\s)({re.escape(abbr)})(?=\s|$)", re.IGNORECASE)
        text = pattern.sub(lambda m: m.group(1) + full, text)
    return text


def standardize_text(text: str) -> str:
    """
    Clean, normalize, and remove specific symbols from text. This function:
    - Replaces certain Unicode characters and normalizes similar symbols to ASCII.
    - Removes specific unwanted symbols and quotation marks.
    - Collapses multiple spaces into a single space.
    """
    # Define replacements for specific Unicode and typographic characters
    REPLACEMENTS = {
        # –––– Invisible / spacing chars ––––––––––––––––––––––––––––––––––––––––
        "\u00a0": " ",  # NO-BREAK SPACE → plain space
        "\u202f": " ",  # NARROW NO-BREAK SPACE → space
        "\u200b": "",  # ZERO-WIDTH SPACE
        "\u200c": "",  # ZERO-WIDTH NON-JOINER
        "\u200d": "",  # ZERO-WIDTH JOINER
        "\u2060": "",  # WORD-JOINER
        # –––– Quotes & apostrophes –––––––––––––––––––––––––––––––––––––––––––––
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "„": '"',
        "‟": '"',
        # –––– Dashes & hyphens ––––––––––––––––––––––––––––––––––––––––––––––––
        "\u2011": "-",  # NON-BREAKING HYPHEN
        "–": "-",  # EN-DASH
        "—": "-",  # EM-DASH
        "−": "-",  # MINUS SIGN
        "--": "-",  # double NB-hyphen → single
        # –––– Ellipsis ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        "…": "...",
        # –––– Ligatures & spelling variants –––––––––––––––––––––––––––––––––––
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ß": "ss",
        # –––– Bullets & list marks ––––––––––––––––––––––––––––––––––––––––––––
        "•": "-",
        "·": "-",
        # –––– Accidental markup leftovers ––––––––––––––––––––––––––––––––––––
        "<i>": "",
        "</i>": "",
        "<b>": "",
        "</b>": "",
        # –––– Catch-alls you already had ––––––––––––––––––––––––––––––––––––––
        "\u00ad": "",  # SOFT HYPHEN
        "- -": "-",  # double hyphen → single
        "/, ": "",  # "Renter/, innen" → "Renterinnen"
    }

    # Apply replacements
    for original, replacement in REPLACEMENTS.items():
        text = text.replace(original, replacement)

    # Some other wierd spaces
    text = re.sub(r"[\u00A0\u202F\u200B\u200C\u200D\u2060]+", " ", text)
    return text


def normalize_capitalization(text):
    """Preserve names, acronyms and compounds; case cannot be inferred safely."""
    return " ".join(text.split())


ELLIPSIS = r"(?:\.{3}|…)"  # ASCII or Unicode


def normalize_triple_dots(text: str) -> str:
    # Normalize bracketed ellipsis first: [...] or […] -> ...
    text = re.sub(r"\[\s*(?:\.{3,}|…+)\s*\]", "...", text)

    # Normalize any triple-dot/ellipsis run to ASCII triple dots.
    # Examples: "..." -> "...", "......" -> "...", "……" -> "..."
    text = re.sub(r"(?:\.{3,}|…+)", "...", text)

    # ────────────────────────────────────────────────────────────────────────
    # 2️⃣  tidy punctuation spacing (same as before, but ellipses are gone)
    # ────────────────────────────────────────────────────────────────────────
    text = (
        text.replace(".,", ".")
        .replace("  ", " ")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" . ", ". ")
        .replace(". <", ".<")
        .replace("! <", "!<")
        .replace("? <", "?<")
        .replace(", <", ",<")
    )

    return text.strip()


def remove_bracketed_text(text):
    # Remove text enclosed in square brackets or parentheses, including the brackets
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\*.*?\*", "", text)
    # Strip <font> tags, preserving inner text
    text = re.sub(r"</?font[^>]*>", "", text, flags=re.I)
    return text.strip()


def tokenize(text):
    # treat "..." as a standalone token
    return re.findall(r"\.\.\.|[^\s]+", text)


def collapse_ngrams(tokens, max_n=3):
    out, i = [], 0
    while i < len(tokens):
        # 1️⃣  Collapse runs of a single repeated token
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        if j - i > 1:  # found a run ≥ 2
            out.append(tokens[i])
            i = j
            continue  # restart loop

        # 2️⃣  Collapse repeated n-grams of length 2-max_n
        replaced = False
        for n in range(max_n, 1, -1):  # skip n=1 (already handled)
            if i + 2 * n <= len(tokens) and tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                k = i + n
                while k + n <= len(tokens) and tokens[i : i + n] == tokens[k : k + n]:
                    k += n
                out.extend(tokens[i : i + n])  # keep one copy
                i = k
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return out


def collapse_text(text, max_n=3):
    return " ".join(collapse_ngrams(tokenize(text), max_n))


class SwissNumberConverter:
    def __init__(self):
        self.currency_map = {
            "$": "Dollar",
            "€": "Euro",
            "£": "Pfund",
            "¥": "Yen",
            "₹": "Rupie",
            "₽": "Rubel",
            "₩": "Won",
            "₫": "Dong",
            "₣": "Franken",
            "₦": "Naira",
            "₴": "Hrywnja",
            "₲": "Guarani",
            "₵": "Cedi",
            "฿": "Baht",
            "₺": "Türkische Lira",
            "₼": "Manat",
            "₪": "Schekel",
            "₭": "Kip",
            "₨": "Rupie",
            "₱": "Peso",
            "₳": "Austral",
            "₡": "Colon",
            "₢": "Cruzeiro",
            "₥": "Mill",
            "₧": "Peseta",
            "₮": "Tugrik",
            "₯": "Drachme",
            "₰": "Pfennig",
            "₷": "Speciedaler",
            "%": "Prozent",
            "Fr.": "Franken",
            "CHF": "Franken",
            "USD": "Dollar",
            "EUR": "Euro",
            "GBP": "Pfund",
            "JPY": "Yen",
            "CNY": "Yuan",
            "INR": "Rupie",
            "RUB": "Rubel",
            "KRW": "Won",
            "VND": "Dong",
            "NGN": "Naira",
            "UAH": "Hrywnja",
            "GHS": "Cedi",
            "THB": "Baht",
            "TRY": "Türkische Lira",
            "AZN": "Manat",
            "ILS": "Schekel",
            "LAK": "Kip",
            "PKR": "Rupie",
            "PHP": "Peso",
            "ARS": "Peso",
            "CRC": "Colon",
            "BRL": "Real",
            "CLP": "Peso",
            "MXN": "Peso",
            "SGD": "Dollar",
            "HKD": "Dollar",
        }

    def combine_numbers(self, text):
        # collapse ISO thin spaces, NBSP, etc. to plain space first
        text = re.sub(r"[\u00A0\u202F]", " ", text)

        # remove ONE space between digit groups of exactly three digits
        text = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", text)

        # remove apostrophes used the same way (Switzerland)
        text = re.sub(r"(?<=\d)'(?=\d{3}\b)", "", text)
        return text

    def replace_komma_w_dot(self, text):
        # Normalizes a comma to a dot for numbers
        # For example, "1,234" becomes "1.234"
        return re.sub(r"(?<=\d),(?=\d)", ".", text)

    def replace_currencies(self, text):
        # Expand currency codes before symbols so a following '%' is still
        # recognizable when deciding whether a number is monetary.
        for symbol, word in sorted(
            self.currency_map.items(), key=lambda item: not any(c.isalpha() for c in item[0])
        ):
            if symbol[0].isalpha():
                # Currency abbreviations require an adjacent amount. "Fr. Müller"
                # and programming-language acronyms are not monetary expressions.
                code = re.escape(symbol)
                text = re.sub(rf"(?<=\d)\s*{code}(?!\w)", lambda m: " " + word, text)
                # A percentage after a currency-like word (e.g. PHP 50%)
                # describes a proportion, not a monetary amount.
                text = re.sub(
                    rf"(?<!\w){code}\s*(?=\d+(?:[.,]\d+)?(?![\d.,]|\s*%))",
                    lambda m: word + " ",
                    text,
                )
            else:
                text = re.sub(re.escape(symbol), lambda m: " " + word + " ", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def convert_numbers(self, text):
        # Only explicitly grouped numbers and the Unicode fraction slash are
        # numerical notation. ASCII slashes can denote dates or opening hours.
        text = self.combine_numbers(text)

        def fraction(match):
            numerator, denominator = map(int, match.groups())
            return str(numerator / denominator) if denominator else match[0]

        text = re.sub(r"(\d+)⁄(\d+)", fraction, text)
        text = self.replace_currencies(text)
        text = self.replace_komma_w_dot(text)

        def group(match):
            integer, fraction = match[1], match[2] or ""
            if integer.startswith("0") or len(integer) <= 4:
                return match[0]
            return f"{int(integer):,}".replace(",", "'") + fraction

        return re.sub(r"(?<![\w.])(\d+)(\.\d+)?(?![\w.])", group, text)

    def convert(self, text):
        """Alias for convert_numbers to support shorthand method name in tests."""
        return self.convert_numbers(text)


def normalize_text(text, language=None):
    """Conservative Unicode/whitespace cleanup; Swiss expansion is explicit.

    Use language="de-CH" only for Swiss German training data. Other languages
    retain their spelling, bracketed speech, numeric identifiers and symbols.
    """
    text = unicodedata.normalize("NFC", text).translate(APOS_TABLE)
    text = re.sub(r"</?(?:font|i|b|u)(?:\s[^>]*)?>", "", text, flags=re.I)
    text = normalize_triple_dots(text)
    if language == "de-CH":
        text = normalize_abbrv(text)
        text = standardize_text(text)
        text = SwissNumberConverter().convert_numbers(text)
    elif language not in (None, "basic"):
        raise ValueError("normalizer_language must be 'basic' or 'de-CH'")
    return " ".join(text.split())
