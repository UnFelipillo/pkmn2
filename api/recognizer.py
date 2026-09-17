"""
recognizer.py
═════════════
Motor de reconocimiento de cartas Pokémon usando hashes perceptuales (pHash).

Flujo:
    imagen recortada (warpPerspective)
        │
        ▼
    preprocesar (crop arte, normalizar)
        │
        ▼
    calcular pHash (64 bits)
        │
        ▼
    buscar vecino más cercano en índice
    (distancia de Hamming < umbral)
        │
        ▼
    devolver carta + confianza

Velocidad: ~2-5ms por consulta (búsqueda en RAM).
"""

import json
import time
import base64
import numpy as np
from pathlib import Path
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass

try:
    from PIL import Image
    import imagehash
    HAS_HASH = True
except ImportError:
    HAS_HASH = False

DATA_DIR = Path(__file__).parent / "data"


# ──────────────────────────────────────────
#  ESTRUCTURAS
# ──────────────────────────────────────────
@dataclass
class RecognitionResult:
    found:      bool
    card_id:    str       = ""
    name:       str       = ""
    set_name:   str       = ""
    set_id:     str       = ""
    number:     str       = ""
    rarity:     str       = ""
    types:      list      = None
    price_usd:  float     = None
    img_small:  str       = ""
    confidence: float     = 0.0
    hamming:    int       = 64
    method:     str       = ""
    time_ms:    float     = 0.0
    message:    str       = ""

    def to_dict(self) -> dict:
        return {
            "found":      self.found,
            "card_id":    self.card_id,
            "name":       self.name,
            "set_name":   self.set_name,
            "set_id":     self.set_id,
            "number":     self.number,
            "rarity":     self.rarity,
            "types":      self.types or [],
            "price_usd":  self.price_usd,
            "img_small":  self.img_small,
            "confidence": round(self.confidence, 3),
            "hamming":    self.hamming,
            "method":     self.method,
            "time_ms":    round(self.time_ms, 1),
            "message":    self.message,
        }


# ──────────────────────────────────────────
#  MOTOR PRINCIPAL
# ──────────────────────────────────────────
class CardRecognizer:
    """
    Carga el índice pHash en RAM y resuelve consultas
    usando distancia de Hamming.

    Parámetros:
        hash_size      : bits del pHash (8 → 64 bits, 16 → 256 bits)
        max_hamming    : distancia máxima para considerar match (0=igual, 64=nada)
        confidence_map : [(hamming, confianza), ...] — curva de confianza
    """

    CONFIDENCE_CURVE = [
        (0,  1.00),   # idéntico
        (5,  0.95),   # casi seguro
        (10, 0.85),
        (15, 0.70),
        (20, 0.50),   # dudoso
        (30, 0.20),
        (64, 0.00),
    ]

    def __init__(
        self,
        hash_size:   int = 8,
        max_hamming: int = 18,
    ):
        self.hash_size   = hash_size
        self.max_hamming = max_hamming
        self.ready       = False

        # Datos cargados en RAM
        self._phash_index: dict[str, str] = {}   # phash_hex → card_id
        self._id_index:    dict[str, dict] = {}   # card_id   → metadata
        self._phash_ints:  np.ndarray | None = None  # array de hashes como uint64
        self._phash_ids:   list[str] = []             # card_ids alineados con el array

        self._load()

    # ── CARGA ──────────────────────────────
    def _load(self):
        phash_path = DATA_DIR / "phash_index.json"
        id_path    = DATA_DIR / "id_index.json"

        if not phash_path.exists():
            print("⚠  data/phash_index.json no encontrado.")
            print("   Ejecuta: python build_library.py")
            return
        if not id_path.exists():
            print("⚠  data/id_index.json no encontrado.")
            return

        t0 = time.time()
        with open(phash_path) as f:
            self._phash_index = json.load(f)
        with open(id_path) as f:
            self._id_index = json.load(f)

        # Pre-convertir hashes a enteros uint64 para búsqueda vectorial rápida
        self._phash_ids  = list(self._phash_index.values())
        phash_hexes      = list(self._phash_index.keys())

        try:
            self._phash_ints = np.array(
                [int(h, 16) for h in phash_hexes],
                dtype=np.uint64
            )
        except Exception:
            self._phash_ints = None

        elapsed = (time.time() - t0) * 1000
        n = len(self._phash_index)
        print(f"✅ Biblioteca cargada: {n:,} cartas en {elapsed:.0f}ms")
        self.ready = True

    def is_ready(self) -> bool:
        return self.ready and len(self._phash_index) > 0

    def stats(self) -> dict:
        return {
            "ready":       self.ready,
            "total_cards": len(self._phash_index),
            "has_numpy":   self._phash_ints is not None,
            "hash_size":   self.hash_size,
        }

    # ── PREPROCESADO ───────────────────────
    def _preprocess(self, img_pil: "Image.Image") -> "Image.Image":
        """
        Normaliza la imagen antes de calcular el hash:
        - Recorta el 8% de cada borde (elimina sombras y bordes blancos)
        - Redimensiona a tamaño fijo
        - Convierte a RGB
        """
        w, h = img_pil.size
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        cropped  = img_pil.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        return cropped.convert("RGB")

    # ── HASH ───────────────────────────────
    def _compute_hash(self, img_pil: "Image.Image") -> "imagehash.ImageHash | None":
        if not HAS_HASH:
            return None
        try:
            processed = self._preprocess(img_pil)
            return imagehash.phash(processed, hash_size=self.hash_size)
        except Exception:
            return None

    def _hash_to_int(self, h: "imagehash.ImageHash") -> int:
        return int(str(h), 16)

    # ── CONFIANZA ──────────────────────────
    def _hamming_to_confidence(self, hamming: int) -> float:
        """Interpola linealmente la curva de confianza."""
        curve = self.CONFIDENCE_CURVE
        for i in range(len(curve) - 1):
            h0, c0 = curve[i]
            h1, c1 = curve[i + 1]
            if h0 <= hamming <= h1:
                t = (hamming - h0) / (h1 - h0)
                return c0 + t * (c1 - c0)
        return 0.0

    # ── BÚSQUEDA ───────────────────────────
    def _search_numpy(self, query_int: int) -> tuple[str, int]:
        """
        Búsqueda vectorial con NumPy: calcula distancia de Hamming
        contra todos los hashes en paralelo usando popcount de XOR.
        O(n) pero muy rápido en la práctica (~1-3ms para 10k hashes).
        """
        q = np.uint64(query_int)
        xor     = np.bitwise_xor(self._phash_ints, q)
        # popcount: número de bits en 1
        hamming = np.array([bin(int(x)).count('1') for x in xor], dtype=np.int32)
        idx     = int(np.argmin(hamming))
        return self._phash_ids[idx], int(hamming[idx])

    def _search_linear(self, query_hash: "imagehash.ImageHash") -> tuple[str, int]:
        """Búsqueda lineal fallback (sin NumPy optimizado)."""
        best_id  = ""
        best_ham = 64
        for hex_str, card_id in self._phash_index.items():
            try:
                stored = imagehash.hex_to_hash(hex_str)
                d      = query_hash - stored   # distancia de Hamming
                if d < best_ham:
                    best_ham = d
                    best_id  = card_id
            except Exception:
                continue
        return best_id, best_ham

    # ── RECONOCIMIENTO ─────────────────────
    def recognize(self, image_input) -> RecognitionResult:
        """
        Reconoce una carta Pokémon.

        Args:
            image_input: puede ser:
                - PIL.Image
                - bytes (JPEG/PNG raw)
                - str (base64)
                - numpy.ndarray (OpenCV BGR)

        Returns:
            RecognitionResult con todos los detalles
        """
        t0 = time.perf_counter()

        if not self.is_ready():
            return RecognitionResult(
                found=False,
                message="Biblioteca no cargada. Ejecuta: python build_library.py",
                time_ms=0
            )
        if not HAS_HASH:
            return RecognitionResult(
                found=False,
                message="Pillow/imagehash no instalado. pip install Pillow imagehash",
                time_ms=0
            )

        # Convertir input a PIL
        try:
            img_pil = self._to_pil(image_input)
        except Exception as e:
            return RecognitionResult(found=False, message=f"Error al decodificar imagen: {e}")

        # Calcular hash
        query_hash = self._compute_hash(img_pil)
        if query_hash is None:
            return RecognitionResult(found=False, message="No se pudo calcular hash de la imagen")

        query_int = self._hash_to_int(query_hash)

        # Búsqueda
        if self._phash_ints is not None:
            card_id, hamming = self._search_numpy(query_int)
            method = "numpy_vectorial"
        else:
            card_id, hamming = self._search_linear(query_hash)
            method = "linear_scan"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        confidence = self._hamming_to_confidence(hamming)

        if hamming > self.max_hamming:
            return RecognitionResult(
                found=False,
                hamming=hamming,
                confidence=confidence,
                method=method,
                time_ms=elapsed_ms,
                message=f"Sin match seguro (hamming={hamming} > umbral={self.max_hamming}). "
                        "Intenta con mejor iluminación o ángulo."
            )

        # Obtener metadatos
        meta = self._id_index.get(card_id, {})

        return RecognitionResult(
            found      = True,
            card_id    = card_id,
            name       = meta.get("name", ""),
            set_name   = meta.get("set_name", ""),
            set_id     = meta.get("set_id", ""),
            number     = meta.get("number", ""),
            rarity     = meta.get("rarity", ""),
            types      = meta.get("types", []),
            price_usd  = meta.get("price_usd"),
            img_small  = meta.get("img_small", ""),
            confidence = confidence,
            hamming    = hamming,
            method     = method,
            time_ms    = elapsed_ms,
            message    = "Carta identificada correctamente",
        )

    # ── MULTI-MATCH ────────────────────────
    def top_matches(self, image_input, n: int = 5) -> list[RecognitionResult]:
        """Devuelve los N mejores candidatos (útil para debugging)."""
        if not self.is_ready() or not HAS_HASH:
            return []

        try:
            img_pil    = self._to_pil(image_input)
            query_hash = self._compute_hash(img_pil)
            if not query_hash:
                return []
            query_int  = self._hash_to_int(query_hash)
        except Exception:
            return []

        # Calcular todas las distancias
        if self._phash_ints is not None:
            q       = np.uint64(query_int)
            xor     = np.bitwise_xor(self._phash_ints, q)
            hammings = np.array([bin(int(x)).count('1') for x in xor], dtype=np.int32)
            top_idx  = np.argsort(hammings)[:n]
            results  = []
            for idx in top_idx:
                cid  = self._phash_ids[int(idx)]
                ham  = int(hammings[idx])
                meta = self._id_index.get(cid, {})
                results.append(RecognitionResult(
                    found=ham <= self.max_hamming,
                    card_id=cid, name=meta.get("name",""),
                    set_name=meta.get("set_name",""), set_id=meta.get("set_id",""),
                    number=meta.get("number",""), rarity=meta.get("rarity",""),
                    types=meta.get("types",[]), price_usd=meta.get("price_usd"),
                    img_small=meta.get("img_small",""),
                    confidence=self._hamming_to_confidence(ham),
                    hamming=ham, method="numpy_vectorial",
                ))
            return results
        return []

    # ── HELPER ─────────────────────────────
    def _to_pil(self, img) -> "Image.Image":
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            import cv2
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        if isinstance(img, bytes):
            return Image.open(BytesIO(img))
        if isinstance(img, str):
            # base64
            if "," in img:
                img = img.split(",", 1)[1]
            data = base64.b64decode(img)
            return Image.open(BytesIO(data))
        raise ValueError(f"Tipo de imagen no soportado: {type(img)}")


# ──────────────────────────────────────────
#  SINGLETON (para no recargar en cada request)
# ──────────────────────────────────────────
_recognizer: CardRecognizer | None = None

def get_recognizer() -> CardRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = CardRecognizer()
    return _recognizer
