#!/usr/bin/env python3
"""
build_library.py
════════════════
Descarga el catálogo completo de pokemontcg.io y construye
una biblioteca local con hashes perceptuales (pHash) para
reconocimiento offline ultra-rápido.

Uso:
    python build_library.py                  # descarga todo (~18 000 cartas)
    python build_library.py --sets sv1 sv2   # solo sets específicos
    python build_library.py --limit 500      # solo las primeras 500 (prueba)
    python build_library.py --api-key TU_KEY # con API key (más rápido, sin rate limit)

Tiempo estimado:
    Sin imágenes (solo metadatos):  ~2 min
    Con imágenes + pHash (completo): ~25-40 min (18 000 cartas)
    Solo sets modernos (sv1-sv8):    ~5-8 min

Salida:
    data/
    ├── cards_meta.json      ← metadatos completos (nombre, set, número, precio)
    ├── phash_index.json     ← {phash_hex: card_id} para búsqueda O(1)
    └── images/              ← thumbnails 200px (opcional con --save-images)
"""

import os
import sys
import json
import time
import hashlib
import argparse
import requests
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
    import imagehash
    HAS_HASH = True
except ImportError:
    HAS_HASH = False
    print("⚠  Instala Pillow e imagehash para generar hashes:")
    print("   pip install Pillow imagehash")

# ──────────────────────────────────────────
TCGAPI   = "https://api.pokemontcg.io/v2"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR  = DATA_DIR / "images"

MODERN_SETS = [
    "sv1","sv2","sv3","sv3pt5","sv4","sv4pt5",
    "sv5","sv6","sv6pt5","sv7","sv8","sv8pt5",
    "swsh1","swsh2","swsh3","swsh4","swsh5","swsh6",
    "swsh7","swsh8","swsh9","swsh10","swsh11","swsh12",
    "swsh12pt5","swsh45",
]

# ──────────────────────────────────────────
def get_session(api_key: str = "") -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "pokemon-scanner-local/1.0"})
    if api_key:
        s.headers["X-Api-Key"] = api_key
    return s


def fetch_page(session, page: int, page_size: int = 250, q: str = "") -> dict:
    params = {"page": page, "pageSize": page_size}
    if q:
        params["q"] = q
    for attempt in range(5):
        try:
            r = session.get(f"{TCGAPI}/cards", params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10)) + 1
                print(f"  ⏳ Rate limit, esperando {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def download_all_cards(session, sets_filter=None, limit=None) -> list[dict]:
    """Descarga todos los metadatos de cartas paginando."""
    q = ""
    if sets_filter:
        q = " OR ".join(f"set.id:{s}" for s in sets_filter)

    print("📋 Obteniendo conteo total...")
    first = fetch_page(session, 1, 1, q)
    total = first.get("totalCount", 0)
    if limit:
        total = min(total, limit)
    page_size = 250
    pages = (total + page_size - 1) // page_size

    print(f"   Total cartas: {total:,}  |  Páginas: {pages}")

    all_cards = []
    for page in range(1, pages + 1):
        data = fetch_page(session, page, page_size, q)
        cards = data.get("data", [])
        all_cards.extend(cards)
        if limit and len(all_cards) >= limit:
            all_cards = all_cards[:limit]
            break
        pct = len(all_cards) / total * 100
        print(f"  📥 {len(all_cards):,}/{total:,} ({pct:.0f}%)", end="\r")
        time.sleep(0.12)  # respetar rate limit sin API key

    print(f"\n  ✅ {len(all_cards):,} cartas descargadas")
    return all_cards


def normalize_card(raw: dict) -> dict:
    """Extrae solo los campos necesarios de cada carta."""
    prices = raw.get("tcgplayer", {}).get("prices", {})
    price  = None
    for tier in ("holofoil", "normal", "reverseHolofoil", "unlimited", "1stEditionHolofoil"):
        if tier in prices and prices[tier].get("market"):
            price = prices[tier]["market"]
            break

    return {
        "id":       raw["id"],
        "name":     raw["name"],
        "set_id":   raw.get("set", {}).get("id", ""),
        "set_name": raw.get("set", {}).get("name", ""),
        "number":   raw.get("number", ""),
        "rarity":   raw.get("rarity", ""),
        "types":    raw.get("types", []),
        "subtypes": raw.get("subtypes", []),
        "hp":       raw.get("hp", ""),
        "price_usd": price,
        "img_small": raw.get("images", {}).get("small", ""),
        "img_large": raw.get("images", {}).get("large", ""),
    }


# ──────────────────────────────────────────
#  PHASH ENGINE
# ──────────────────────────────────────────
def compute_phash(img_url: str, session: requests.Session, size: int = 8) -> str | None:
    """Descarga imagen y calcula pHash de 64 bits."""
    if not img_url or not HAS_HASH:
        return None
    try:
        r = session.get(img_url, timeout=15)
        r.raise_for_status()
        img  = Image.open(BytesIO(r.content)).convert("RGB")
        ph   = imagehash.phash(img, hash_size=size)
        return str(ph)
    except Exception:
        return None


def save_thumbnail(img_url: str, card_id: str, session: requests.Session) -> bool:
    """Guarda thumbnail 200px localmente."""
    try:
        r = session.get(img_url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((200, 280))
        path = IMG_DIR / f"{card_id}.jpg"
        img.save(path, "JPEG", quality=85)
        return True
    except Exception:
        return False


def build_phash_index(cards_meta: list[dict], session: requests.Session,
                       save_images: bool = False, workers: int = 8) -> dict:
    """
    Descarga imágenes en paralelo y construye índice:
    { phash_hex: card_id, ... }
    """
    if not HAS_HASH:
        print("❌ No se puede generar índice sin Pillow/imagehash.")
        return {}

    if save_images:
        IMG_DIR.mkdir(exist_ok=True)

    phash_index = {}
    total = len(cards_meta)
    done  = 0

    def process(card):
        ph = compute_phash(card["img_small"], session)
        if save_images and card["img_small"]:
            save_thumbnail(card["img_small"], card["id"], session)
        return card["id"], ph

    print(f"\n🔍 Generando pHashes para {total:,} cartas ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, c): c for c in cards_meta}
        for fut in as_completed(futures):
            done += 1
            card_id, ph = fut.result()
            if ph:
                phash_index[ph] = card_id
            if done % 100 == 0 or done == total:
                print(f"  🔢 {done:,}/{total:,} ({done/total*100:.0f}%)", end="\r")

    print(f"\n  ✅ {len(phash_index):,} hashes generados")
    return phash_index


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Construye biblioteca local de cartas Pokémon TCG")
    parser.add_argument("--api-key",     default="",     help="API Key de pokemontcg.io (opcional pero recomendado)")
    parser.add_argument("--sets",        nargs="*",      help="IDs de sets a incluir (ej: sv1 sv2 sv3). Por defecto: sets modernos")
    parser.add_argument("--all-sets",    action="store_true", help="Incluir TODOS los sets históricos (~18k cartas)")
    parser.add_argument("--limit",       type=int,       help="Limitar número de cartas (para pruebas)")
    parser.add_argument("--no-hash",     action="store_true", help="Solo metadatos, sin pHashes (más rápido)")
    parser.add_argument("--save-images", action="store_true", help="Guardar thumbnails localmente en data/images/")
    parser.add_argument("--workers",     type=int, default=8, help="Hilos paralelos para descarga de imágenes (default: 8)")
    args = parser.parse_args()

    print("═" * 55)
    print("  🃏 Pokémon TCG — Constructor de biblioteca local")
    print("═" * 55)

    session = get_session(args.api_key)
    if args.api_key:
        print(f"  🔑 API Key: {args.api_key[:12]}…")
    else:
        print("  ⚠  Sin API Key — rate limit de ~1000 req/día")
        print("     Obtén una gratis en https://pokemontcg.io")

    # Decidir qué sets incluir
    sets_filter = None
    if not args.all_sets:
        sets_filter = args.sets if args.sets else MODERN_SETS
        print(f"\n  📦 Sets a descargar: {len(sets_filter)}")
        for s in sets_filter:
            print(f"     • {s}")
    else:
        print("\n  📦 Descargando TODOS los sets históricos")

    # 1. Descargar metadatos
    print("\n📥 Fase 1 — Descargando metadatos...")
    raw_cards = download_all_cards(session, sets_filter, args.limit)
    cards_meta = [normalize_card(c) for c in raw_cards]

    # 2. Guardar metadatos
    meta_path = DATA_DIR / "cards_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cards_meta, f, ensure_ascii=False, indent=2)
    size_mb = meta_path.stat().st_size / 1_048_576
    print(f"  💾 Guardado: data/cards_meta.json ({size_mb:.1f} MB)")

    # 3. Índice por ID para búsqueda O(1)
    id_index = {c["id"]: c for c in cards_meta}
    id_path = DATA_DIR / "id_index.json"
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(id_index, f, ensure_ascii=False)
    print(f"  💾 Guardado: data/id_index.json")

    # 4. pHash index
    if not args.no_hash:
        print("\n🔢 Fase 2 — Generando índice de hashes perceptuales...")
        phash_index = build_phash_index(
            cards_meta, session,
            save_images=args.save_images,
            workers=args.workers
        )
        phash_path = DATA_DIR / "phash_index.json"
        with open(phash_path, "w") as f:
            json.dump(phash_index, f)
        size_mb = phash_path.stat().st_size / 1_048_576
        print(f"  💾 Guardado: data/phash_index.json ({size_mb:.1f} MB)")
    else:
        print("\n  ⏭  Saltando generación de pHashes (--no-hash)")

    print("\n" + "═" * 55)
    print(f"  ✅ Biblioteca lista — {len(cards_meta):,} cartas indexadas")
    print(f"  📂 Archivos en: {DATA_DIR.resolve()}")
    print("═" * 55)

    if not args.no_hash:
        print("""
  Próximo paso:
    python run_local.py   ← servidor con reconocimiento local
""")


if __name__ == "__main__":
    main()
