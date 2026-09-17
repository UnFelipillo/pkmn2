"""
run_local.py — Servidor de desarrollo local

Uso:
    python run_local.py

Abre: http://localhost:8000

PRIMER USO — genera la biblioteca de cartas primero:
    python build_library.py --sets sv1 sv2 sv3 sv3pt5 sv4

Luego ya puedes correr el servidor y escanear offline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from fastapi.staticfiles import StaticFiles
from api.index import app, STATIC_DIR

if __name__ == "__main__":
    print("=" * 58)
    print("  🃏 Pokémon Card Scanner — Servidor local v2.0")
    print("=" * 58)

    from pathlib import Path
    data_dir = Path(__file__).parent / "data"
    phash    = data_dir / "phash_index.json"
    if not phash.exists():
        print("""
  ⚠  Biblioteca local NO encontrada.
     El scanner usará la API remota como fallback.

  Para activar reconocimiento offline (~2-5ms por carta):
     python build_library.py --sets sv1 sv2 sv3 sv3pt5 sv4

  Para sets completos modernos:
     python build_library.py

  Para TODO el catálogo histórico (~18k cartas, ~30min):
     python build_library.py --all-sets
""")
    else:
        import json
        n = len(json.load(open(phash)))
        print(f"\n  ✅ Biblioteca lista: {n:,} cartas indexadas\n")

    print(f"  🌐 Abre: http://localhost:8000\n")
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
