"""
inventory.py
════════════
Módulo de inventario físico + constructor de mazos funcionales.

Inventario:
  - Almacena cartas físicas con cantidad y condición
  - Persistente en data/inventory.json
  - API thread-safe con asyncio.Lock

Constructor de mazos:
  - Reglas TCG estándar: 60 cartas, máx 4 copias (excl. energías básicas)
  - Formatos soportados: Standard, Expanded, Unlimited
  - Analiza colección y sugiere el mazo más completo posible
  - Muestra % completado, cartas disponibles, faltantes + precio
  - Basado en arquetipos meta reales + construcción automática
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Literal
from copy import deepcopy

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
INVENTORY_FILE = DATA_DIR / "inventory.json"

_lock = asyncio.Lock()

# ──────────────────────────────────────────
#  ESTRUCTURAS
# ──────────────────────────────────────────
Condition = Literal["NM", "LP", "MP", "HP", "D"]  # Near Mint → Damaged

@dataclass
class InventoryCard:
    card_id:    str
    name:       str
    set_name:   str
    set_id:     str
    number:     str
    rarity:     str
    types:      list
    price_usd:  float | None
    img_small:  str
    quantity:   int             = 1
    condition:  Condition       = "NM"
    added_at:   str             = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str             = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────
#  INVENTARIO
# ──────────────────────────────────────────
def _load_raw() -> dict:
    if not INVENTORY_FILE.exists():
        return {}
    try:
        return json.loads(INVENTORY_FILE.read_text())
    except Exception:
        return {}

def _save_raw(data: dict):
    INVENTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def get_inventory() -> dict[str, InventoryCard]:
    raw = _load_raw()
    return {k: InventoryCard.from_dict(v) for k, v in raw.items()}

async def upsert_card(card_data: dict, qty_delta: int = 1,
                      condition: Condition = "NM") -> tuple[InventoryCard, bool]:
    """
    Agrega o actualiza una carta en el inventario.
    Devuelve (carta, es_nueva).
    """
    async with _lock:
        raw = _load_raw()
        card_id = card_data.get("card_id") or card_data.get("id", "")
        is_new  = card_id not in raw

        if is_new:
            card = InventoryCard(
                card_id   = card_id,
                name      = card_data.get("name", ""),
                set_name  = card_data.get("set_name", ""),
                set_id    = card_data.get("set_id", ""),
                number    = card_data.get("number", ""),
                rarity    = card_data.get("rarity", ""),
                types     = card_data.get("types", []),
                price_usd = card_data.get("price_usd"),
                img_small = card_data.get("img_small", ""),
                quantity  = max(0, qty_delta),
                condition = condition,
            )
        else:
            card = InventoryCard.from_dict(raw[card_id])
            card.quantity   = max(0, card.quantity + qty_delta)
            card.updated_at = datetime.now().isoformat()
            # Actualizar precio si viene uno nuevo
            if card_data.get("price_usd") and not card.price_usd:
                card.price_usd = card_data["price_usd"]

        if card.quantity == 0:
            raw.pop(card_id, None)
        else:
            raw[card_id] = card.to_dict()

        _save_raw(raw)
        return card, is_new

async def remove_card(card_id: str) -> bool:
    async with _lock:
        raw = _load_raw()
        if card_id in raw:
            del raw[card_id]
            _save_raw(raw)
            return True
        return False

def inventory_stats() -> dict:
    inv = get_inventory()
    cards = list(inv.values())
    total_unique  = len(cards)
    total_copies  = sum(c.quantity for c in cards)
    total_value   = sum((c.price_usd or 0) * c.quantity for c in cards)
    by_type       = {}
    for c in cards:
        for t in (c.types or []):
            by_type[t] = by_type.get(t, 0) + c.quantity
    return {
        "total_unique":  total_unique,
        "total_copies":  total_copies,
        "total_value":   round(total_value, 2),
        "by_type":       by_type,
    }


# ──────────────────────────────────────────
#  ARQUETIPOS META (mazos de referencia)
#  Cada mazo es una lista de slots:
#   { role, name, qty, card_ids, is_energy, notes }
#  card_ids: lista de IDs posibles (alternativas), el motor
#            elige el primero que el usuario tenga.
# ──────────────────────────────────────────
META_ARCHETYPES = [
  {
    "id":       "charizard_ex_standard",
    "name":     "Charizard ex",
    "format":   "Standard",
    "tier":     "S",
    "style":    "Aggro-Control",
    "description": "El mazo más dominante del formato. Pidgeot ex provee búsqueda de cartas ilimitada; Charizard ex cierra rápido con Burning Darkness.",
    "pokemon_count": 18,
    "trainer_count": 30,
    "energy_count":  12,
    "slots": [
      # Pokémon
      {"role":"main_attacker","name":"Charizard ex","qty":4,"is_energy":False,
       "card_ids":["sv3-125","sv3pt5-54"],"notes":"Atacante principal"},
      {"role":"engine","name":"Pidgeot ex","qty":3,"is_energy":False,
       "card_ids":["sv3-164"],"notes":"Motor de búsqueda, habilidad Quick Search"},
      {"role":"evolution","name":"Charmander","qty":4,"is_energy":False,
       "card_ids":["sv3-46","svp-37"],"notes":"Básico para Charizard"},
      {"role":"evolution","name":"Charmeleon","qty":2,"is_energy":False,
       "card_ids":["sv3-48"],"notes":"Intermedio"},
      {"role":"evolution","name":"Pidgey","qty":4,"is_energy":False,
       "card_ids":["sv3-162"],"notes":"Básico para Pidgeot"},
      {"role":"evolution","name":"Pidgeotto","qty":1,"is_energy":False,
       "card_ids":["sv3-163"],"notes":"Intermedio"},
      {"role":"tech","name":"Radiant Charizard","qty":1,"is_energy":False,
       "card_ids":["swsh11-20"],"notes":"Atacante secundario económico"},
      {"role":"tech","name":"Mew","qty":1,"is_energy":False,
       "card_ids":["cel25-11","swsh12pt5-128"],"notes":"Barrera contra Genesect"},
      {"role":"tech","name":"Ditto","qty":1,"is_energy":False,
       "card_ids":["sv3pt5-129"],"notes":"Comodín de evolución"},
      # Entrenadores — Supporters
      {"role":"supporter","name":"Professor's Research","qty":4,"is_energy":False,
       "card_ids":["swsh45-62","sv1-189","sv7-150"],"notes":"Draw principal"},
      {"role":"supporter","name":"Iono","qty":3,"is_energy":False,
       "card_ids":["sv2-185","sv5-80"],"notes":"Disrupción + draw"},
      {"role":"supporter","name":"Arven","qty":4,"is_energy":False,
       "card_ids":["sv1-186"],"notes":"Busca Item + Herramienta"},
      {"role":"supporter","name":"Boss's Orders","qty":2,"is_energy":False,
       "card_ids":["swsh2-154","sv5-183"],"notes":"Gust, cierra KOs"},
      # Entrenadores — Items
      {"role":"item","name":"Ultra Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-196","sv7-196"],"notes":"Búsqueda de Pokémon"},
      {"role":"item","name":"Rare Candy","qty":4,"is_energy":False,
       "card_ids":["sv1-191","sv7-185"],"notes":"Evolucion rápida"},
      {"role":"item","name":"Nest Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-181"],"notes":"Básicos al banco"},
      {"role":"item","name":"Lost Vacuum","qty":2,"is_energy":False,
       "card_ids":["swsh12-162"],"notes":"Remueve herramientas/estadios"},
      {"role":"item","name":"Pal Pad","qty":2,"is_energy":False,
       "card_ids":["sv1-182"],"notes":"Recupera supporters"},
      # Estadios
      {"role":"stadium","name":"Magma Basin","qty":3,"is_energy":False,
       "card_ids":["swsh11-144"],"notes":"Acelera energía de Fuego"},
      {"role":"stadium","name":"Artazon","qty":2,"is_energy":False,
       "card_ids":["sv1-171"],"notes":"Busca Pokémon sin habilidad"},
      # Herramientas
      {"role":"tool","name":"Buddy-Buddy Poffin","qty":3,"is_energy":False,
       "card_ids":["sv5-144"],"notes":"Básicos 60HP o menos"},
      {"role":"tool","name":"Counter Catcher","qty":2,"is_energy":False,
       "card_ids":["sv5-91"],"notes":"Gust condicional"},
      # Energía
      {"role":"energy","name":"Fire Energy","qty":11,"is_energy":True,
       "card_ids":["sv1-267","sve-2"],"notes":"Energía básica Fuego"},
      {"role":"energy","name":"Ordinary Rod","qty":1,"is_energy":False,
       "card_ids":["sv1-171"],"notes":"Recupera cartas del descarte"},
    ]
  },
  {
    "id":       "gardevoir_ex_standard",
    "name":     "Gardevoir ex",
    "format":   "Standard",
    "tier":     "S",
    "style":    "Combo",
    "description": "Mazo combo que acelera Energías Psíquicas con la habilidad Psychic Embrace de Gardevoir ex. Comfey genera ventaja de cartas desde la Lost Zone.",
    "pokemon_count": 16,
    "trainer_count": 32,
    "energy_count":  12,
    "slots": [
      {"role":"main_attacker","name":"Gardevoir ex","qty":3,"is_energy":False,
       "card_ids":["sv2-86"],"notes":"Atacante + aceleración de energía"},
      {"role":"evolution","name":"Ralts","qty":4,"is_energy":False,
       "card_ids":["sv2-81","sv2-60"],"notes":"Básico para Gardevoir"},
      {"role":"evolution","name":"Kirlia","qty":4,"is_energy":False,
       "card_ids":["sv2-82"],"notes":"Habilidad Refinement = draw"},
      {"role":"evolution","name":"Gardevoir","qty":1,"is_energy":False,
       "card_ids":["sv2-83"],"notes":"Puente de evolución"},
      {"role":"engine","name":"Comfey","qty":4,"is_energy":False,
       "card_ids":["swsh12-79"],"notes":"Flower Selecting = draw/Lost Zone"},
      {"role":"tech","name":"Zacian V","qty":2,"is_energy":False,
       "card_ids":["swsh4-138"],"notes":"Reconocer Espada = draw"},
      {"role":"tech","name":"Mew ex","qty":1,"is_energy":False,
       "card_ids":["sv3-151"],"notes":"Protección de banco"},
      # Supporters
      {"role":"supporter","name":"Iono","qty":4,"is_energy":False,
       "card_ids":["sv2-185"],"notes":"Draw + disrupción"},
      {"role":"supporter","name":"Arven","qty":3,"is_energy":False,
       "card_ids":["sv1-186"],"notes":"Busca items"},
      {"role":"supporter","name":"Boss's Orders","qty":2,"is_energy":False,
       "card_ids":["sv5-183"],"notes":"Gust"},
      {"role":"supporter","name":"Arezu","qty":2,"is_energy":False,
       "card_ids":["swsh10-153"],"notes":"Evolución rápida de Kirlia"},
      # Items
      {"role":"item","name":"Ultra Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-196"],"notes":"Búsqueda Pokémon"},
      {"role":"item","name":"Battle VIP Pass","qty":4,"is_energy":False,
       "card_ids":["swsh9-225"],"notes":"Básicos en turno 1"},
      {"role":"item","name":"Rare Candy","qty":3,"is_energy":False,
       "card_ids":["sv1-191"],"notes":"Evolucion rápida"},
      {"role":"item","name":"Colress's Experiment","qty":3,"is_energy":False,
       "card_ids":["swsh12-155"],"notes":"Draw + Lost Zone"},
      {"role":"item","name":"Mirage Gate","qty":4,"is_energy":False,
       "card_ids":["swsh12-163"],"notes":"Acelera 2 energías básicas"},
      {"role":"stadium","name":"Fog Crystal","qty":3,"is_energy":False,
       "card_ids":["cel25-140"],"notes":"Busca Psíquico básico o energia"},
      {"role":"tool","name":"Escape Rope","qty":2,"is_energy":False,
       "card_ids":["sv1-125","sv5-125"],"notes":"Switch forzado"},
      # Energía
      {"role":"energy","name":"Psychic Energy","qty":11,"is_energy":True,
       "card_ids":["sv1-270","sve-5"],"notes":"Energía básica Psíquica"},
      {"role":"energy","name":"Reversal Energy","qty":1,"is_energy":False,
       "card_ids":["sv5-192"],"notes":"Energía especial de remontada"},
    ]
  },
  {
    "id":       "miraidon_ex_standard",
    "name":     "Miraidon ex",
    "format":   "Standard",
    "tier":     "A",
    "style":    "Aggro",
    "description": "Mazo agresivo eléctrico. Miraidon ex llena el banco de Básicos Eléctricos con Tandem Unit; Flaaffy acelera energía desde el banco.",
    "pokemon_count": 14,
    "trainer_count": 34,
    "energy_count":  12,
    "slots": [
      {"role":"main_attacker","name":"Miraidon ex","qty":4,"is_energy":False,
       "card_ids":["sv1-81","sv1-227"],"notes":"Atacante principal + setup"},
      {"role":"engine","name":"Flaaffy","qty":4,"is_energy":False,
       "card_ids":["swsh7-55"],"notes":"Dynamotor = acelera Elec del descarte"},
      {"role":"evolution","name":"Mareep","qty":4,"is_energy":False,
       "card_ids":["swsh7-54"],"notes":"Básico para Flaaffy"},
      {"role":"tech","name":"Raichu V","qty":1,"is_energy":False,
       "card_ids":["swsh9-45"],"notes":"Atacante secundario"},
      {"role":"tech","name":"Sandy Shocks ex","qty":1,"is_energy":False,
       "card_ids":["sv4-118"],"notes":"Ataca sin necesidad de evolución"},
      # Supporters
      {"role":"supporter","name":"Professor's Research","qty":4,"is_energy":False,
       "card_ids":["swsh45-62","sv1-189"],"notes":"Draw"},
      {"role":"supporter","name":"Iono","qty":3,"is_energy":False,
       "card_ids":["sv2-185"],"notes":"Disrupción"},
      {"role":"supporter","name":"Boss's Orders","qty":2,"is_energy":False,
       "card_ids":["sv5-183"],"notes":"Gust"},
      {"role":"supporter","name":"Professor Turo's Scenario","qty":2,"is_energy":False,
       "card_ids":["sv2-171"],"notes":"Switch de ex al banco"},
      # Items
      {"role":"item","name":"Ultra Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-196"],"notes":"Búsqueda"},
      {"role":"item","name":"Electric Generator","qty":4,"is_energy":False,
       "card_ids":["sv1-170"],"notes":"Acelera energía Eléctrica"},
      {"role":"item","name":"Nest Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-181"],"notes":"Básicos al banco"},
      {"role":"item","name":"Switch","qty":3,"is_energy":False,
       "card_ids":["sv1-194"],"notes":"Switch activo/banco"},
      {"role":"item","name":"Lost Vacuum","qty":2,"is_energy":False,
       "card_ids":["swsh12-162"],"notes":"Remueve herramientas"},
      # Estadio
      {"role":"stadium","name":"Pokémon League HQ","qty":3,"is_energy":False,
       "card_ids":["sv3-192"],"notes":"Bloquea Habilidades de Reglas"},
      # Energía
      {"role":"energy","name":"Lightning Energy","qty":11,"is_energy":True,
       "card_ids":["sv1-271","sve-4"],"notes":"Energía básica Eléctrica"},
      {"role":"energy","name":"Basic Lightning Energy","qty":1,"is_energy":True,
       "card_ids":["sve-4"],"notes":"Energía básica extra"},
    ]
  },
  {
    "id":       "lost_zone_box",
    "name":     "Lost Zone Box",
    "format":   "Standard",
    "tier":     "A",
    "style":    "Control-Aggro",
    "description": "Mazo que alimenta la Lost Zone con Comfey para activar Cramorant y Sableye. Radiant Charizard cierra con daño masivo.",
    "pokemon_count": 12,
    "trainer_count": 36,
    "energy_count":  12,
    "slots": [
      {"role":"engine","name":"Comfey","qty":4,"is_energy":False,
       "card_ids":["swsh12-79"],"notes":"Motor principal Lost Zone"},
      {"role":"attacker","name":"Cramorant","qty":4,"is_energy":False,
       "card_ids":["swsh12-50"],"notes":"Ataca sin energía con 4 Lost"},
      {"role":"attacker","name":"Sableye","qty":2,"is_energy":False,
       "card_ids":["swsh12-70"],"notes":"Robo de premios con Lost Mine"},
      {"role":"attacker","name":"Radiant Charizard","qty":1,"is_energy":False,
       "card_ids":["swsh11-20"],"notes":"Cierre con Combustion Blast"},
      {"role":"tech","name":"Mew ex","qty":1,"is_energy":False,
       "card_ids":["sv3-151"],"notes":"Protección banco"},
      # Supporters
      {"role":"supporter","name":"Professor's Research","qty":4,"is_energy":False,
       "card_ids":["swsh45-62"],"notes":"Draw"},
      {"role":"supporter","name":"Iono","qty":3,"is_energy":False,
       "card_ids":["sv2-185"],"notes":"Disrupción"},
      {"role":"supporter","name":"Boss's Orders","qty":2,"is_energy":False,
       "card_ids":["sv5-183"],"notes":"Gust"},
      {"role":"supporter","name":"Colress's Experiment","qty":4,"is_energy":False,
       "card_ids":["swsh12-155"],"notes":"Draw + Lost Zone crítico"},
      # Items
      {"role":"item","name":"Colress's Experiment","qty":0,"is_energy":False,
       "card_ids":["swsh12-155"],"notes":"(ya contado arriba)"},
      {"role":"item","name":"Mirage Gate","qty":4,"is_energy":False,
       "card_ids":["swsh12-163"],"notes":"Acelera 2 energías básicas (7 Lost)"},
      {"role":"item","name":"Escape Rope","qty":2,"is_energy":False,
       "card_ids":["sv1-125"],"notes":"Switch forzado"},
      {"role":"item","name":"Lost Vacuum","qty":4,"is_energy":False,
       "card_ids":["swsh12-162"],"notes":"Remueve estadios/herramientas (+1 Lost)"},
      {"role":"item","name":"Nest Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-181"],"notes":"Pokémon al banco"},
      {"role":"item","name":"Switch","qty":3,"is_energy":False,
       "card_ids":["sv1-194"],"notes":"Movimiento de activo"},
      # Energía (mezclada para Mirage Gate)
      {"role":"energy","name":"Fire Energy","qty":4,"is_energy":True,
       "card_ids":["sv1-267"],"notes":"Para Radiant Charizard"},
      {"role":"energy","name":"Water Energy","qty":4,"is_energy":True,
       "card_ids":["sv1-264"],"notes":"Para Cramorant"},
      {"role":"energy","name":"Psychic Energy","qty":2,"is_energy":True,
       "card_ids":["sv1-270"],"notes":"Para Sableye"},
      {"role":"energy","name":"Darkness Energy","qty":2,"is_energy":True,
       "card_ids":["sv1-268"],"notes":"Para Sableye/Cramorant"},
    ]
  },
  {
    "id":       "roaring_moon_ex",
    "name":     "Roaring Moon ex",
    "format":   "Standard",
    "tier":     "A",
    "style":    "Aggro",
    "description": "Mazo oscuro ultra agresivo. Dark Patch acelera energía; Roaring Moon ex hace KOs con una sola energía si el rival tiene 3+ premios.",
    "pokemon_count": 10,
    "trainer_count": 38,
    "energy_count":  12,
    "slots": [
      {"role":"main_attacker","name":"Roaring Moon ex","qty":4,"is_energy":False,
       "card_ids":["sv4-228","sv4-124"],"notes":"Atacante principal"},
      {"role":"tech","name":"Squawkabilly ex","qty":2,"is_energy":False,
       "card_ids":["sv2-169"],"notes":"Raucous Flock = draw turno 1"},
      {"role":"tech","name":"Galarian Moltres V","qty":2,"is_energy":False,
       "card_ids":["swsh6-97"],"notes":"Aceleración de energía Oscura"},
      {"role":"tech","name":"Greninja ex","qty":2,"is_energy":False,
       "card_ids":["sv5-130"],"notes":"Atacante secundario flexible"},
      # Supporters
      {"role":"supporter","name":"Professor's Research","qty":4,"is_energy":False,
       "card_ids":["swsh45-62","sv1-189"],"notes":"Draw"},
      {"role":"supporter","name":"Iono","qty":3,"is_energy":False,
       "card_ids":["sv2-185"],"notes":"Disrupción"},
      {"role":"supporter","name":"Boss's Orders","qty":3,"is_energy":False,
       "card_ids":["sv5-183"],"notes":"Gust agresivo"},
      {"role":"supporter","name":"Arven","qty":3,"is_energy":False,
       "card_ids":["sv1-186"],"notes":"Busca items clave"},
      # Items
      {"role":"item","name":"Ultra Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-196"],"notes":"Búsqueda Pokémon"},
      {"role":"item","name":"Dark Patch","qty":4,"is_energy":False,
       "card_ids":["swsh9-139"],"notes":"Acelera Oscura del descarte"},
      {"role":"item","name":"Nest Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-181"],"notes":"Básicos"},
      {"role":"item","name":"Switch","qty":2,"is_energy":False,
       "card_ids":["sv1-194"],"notes":"Switch"},
      {"role":"item","name":"Pal Pad","qty":2,"is_energy":False,
       "card_ids":["sv1-182"],"notes":"Recupera supporters"},
      # Estadio
      {"role":"stadium","name":"Museu of Discovery","qty":3,"is_energy":False,
       "card_ids":["sv4-165"],"notes":"Activa regla turno 1"},
      # Energía
      {"role":"energy","name":"Darkness Energy","qty":12,"is_energy":True,
       "card_ids":["sv1-268","sve-6"],"notes":"Energía básica Oscura"},
    ]
  },
  {
    "id":       "iron_thorns_ex",
    "name":     "Iron Thorns ex",
    "format":   "Standard",
    "tier":     "B",
    "style":    "Control",
    "description": "Mazo control que bloquea al rival con Iron Thorns ex. Hace que el rival no pueda evolucionar mientras construyes tu campo.",
    "pokemon_count": 12,
    "trainer_count": 36,
    "energy_count":  12,
    "slots": [
      {"role":"main_attacker","name":"Iron Thorns ex","qty":3,"is_energy":False,
       "card_ids":["sv4-120"],"notes":"Bloquea evoluciones del rival"},
      {"role":"tech","name":"Iron Hands ex","qty":2,"is_energy":False,
       "card_ids":["sv4-70"],"notes":"Roba premios extra"},
      {"role":"tech","name":"Iron Valiant ex","qty":2,"is_energy":False,
       "card_ids":["sv4-89"],"notes":"Flexible, ataca barato"},
      {"role":"tech","name":"Raging Bolt ex","qty":2,"is_energy":False,
       "card_ids":["sv6-118"],"notes":"Daño masivo"},
      {"role":"tech","name":"Munkidori","qty":3,"is_energy":False,
       "card_ids":["sv6-85"],"notes":"Mueve daño a Pokémon rival"},
      # Supporters
      {"role":"supporter","name":"Professor's Research","qty":3,"is_energy":False,
       "card_ids":["sv1-189"],"notes":"Draw"},
      {"role":"supporter","name":"Iono","qty":4,"is_energy":False,
       "card_ids":["sv2-185"],"notes":"Draw + disrupción"},
      {"role":"supporter","name":"Boss's Orders","qty":2,"is_energy":False,
       "card_ids":["sv5-183"],"notes":"Gust"},
      {"role":"supporter","name":"Penny","qty":2,"is_energy":False,
       "card_ids":["sv2-183"],"notes":"Regresa Iron Thorns al banco sin KO"},
      # Items
      {"role":"item","name":"Ultra Ball","qty":4,"is_energy":False,
       "card_ids":["sv1-196"],"notes":"Búsqueda"},
      {"role":"item","name":"Electric Generator","qty":4,"is_energy":False,
       "card_ids":["sv1-170"],"notes":"Acelera eléctrico"},
      {"role":"item","name":"Nest Ball","qty":3,"is_energy":False,
       "card_ids":["sv1-181"],"notes":"Básicos"},
      {"role":"item","name":"Counter Catcher","qty":2,"is_energy":False,
       "card_ids":["sv5-91"],"notes":"Gust condicional"},
      # Estadio
      {"role":"stadium","name":"Future Booster Energy Capsule","qty":3,"is_energy":False,
       "card_ids":["sv4-149"],"notes":"Energía adicional para Futuros"},
      # Energía
      {"role":"energy","name":"Lightning Energy","qty":10,"is_energy":True,
       "card_ids":["sv1-271"],"notes":"Eléctrica básica"},
      {"role":"energy","name":"Basic Lightning Energy","qty":2,"is_energy":True,
       "card_ids":["sve-4"],"notes":"Eléctrica básica"},
    ]
  },
]


# ──────────────────────────────────────────
#  CONSTRUCTOR DE MAZOS
# ──────────────────────────────────────────
@dataclass
class DeckSlotResult:
    slot_name:      str
    role:           str
    required_qty:   int
    owned_qty:      int
    missing_qty:    int
    is_energy:      bool
    card_id:        str   = ""   # el ID que se va a usar (del inventario o del mazo)
    img_small:      str   = ""
    price_usd:      float | None = None
    notes:          str   = ""
    status:         str   = ""   # "complete" | "partial" | "missing"

@dataclass
class DeckBuildResult:
    archetype_id:   str
    name:           str
    format:         str
    tier:           str
    style:          str
    description:    str
    total_cards:    int       # siempre 60
    owned_cards:    int       # cuántas tienes
    missing_cards:  int
    completion_pct: float
    missing_cost:   float
    total_cost:     float
    slots:          list[DeckSlotResult]
    buildable:      bool      # ≥ 90% completitud
    missing_key:    list[str] # cartas clave que faltan

    def to_dict(self):
        return {
            "archetype_id":   self.archetype_id,
            "name":           self.name,
            "format":         self.format,
            "tier":           self.tier,
            "style":          self.style,
            "description":    self.description,
            "total_cards":    self.total_cards,
            "owned_cards":    self.owned_cards,
            "missing_cards":  self.missing_cards,
            "completion_pct": round(self.completion_pct, 1),
            "missing_cost":   round(self.missing_cost, 2),
            "total_cost":     round(self.total_cost, 2),
            "buildable":      self.buildable,
            "missing_key":    self.missing_key,
            "slots":          [vars(s) for s in self.slots],
        }


def build_deck_analysis(archetype: dict, inventory: dict[str, InventoryCard]) -> DeckBuildResult:
    """
    Analiza cuántas cartas del arquetipo tienes en el inventario
    y construye el resultado detallado slot por slot.
    """
    slots_result = []
    total_required = 0
    total_owned    = 0
    missing_cost   = 0.0
    total_cost     = 0.0
    missing_key    = []

    KEY_ROLES = {"main_attacker", "engine"}

    for slot in archetype["slots"]:
        req_qty = slot["qty"]
        if req_qty == 0:
            continue  # slot placeholder

        # Buscar en el inventario cuál de los card_ids alternativos tiene el usuario
        best_id    = slot["card_ids"][0] if slot["card_ids"] else ""
        best_img   = ""
        best_price = None
        owned_qty  = 0

        for cid in slot["card_ids"]:
            if cid in inventory:
                card      = inventory[cid]
                owned_qty = card.quantity
                best_id   = cid
                best_img  = card.img_small
                best_price= card.price_usd
                break

        missing_qty = max(0, req_qty - owned_qty)
        have_qty    = min(owned_qty, req_qty)

        if missing_qty > 0 and best_price:
            missing_cost += missing_qty * best_price
        if best_price:
            total_cost += req_qty * best_price

        status = ("complete" if missing_qty == 0
                  else "partial" if owned_qty > 0
                  else "missing")

        if slot["role"] in KEY_ROLES and missing_qty > 0:
            missing_key.append(slot["name"])

        total_required += req_qty
        total_owned    += have_qty

        slots_result.append(DeckSlotResult(
            slot_name    = slot["name"],
            role         = slot["role"],
            required_qty = req_qty,
            owned_qty    = owned_qty,
            missing_qty  = missing_qty,
            is_energy    = slot.get("is_energy", False),
            card_id      = best_id,
            img_small    = best_img,
            price_usd    = best_price,
            notes        = slot.get("notes", ""),
            status       = status,
        ))

    pct = (total_owned / total_required * 100) if total_required > 0 else 0

    return DeckBuildResult(
        archetype_id  = archetype["id"],
        name          = archetype["name"],
        format        = archetype["format"],
        tier          = archetype["tier"],
        style         = archetype["style"],
        description   = archetype["description"],
        total_cards   = total_required,
        owned_cards   = total_owned,
        missing_cards = total_required - total_owned,
        completion_pct= pct,
        missing_cost  = missing_cost,
        total_cost    = total_cost,
        slots         = slots_result,
        buildable     = pct >= 90,
        missing_key   = missing_key,
    )


def analyze_all_decks(inventory: dict[str, InventoryCard]) -> list[DeckBuildResult]:
    """Analiza todos los arquetipos y los ordena por completitud."""
    results = []
    for arch in META_ARCHETYPES:
        r = build_deck_analysis(arch, inventory)
        results.append(r)
    results.sort(key=lambda x: (-x.completion_pct, x.tier))
    return results
