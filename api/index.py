"""
api/index.py v3 — Scanner + Inventario + Constructor de Mazos
"""
import cv2, numpy as np, base64, math, time, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Literal

from api.recognizer import get_recognizer
from api.inventory  import (
    get_inventory, upsert_card, remove_card, inventory_stats,
    analyze_all_decks, InventoryCard, Condition
)

app = FastAPI(title="Pokemon Card Scanner API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent.parent / "static"
TCGAPI     = "https://api.pokemontcg.io/v2"

# ── MODELOS ────────────────────────────────
class ScanRequest(BaseModel):
    image:    str
    debug:    bool  = False
    top_n:    int   = 1
    min_conf: float = 0.50

class AddToInventoryRequest(BaseModel):
    card_id:   str
    name:      str
    set_name:  str   = ""
    set_id:    str   = ""
    number:    str   = ""
    rarity:    str   = ""
    types:     list  = []
    price_usd: float | None = None
    img_small: str   = ""
    quantity:  int   = 1
    condition: Condition = "NM"

class UpdateQtyRequest(BaseModel):
    delta:     int   = 1
    condition: Condition = "NM"

class CardMatch(BaseModel):
    card_id:   str   = ""
    name:      str   = ""
    set_name:  str   = ""
    set_id:    str   = ""
    number:    str   = ""
    rarity:    str   = ""
    types:     list  = []
    price_usd: float | None = None
    img_small: str   = ""
    confidence:float = 0.0
    hamming:   int   = 64
    source:    str   = ""
    in_inventory: bool = False
    inventory_qty: int = 0

class ScanResponse(BaseModel):
    success:        bool
    card_image:     str | None = None
    debug_image:    str | None = None
    detection_conf: float      = 0.0
    matches:        list[CardMatch] = []
    best_match:     CardMatch | None = None
    time_ms:        dict       = {}
    message:        str        = ""

# ── OPENCV ────────────────────────────────
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    return rect

def dist(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl,tr,br,bl = rect
    out_w = int(max(dist(tl,tr), dist(bl,br)))
    out_h = int(max(dist(tl,bl), dist(tr,br)))
    if out_h < out_w: out_w, out_h = out_h, out_w
    exp_h = int(out_w * 88/63)
    if out_w > 0 and abs(out_h - exp_h)/max(out_h,1) < 0.3: out_h = exp_h
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype="float32")
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (out_w, out_h))

def detect_and_warp(image, debug=False):
    h,w = image.shape[:2]; area = h*w
    dbg = image.copy() if debug else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    med = np.median(blurred)
    edges = cv2.Canny(blurred, int(max(0,.66*med)), int(min(255,1.33*med)))
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, dbg, 0.0
    candidates = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < area*.05 or a > area*.95: continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx)!=4 or not cv2.isContourConvex(approx): continue
        x,y,bw,bh = cv2.boundingRect(approx)
        ratio = max(bw,bh)/min(bw,bh) if min(bw,bh)>0 else 0
        if abs(ratio - 88/63)/(88/63) > 0.35: continue
        sol = a/max(cv2.contourArea(cv2.convexHull(cnt)),1)
        candidates.append({"c":approx,"area":a,"sol":sol,"score":a*sol})
    if not candidates:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(cnt)
            if a < area*.05: continue
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
            if len(approx)==4:
                candidates.append({"c":approx,"area":a,"sol":.5,"score":a*.5}); break
    if not candidates: return None, dbg, 0.0
    best = max(candidates, key=lambda x: x["score"])
    pts  = best["c"].reshape(4,2).astype("float32")
    conf = min(1.0, (best["area"]/area)*best["sol"]*3)
    if debug and dbg is not None:
        cv2.drawContours(dbg,[best["c"]],-1,(0,255,0),3)
        for i,pt in enumerate(order_points(pts)):
            cv2.circle(dbg,tuple(pt.astype(int)),8,[(0,0,255),(0,255,0),(255,0,0),(0,165,255)][i],-1)
        cv2.putText(dbg,f"Conf:{conf:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
    return four_point_transform(image, pts), dbg, conf

def b64_to_cv2(b64):
    if "," in b64: b64=b64.split(",",1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64),np.uint8), cv2.IMREAD_COLOR)

def cv2_to_b64(img, q=90):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,q])
    return base64.b64encode(buf).decode()

async def api_fallback(name: str = "") -> CardMatch | None:
    if not name: return None
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{TCGAPI}/cards",
                params={"q":f'name:"{name}"',"pageSize":1,"orderBy":"-set.releaseDate"})
            data = r.json().get("data",[])
            if not data: return None
            c = data[0]
            prices = c.get("tcgplayer",{}).get("prices",{})
            price = next((v.get("market") for v in prices.values() if v.get("market")), None)
            return CardMatch(card_id=c["id"],name=c["name"],
                set_name=c.get("set",{}).get("name",""),set_id=c.get("set",{}).get("id",""),
                number=c.get("number",""),rarity=c.get("rarity",""),
                types=c.get("types",[]),price_usd=price,
                img_small=c.get("images",{}).get("small",""),
                confidence=0.5,source="api_fallback")
    except: return None

def enrich_with_inventory(match: CardMatch, inventory: dict) -> CardMatch:
    """Agrega datos de inventario al match."""
    if match.card_id in inventory:
        match.in_inventory  = True
        match.inventory_qty = inventory[match.card_id].quantity
    return match

# ── ENDPOINTS: SCANNER ─────────────────────
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR/"index.html"))

@app.get("/api/health")
async def health():
    rec = get_recognizer(); s = rec.stats()
    return {"status":"ok","opencv":cv2.__version__,
            "library_ready":s["ready"],"total_cards":s["total_cards"],
            "search_engine":"numpy" if s["has_numpy"] else "linear",
            "inventory_cards": len(get_inventory())}

@app.post("/api/scan", response_model=ScanResponse)
async def scan_card(req: ScanRequest):
    T = {}
    try:
        t=time.perf_counter(); img=b64_to_cv2(req.image)
        if img is None: raise ValueError("imagen inválida")
        T["decode_ms"]=round((time.perf_counter()-t)*1000,1)
    except Exception as e:
        raise HTTPException(400, str(e))

    t=time.perf_counter()
    card_img, debug_img, det_conf = detect_and_warp(img, debug=req.debug)
    T["opencv_ms"]=round((time.perf_counter()-t)*1000,1)

    if card_img is None:
        return ScanResponse(success=False, detection_conf=det_conf,
            debug_image=cv2_to_b64(debug_img) if debug_img is not None else None,
            time_ms=T, message="No se detectó carta. Usa fondo oscuro y buena luz.")

    card_b64 = cv2_to_b64(card_img, q=92)
    inventory = get_inventory()
    matches: list[CardMatch] = []
    rec = get_recognizer()

    t=time.perf_counter()
    if rec.is_ready():
        top = rec.top_matches(card_img, n=min(req.top_n,5))
        for r in top:
            m = CardMatch(card_id=r.card_id,name=r.name,set_name=r.set_name,
                set_id=r.set_id,number=r.number,rarity=r.rarity,
                types=r.types or [],price_usd=r.price_usd,img_small=r.img_small,
                confidence=r.confidence,hamming=r.hamming,source="local_phash")
            matches.append(enrich_with_inventory(m, inventory))
        T["recognition_ms"]=round((time.perf_counter()-t)*1000,1)

    best = matches[0] if matches else None
    if not best or best.confidence < req.min_conf:
        fb = await api_fallback(best.name if best else "")
        if fb:
            fb = enrich_with_inventory(fb, inventory)
            if not best or fb.confidence > best.confidence: best=fb
            matches.insert(0, fb)
        T["fallback_ms"]=round((time.perf_counter()-t)*1000,1)

    found = bool(best and best.confidence >= req.min_conf)
    inv_note = ""
    if found and best:
        if best.in_inventory:
            inv_note = f" | 📦 En inventario: {best.inventory_qty} cop."
        else:
            inv_note = " | ➕ No está en tu inventario"

    msg = (f"✅ {best.name} — {best.set_name} #{best.number} | "
           f"{best.confidence:.0%} confianza{inv_note}" if found and best else
           "Carta recortada pero no identificada. Ejecuta build_library.py")

    return ScanResponse(success=found, card_image=card_b64,
        debug_image=cv2_to_b64(debug_img) if debug_img is not None else None,
        detection_conf=round(det_conf,3), matches=matches, best_match=best,
        time_ms=T, message=msg)

# ── ENDPOINTS: INVENTARIO ──────────────────
@app.get("/api/inventory")
async def get_inv():
    inv = get_inventory()
    return {
        "cards": [c.to_dict() for c in sorted(inv.values(), key=lambda x: x.name)],
        "stats": inventory_stats(),
    }

@app.post("/api/inventory")
async def add_to_inv(req: AddToInventoryRequest):
    card_data = req.model_dump()
    card, is_new = await upsert_card(card_data, qty_delta=req.quantity, condition=req.condition)
    return {
        "card":    card.to_dict(),
        "is_new":  is_new,
        "message": f"{'✅ Agregada' if is_new else f'📦 Actualizada — ahora tienes {card.quantity}'}: {card.name}",
    }

@app.patch("/api/inventory/{card_id}")
async def update_qty(card_id: str, req: UpdateQtyRequest):
    inv = get_inventory()
    if card_id not in inv and req.delta < 0:
        raise HTTPException(404, "Carta no encontrada")
    card_data = inv[card_id].to_dict() if card_id in inv else {"card_id": card_id}
    card, is_new = await upsert_card(card_data, qty_delta=req.delta)
    return {"card": card.to_dict(), "message": f"Cantidad actualizada: {card.quantity}"}

@app.delete("/api/inventory/{card_id}")
async def delete_from_inv(card_id: str):
    ok = await remove_card(card_id)
    if not ok: raise HTTPException(404, "Carta no encontrada")
    return {"message": "Carta eliminada del inventario"}

@app.get("/api/inventory/stats")
async def inv_stats():
    return inventory_stats()

# ── ENDPOINTS: MAZOS ──────────────────────
@app.get("/api/decks")
async def get_deck_suggestions():
    inventory = get_inventory()
    results   = analyze_all_decks(inventory)
    return {
        "decks":     [r.to_dict() for r in results],
        "buildable": [r.to_dict() for r in results if r.buildable],
        "summary": {
            "total_archetypes": len(results),
            "buildable_count":  sum(1 for r in results if r.buildable),
            "best_deck":        results[0].name if results else None,
            "best_pct":         round(results[0].completion_pct, 1) if results else 0,
        }
    }

@app.get("/api/decks/{archetype_id}")
async def get_deck_detail(archetype_id: str):
    from api.inventory import META_ARCHETYPES, build_deck_analysis
    arch = next((a for a in META_ARCHETYPES if a["id"] == archetype_id), None)
    if not arch: raise HTTPException(404, "Arquetipo no encontrado")
    inventory = get_inventory()
    result    = build_deck_analysis(arch, inventory)
    return result.to_dict()

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
