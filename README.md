# 🃏 Pokémon Card Scanner

Detecta cartas Pokémon con la cámara web, aplica corrección de perspectiva (warpPerspective) con OpenCV y entrega una imagen limpia y recta de la carta.

## Stack

| Capa | Tecnología |
|------|------------|
| Backend | Python · FastAPI · OpenCV |
| Frontend | HTML/CSS/JS vanilla |
| Hosting | Vercel (serverless) |

## Cómo funciona el algoritmo

```
Frame de webcam (JPEG base64)
        │
        ▼
  Escala de grises
        │
        ▼
  Gaussian Blur (5×5)
        │
        ▼
  Canny Edge Detection (adaptivo por mediana)
        │
        ▼
  Dilatación de bordes (cerrar gaps)
        │
        ▼
  findContours → filtrar cuadriláteros
        │
        ▼
  Filtro por ratio carta Pokémon (88/63 ≈ 1.397)
        │
        ▼
  order_points → [TL, TR, BR, BL]
        │
        ▼
  getPerspectiveTransform + warpPerspective
        │
        ▼
  Carta limpia (JPEG base64)
```

---

## Instalación local

### 1. Requisitos
- Python 3.10+
- pip

### 2. Clonar e instalar dependencias

```bash
git clone <tu-repo>
cd pokemon-scanner
pip install -r requirements.txt
```

### 3. Ejecutar servidor local

```bash
python run_local.py
```

Abre `http://localhost:8000` en tu navegador.

> **Nota:** La cámara requiere HTTPS o `localhost`. En red local usa `localhost`, no la IP.

---

## Deploy en Vercel

### Opción A: CLI (recomendado)

```bash
# 1. Instalar Vercel CLI
npm i -g vercel

# 2. Login
vercel login

# 3. Deploy
vercel

# 4. Producción
vercel --prod
```

### Opción B: GitHub + Vercel Dashboard

1. Sube el proyecto a un repositorio GitHub
2. Ve a [vercel.com/new](https://vercel.com/new)
3. Importa el repositorio
4. Vercel detecta automáticamente la configuración
5. Click en **Deploy**

---

## Endpoints de la API

### `POST /api/scan`

Recibe un frame y devuelve la carta detectada.

**Request:**
```json
{
  "image": "<base64 JPEG/PNG>",
  "debug": false
}
```

**Response:**
```json
{
  "success": true,
  "card_image": "<base64 JPEG de la carta recortada>",
  "debug_image": null,
  "confidence": 0.87,
  "message": "Carta detectada correctamente"
}
```

### `GET /api/health`

```json
{
  "status": "ok",
  "opencv_version": "4.10.0",
  "numpy_version": "1.26.4"
}
```

---

## Tips para mejor detección

| Situación | Recomendación |
|-----------|---------------|
| Fondo | Usa fondo oscuro o blanco uniforme, sin patrones |
| Iluminación | Luz difusa, evita reflejos directos en la carta |
| Ángulo | Mantén la carta lo más plana posible (< 30° de inclinación) |
| Distancia | La carta debe ocupar al menos 30% del frame |
| Movimiento | Mantén la carta estable 1-2 segundos |

## Estructura del proyecto

```
pokemon-scanner/
├── api/
│   └── index.py          # FastAPI + OpenCV (el cerebro)
├── static/
│   └── index.html        # Frontend completo (webcam + UI)
├── run_local.py          # Servidor de desarrollo local
├── requirements.txt      # Dependencias Python
├── vercel.json           # Configuración de deploy
└── README.md
```
