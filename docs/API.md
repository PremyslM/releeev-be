# Brain Graph API — Reference

Knowledge-linking backend: ukládá záchyty (screenshoty / fotky / odkazy), pomocí
Claude Vision z nich vytahuje strukturovaná data, počítá multilingvální embeddingy
(Voyage AI) a v grafové databázi Neo4j je propojuje do smysluplného grafu znalostí.

- **Base URL:** `http://localhost:8000`
- **Formát:** JSON (`Content-Type: application/json`)
- **Autentizace:** žádná (lokální nasazení). API klíče (Anthropic, Voyage) jsou
  serverové, čtou se z `.env`.
- **Interaktivní docs:** FastAPI generuje automaticky:
  - Swagger UI → `http://localhost:8000/docs`
  - ReDoc → `http://localhost:8000/redoc`
  - OpenAPI JSON → `http://localhost:8000/openapi.json`

---

## Obsah

1. [Záchyty](#záchyty)
   - [`POST /analyze-capture`](#post-analyze-capture)
   - [`GET /captures/{id}`](#get-capturesid)
   - [`GET /search`](#get-search)
2. [Údržba & přepočty](#údržba--přepočty)
   - [`POST /reanalyze-topics/{id}`](#post-reanalyze-topicsid)
   - [`POST /reembed`](#post-reembed)
   - [`POST /relink`](#post-relink)
   - [`POST /merge-duplicates`](#post-merge-duplicates)
   - [`POST /group-subjects`](#post-group-subjects)
3. [Vizualizace & zdraví](#vizualizace--zdraví)
   - [`GET /graph-data`](#get-graph-data)
   - [`GET /health`](#get-health)
4. [Datové modely](#datové-modely)
5. [Typy vazeb](#typy-vazeb)
6. [Chyby](#chyby)
7. [Typický workflow](#typický-workflow)

---

## Záchyty

### `POST /analyze-capture`

Nahraje obrázek, vytáhne z něj atributy přes Claude Vision, spočítá embedding a
najde + uloží vazby na existující záchyty.

Obsahuje **deduplikaci podle hash** — pokud nahraješ bajtově shodný obrázek,
vrátí se existující záchyt bez nového volání Claude.

**Automatická údržba:** po uložení a propojení se nový záchyt automaticky zařadí
do **větve předmětu** (`Subject`), pokud zobrazuje stejnou věc jako existující
záchyt (např. stejný produkt z jiného e-shopu). Není třeba ručně volat
`/group-subjects`. Pokud k zařazení došlo, vrátí se název větve v `subject_name`.

**Request body** — [`AnalyzeCaptureRequest`](#analyzecapturerequest)

```json
{
  "image_base64": "<base64 obrázku bez prefixu data:>",
  "image_mime": "image/png"
}
```

> `image_mime` musí odpovídat `^image/(jpeg|png|webp|gif)$`. Base64 nesmí
> obsahovat newliny (na macOS pozor: `base64 ... | tr -d '\n'`).

**Response 200** — [`AnalyzeCaptureResponse`](#analyzecaptureresponse)

```json
{
  "capture_id": "cd3d60e7-b172-46be-b4e8-23ad1cd11b2d",
  "attributes": {
    "title": "Tesla Electric SUV on Country Road",
    "summary": "A Czech article about a Tesla electric SUV…",
    "raw_text": "…",
    "capture_type": "screenshot",
    "topics": ["Tesla", "electric vehicle", "automotive"],
    "people": ["James May"],
    "places": ["Czech Republic"],
    "language": "cs",
    "sentiment": "neutral",
    "confidence": 0.95
  },
  "embedding_dim": 1024,
  "linked_to": [
    {
      "capture_id": "972a182e-…",
      "link_type": "TOPIC_OVERLAP",
      "strength": 0.74,
      "title": "Electric Car Configurator Pricing (Czech)",
      "summary": "…",
      "reasons": ["témata: electric vehicle, Tesla", "kategorie: automotive"]
    }
  ],
  "processing_ms": 4120,
  "subject_name": null
}
```

**Příklad (curl)**

```bash
B64=$(base64 -i screenshot.png | tr -d '\n')
curl -s -X POST http://localhost:8000/analyze-capture \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$B64\",\"image_mime\":\"image/png\"}"
```

**Chyby:** `401` neplatný Anthropic klíč · `429` rate limit Claude · `502` chyba
Claude API · `503` chyba databáze.

---

### `POST /analyze-url`

Druhá možnost nahrání (vedle obrázku): uživatel pošle **odkaz**. Server stáhne
stránku, vytáhne obsah (title, OG meta, text), prožene Claudem a spustí stejný
pipeline jako u obrázku — embedding, vazby, auto-zařazení do větve. `source` =
doména, originální URL se uloží pro znovuotevření.

Má **deduplikaci podle URL** — stejný odkaz vrátí existující záznam.

**Request body** — [`AnalyzeUrlRequest`](#analyzeurlrequest)

```json
{ "url": "https://en.wikipedia.org/wiki/Tesla_Model_3" }
```

**Response 200** — [`AnalyzeCaptureResponse`](#analyzecaptureresponse) (`capture_type` = `document`).
Detail záznamu (`GET /captures/{id}`) pak nese původní URL v `attributes.extra.url`.

**Příklad**

```bash
curl -s -X POST http://localhost:8000/analyze-url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.alza.cz/…"}'
```

**Chyby:** `400` stránku nelze načíst · `422` bez čitelného obsahu · `502` chyba Claude.

---

### `GET /captures/{id}`

Vrátí jeden záchyt včetně atributů, všech vazeb (s vysvětlením „proč") a — pokud
je součástí větve — i informaci o předmětu (`Subject`) a dalších zdrojích.

**Path parametry:** `id` — UUID záchytu.

**Response 200** — [`CaptureDetailResponse`](#capturedetailresponse)

```json
{
  "capture": {
    "id": "972a182e-…",
    "created_at": "2026-06-04T12:00:00",
    "image_hash": "…",
    "attributes": { "title": "Electric Car Configurator…", "topics": ["Tesla", "…"], "…": "…" },
    "embedding": [0.0856, -0.0205, "…"]
  },
  "connections": [
    {
      "source_id": "972a182e-…",
      "target_id": "40599d7f-…",
      "link_type": "TOPIC_OVERLAP",
      "strength": 0.74,
      "other_title": "Tesla Electric SUV on Country Road",
      "other_summary": "…",
      "shared_topics": ["Tesla", "electric vehicle"],
      "reasons": ["témata: electric vehicle, Tesla", "kategorie: automotive"]
    }
  ],
  "subject": {
    "subject_id": "…",
    "name": "Gorenje W3NGPI62SBS Washing Machine",
    "kind": "produkt",
    "siblings": [
      { "id": "8a5c226f-…", "title": "Gorenje Washing Machine on Alza.cz", "source": "alza.cz" }
    ]
  }
}
```

> `subject` je `null`, pokud záchyt není seskupený do žádné větve.

**Chyby:** `404` záchyt nenalezen.

---

### `GET /search`

Hybridní multilingvální vyhledávání. Kombinuje **semantic** (embedding),
**fulltext** a **topic** (přesná shoda tématu) přes **Reciprocal Rank Fusion**.
Funguje cross-lingválně — český dotaz najde anglický obsah.

**Query parametry:**

| Param    | Typ    | Default    | Popis |
|----------|--------|------------|-------|
| `q`        | string | (povinné)  | Dotaz v jakémkoliv jazyce, min. délka 1 |
| `limit`    | int    | `20`       | Max. počet výsledků (1–100) |
| `mode`     | string | `hybrid`   | `hybrid` \| `semantic` \| `fulltext` |
| `expand`   | bool   | `true`     | Claude expanze/překlad krátkých cizojazyčných dotazů |
| `category` | string | —          | Filtr podle domény, např. `automotive`, `watches` |
| `source`   | string | —          | Filtr podle zdroje, např. `alza.cz` (substring) |
| `lang`     | string | —          | Filtr podle jazyka, např. `cs`, `en` |
| `type`     | string | —          | Filtr podle `capture_type` (`screenshot`/`photo`/…) |
| `from`     | string | —          | Od data `YYYY-MM-DD` (včetně) |
| `to`       | string | —          | Do data `YYYY-MM-DD` (včetně) |

> Facety lze libovolně kombinovat: `?q=tutorial&category=programming&lang=cs`.
> Relevance se normalizuje vůči nejlepšímu výsledku **po** aplikaci filtrů.

**Response 200** — [`SearchResponse`](#searchresponse)

```json
{
  "query": "boty",
  "results": [
    { "capture_id": "…", "title": "Converse Chuck Taylor All Star",
      "summary": "…", "score": 1.0, "created_at": "2026-06-04T12:00:00" }
  ],
  "total": 5,
  "facets": {
    "category": { "fashion": 4, "watches": 1 },
    "lang": { "cs": 3, "en": 2 }
  }
}
```

> `score` je normalizovaná relevance v rozsahu `0–1` (1.0 = nejlepší shoda).
> Výsledky pod 40 % top skóre se odřezávají jako šum.
>
> **`facets`** = automaticky odvozené možnosti zúžení nad aktuálními výsledky
> (`{typ: {hodnota: počet}}`). Klient z nich vykreslí klikací filtry — uživatel
> nic nenastavuje. Vrací se jen typy s ≥ 2 různými hodnotami (jinak není co
> filtrovat). Kliknutím se přidá odpovídající facet parametr a hledání se zopakuje.

**Příklad**

```bash
curl -s "http://localhost:8000/search?q=programov%C3%A1n%C3%AD&limit=5"
```

**Poznámky k výkonu:**
- Dotazy a embeddingy se **cachují** — opakované/upřesňované hledání je ~25 ms.
- Claude expanze se volá jen když raw dotaz netrefí žádné téma (typicky cizí
  jazyk). Anglický dotaz, který trefí téma, Claude přeskočí a je výrazně rychlejší.

---

## Navigace entit

Procházení „podle věcí", ne podle záchytů — páteř navigace druhého mozku.

### `GET /entities`

Landmarks: nejčastější entity každého typu (pro rozcestník „Procházet").

**Query:** `limit` (default 30, 1–200).

**Response 200** — [`EntityListResponse`](#entitylistresponse)

```json
{
  "topics":   [{ "name": "automotive", "kind": "topic", "count": 10 }],
  "people":   [{ "name": "James May", "kind": "person", "count": 1 }],
  "places":   [{ "name": "Czech Republic", "kind": "place", "count": 12 }],
  "subjects": [{ "name": "Gorenje Washing Machine", "kind": "subject", "count": 3 }]
}
```

### `GET /entities/{kind}/{name}`

Stránka jedné entity: její záchyty + spoluvyskytující se entity (na proklik).

**Path:** `kind` ∈ `topic` \| `person` \| `place`; `name` (URL-encoded).
**Query:** `limit` (default 50).

**Response 200** — [`EntityDetailResponse`](#entitydetailresponse)

```json
{
  "entity": { "name": "automotive", "kind": "topic", "count": 10 },
  "captures": [
    { "capture_id": "…", "title": "Tesla Electric SUV …", "summary": "…" }
  ],
  "related": [
    { "name": "electric vehicle", "kind": "topic", "count": 7 },
    { "name": "Tesla", "kind": "topic", "count": 3 }
  ]
}
```

**Chyby:** `400` neznámý `kind` · `404` entita nenalezena.

---

## Údržba & přepočty

Tyto endpointy slouží k údržbě grafu po změnách dat nebo logiky. Jsou
idempotentní a opakovaně spustitelné.

### `POST /reanalyze-topics/{id}`

Znovu vytáhne témata, kategorii (z řízeného slovníku), osoby a místa z uloženého
textu (bez obrázku — jen Claude na text). Hodí se pro záchyty s prázdnými tématy.
**Nepřepočítává embedding** (kvůli Voyage rate limitu) — poté spusť `POST /reembed`.

**Response 200**

```json
{
  "capture_id": "972a182e-…",
  "category": "automotive",
  "topics": ["Tesla", "electric vehicle", "car configurator", "automotive"],
  "people": [],
  "places": ["Czech Republic"],
  "note": "Run POST /reembed to refresh embeddings (batched, avoids rate limit)"
}
```

**Chyby:** `404` nenalezeno · `422` žádný text k analýze · `502` chyba Claude.

---

### `POST /reembed`

Přepočítá embeddingy pro záchyty s nulovým/chybějícím vektorem **jedním
dávkovým Voyage requestem** (obejde free-tier limit 3 req/min).

**Query parametry:**

| Param   | Typ  | Default | Popis |
|---------|------|---------|-------|
| `force` | bool | `false` | Přepočítat VŠECHNY záchyty, ne jen nulové |

**Response 200**

```json
{ "status": "done", "updated": 15, "skipped": 0, "failed": 0,
  "message": "Přepočítáno: 15, přeskočeno (OK): 0, chyby: 0" }
```

---

### `POST /relink`

Smaže všechny vazby (`LINKED_TO`) a přepočítá je od nuly podle aktuálního
nastavení engine (prahy, váhy, akumulace důkazů). Spusť po nahrání nových
záchytů nebo po změně logiky linkování.

**Response 200**

```json
{ "status": "done", "captures_processed": 15, "links_created": 22, "failed": 0,
  "message": "Zpracováno 15 záznamů, vytvořeno 22 vazeb, chyby: 0" }
```

---

### `POST /merge-duplicates`

Sloučí záchyty, které jsou **bajtově/obsahově shodná entita** (např. dvakrát
vyfocená stejná stránka), do jednoho uzlu. Přenese hrany, zachová původ
(`merged_count`) a smaže duplikát. Slučuje jen při shodě názvu — ne jen podle
embeddingu (různé věci mívají taky vysokou podobnost).

**Query parametry:**

| Param       | Typ   | Default | Rozsah    | Popis |
|-------------|-------|---------|-----------|-------|
| `threshold` | float | `0.88`  | 0.80–1.0  | Vstupní práh podobnosti kandidátů |

**Response 200**

```json
{
  "status": "done",
  "merged_pairs": 1,
  "remaining_captures": 15,
  "details": [{ "kept": "e97b79ec-…", "kept_title": "GORENJE … Washing Machine",
                "merged": "afb09c64-…", "similarity": 0.908 }],
  "message": "Sloučeno 1 duplicit, zůstává 15 záznamů."
}
```

---

### `POST /group-subjects`

**Nedestruktivně** seskupí záchyty stejné reálné věci (stejný produkt z různých
zdrojů — Alza, jiný eshop, vlastní fotka) pod jeden uzel `Subject` přes
`(Capture)-[:DEPICTS]->(Subject)`. Žádný záchyt se nemaže; každý si drží svůj
`source`. Funguje pro produkty i jiné entity (osoby, místa…). Přebuduje větve od
nuly při každém volání.

**Query parametry:**

| Param       | Typ   | Default | Rozsah    | Popis |
|-------------|-------|---------|-----------|-------|
| `threshold` | float | `0.85`  | 0.70–1.0  | Vstupní práh podobnosti kandidátů |

**Response 200**

```json
{
  "status": "done",
  "subjects": 1,
  "details": [{ "name": "Gorenje W3NGPI62SBS Washing Machine", "kind": "produkt",
                "sources": ["alza.cz", "neznámý zdroj"], "captures": 3 }],
  "message": "Vytvořeno 1 větví (předmětů). Gorenje W3NGPI62SBS … (3× — alza.cz, neznámý zdroj)"
}
```

---

## Vizualizace & zdraví

### `GET /graph-data`

Vrátí všechny uzly a hrany ve formátu pro vis-network. Obsahuje záchyty,
vazby `LINKED_TO`, hub uzly `Subject` a větve `DEPICTS`.

**Response 200**

```json
{
  "nodes": [
    { "id": "972a182e-…", "label": "Electric Car Configurator…", "title": "…",
      "created_at": "2026-06-04T12:00:00", "merged_count": 1, "type": "capture" },
    { "id": "subj-…", "label": "Gorenje W3NGPI62SBS Washing Machine",
      "title": "produkt · 3 zdrojů", "type": "subject" }
  ],
  "edges": [
    { "from": "972a182e-…", "to": "40599d7f-…", "value": 0.74,
      "title": "TOPIC_OVERLAP", "type": "link" },
    { "from": "subj-…", "to": "8a5c226f-…", "value": 1.0,
      "title": "DEPICTS", "type": "depicts" }
  ]
}
```

> Uzly: `type` = `capture` \| `subject`. Hrany: `type` = `link` \| `depicts`.
> `value` u vazeb = síla 0–1, u DEPICTS konstanta.

---

### `GET /health`

Zdravotní kontrola. (Není v OpenAPI schématu.)

```json
{ "status": "ok", "env": "production" }
```

---

## Datové modely

### `AnalyzeCaptureRequest`

| Pole           | Typ    | Popis |
|----------------|--------|-------|
| `image_base64` | string | Base64 obrázku (bez `data:` prefixu, bez newlinů) |
| `image_mime`   | string | `image/jpeg` \| `image/png` \| `image/webp` \| `image/gif` |

### `AnalyzeCaptureResponse`

| Pole            | Typ                       | Popis |
|-----------------|---------------------------|-------|
| `capture_id`    | string (UUID)             | ID nového/existujícího záchytu |
| `attributes`    | `ExtractedAttributes`     | Vytažená data |
| `embedding_dim` | int                       | Dimenze embeddingu (1024) |
| `linked_to`     | `LinkedCapture[]`         | Nalezené vazby |
| `processing_ms` | int                       | Doba zpracování |

### `ExtractedAttributes`

| Pole           | Typ        | Default     | Popis |
|----------------|------------|-------------|-------|
| `title`        | string     | `""`        | Krátký název |
| `summary`      | string     | `""`        | Shrnutí 2–3 věty |
| `raw_text`     | string     | `""`        | Veškerý čitelný text |
| `capture_type` | string     | `unknown`   | `screenshot` \| `photo` \| `document` \| `unknown` |
| `topics`       | string[]   | `[]`        | Témata (vč. doménové kategorie) |
| `people`       | string[]   | `[]`        | Zmíněné osoby |
| `places`       | string[]   | `[]`        | Místa / značky |
| `language`     | string     | `"en"`      | ISO-639-1 kód |
| `sentiment`    | string     | `"neutral"` | `positive` \| `negative` \| `neutral` |
| `confidence`   | float      | `1.0`       | Jistota extrakce 0–1 |

### `LinkedCapture`

| Pole         | Typ        | Popis |
|--------------|------------|-------|
| `capture_id` | string     | ID protistrany |
| `link_type`  | `LinkType` | Typ vazby (pro zobrazení) |
| `strength`   | float      | Síla 0–1 |
| `title`      | string     | Název protistrany |
| `summary`    | string     | Shrnutí protistrany |
| `reasons`    | string[]   | Vysvětlení vazby („proč") |

### `CaptureLink` (v detailu)

Jako `LinkedCapture`, ale s `source_id`, `target_id`, `other_title`,
`other_summary`, `shared_topics[]`, `reasons[]`.

### `CaptureDetailResponse`

| Pole          | Typ              | Popis |
|---------------|------------------|-------|
| `capture`     | `CaptureNode`    | Záchyt (vč. embeddingu) |
| `connections` | `CaptureLink[]`  | Vazby seřazené dle síly |
| `subject`     | `SubjectInfo`?   | Větev předmětu nebo `null` |

### `SubjectInfo`

| Pole         | Typ                | Popis |
|--------------|--------------------|-------|
| `subject_id` | string             | ID předmětu |
| `name`       | string             | Kanonický název |
| `kind`       | string             | `produkt` \| `věc` |
| `siblings`   | `SubjectSibling[]` | Stejná věc na jiných zdrojích (`id`, `title`, `source`) |

### `SearchResult` / `SearchResponse`

`SearchResult`: `capture_id`, `title`, `summary`, `score` (0–1), `created_at`.
`SearchResponse`: `query`, `results: SearchResult[]`, `total`.

---

## Typy vazeb

Hodnota `link_type` (enum `LinkType`) určuje ikonu/štítek vazby:

| Hodnota                 | Význam |
|-------------------------|--------|
| `TOPIC_OVERLAP`         | Společná konkrétní témata |
| `EMBEDDING_SIMILARITY`  | Významová (sémantická) podobnost / doménová kategorie / téměř duplikát |
| `ENTITY_MATCH`          | Stejná konkrétní osoba / místo / značka |
| `TEMPORAL_PROXIMITY`    | Kontextová vazba (místo/čas) — připraveno pro iOS geo/session signály |

Síla vazby vzniká **akumulací důkazů** — několik slabých signálů (doména +
podobnost) se sčítá; obecně: < 0.35 = slabá vazba, ≥ 0.35 = silná.

---

## Chyby

Všechny chyby mají jednotný tvar (`ErrorResponse`):

```json
{ "error": "Claude API error", "detail": "…", "code": 502 }
```

| Status | Význam |
|--------|--------|
| `400` / `422` | Neplatný vstup (validace) |
| `401`  | Neplatný Anthropic API klíč |
| `404`  | Záchyt nenalezen |
| `429`  | Rate limit Claude API |
| `502`  | Chyba Claude API |
| `503`  | Chyba databáze (Neo4j) |
| `500`  | Neočekávaná chyba serveru |

---

## Typický workflow

```text
1. POST /analyze-capture        ← nahraj screenshot/foto (opakuj pro každý)
2. POST /group-subjects         ← seskup stejné věci z různých zdrojů do větví
3. GET  /search?q=…             ← vyhledávej (česky i anglicky)
4. GET  /captures/{id}          ← detail + vazby + větev
5. GET  /graph-data             ← vykresli celý graf

Údržba po hromadných změnách:
  POST /reanalyze-topics/{id}   → POST /reembed → POST /relink
```

**Pořadí při hromadné reanalýze** je důležité kvůli Voyage rate limitu:
nejdřív reanalyzuj témata (jen Claude), pak jednou dávkově přepočítej embeddingy
(`/reembed`), nakonec přepoj vazby (`/relink`).
