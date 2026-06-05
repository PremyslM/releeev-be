"""
Seed the Brain Graph with ~50 realistic synthetic captures for testing.

Inserts captures directly into Neo4j (skipping Claude Vision) but computes REAL
multilingual embeddings via Voyage in a SINGLE batched request — so it stays well
under the free-tier 3 RPM limit.

The dataset deliberately exercises every feature:
  • many domains (automotive, programming, fashion, watches, appliances, …)
  • shared specific topics  → strong links
  • shared domain only      → weak category links
  • same product, several sources → Subject branches (/group-subjects)
  • near-identical pairs    → duplicate merging (/merge-duplicates)

Run:  docker exec brain_graph_api python -m scripts.seed_data
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta

from src import queries as Q
from src.neo4j_driver import get_session, close_driver
from src.vision_extractor import getEmbeddingsBatch

# (title, summary, category, [specific topics], [people], [places], lang)
CAPTURES: list[tuple] = [
    # ── automotive ──────────────────────────────────────────────────────────
    ("BMW i4 eDrive40 Configurator", "Konfigurátor elektrického BMW i4 s dojezdem 590 km a cenou od 1 390 000 Kč.", "automotive", ["BMW", "BMW i4", "electric vehicle", "car configurator"], [], ["Czech Republic"], "cs"),
    ("Audi Q4 e-tron Review – CarWow", "Video review elektrického SUV Audi Q4 e-tron, srovnání s Teslou Model Y.", "automotive", ["Audi", "Audi Q4", "electric SUV", "car review"], ["Mat Watson"], [], "en"),
    ("Škoda Enyaq iV 80 na Sauto.cz", "Inzerát ojetého elektromobilu Škoda Enyaq iV, rok 2023, 35 000 km.", "automotive", ["Škoda", "Škoda Enyaq", "electric vehicle", "used car"], [], ["Czech Republic"], "cs"),
    ("Volkswagen ID.3 Pro Performance", "Produktová stránka VW ID.3 s baterií 58 kWh a dojezdem 426 km.", "automotive", ["Volkswagen", "VW ID.3", "electric vehicle"], [], [], "cs"),
    ("Porsche Taycan Turbo S YouTube", "YouTube recenze sportovního elektromobilu Porsche Taycan Turbo S.", "automotive", ["Porsche", "Porsche Taycan", "electric vehicle", "sports car"], [], [], "en"),

    # ── programming ─────────────────────────────────────────────────────────
    ("Python list comprehension – Stack Overflow", "Stack Overflow odpověď na použití list comprehension v Pythonu pro filtrování.", "programming", ["Python", "list comprehension", "Stack Overflow"], [], [], "en"),
    ("React useEffect Hook Documentation", "Oficiální React dokumentace k useEffect hooku a cleanup funkcím.", "programming", ["React", "useEffect", "JavaScript", "hooks"], [], [], "en"),
    ("Docker Compose Tutorial – freeCodeCamp", "Tutoriál na orchestraci kontejnerů pomocí docker-compose.yml.", "programming", ["Docker", "Docker Compose", "DevOps", "containers"], [], [], "en"),
    ("Rust Ownership Explained", "Článek vysvětlující ownership a borrowing v jazyce Rust.", "programming", ["Rust", "ownership", "memory safety"], [], [], "en"),
    ("SQL JOIN Types – W3Schools", "W3Schools přehled INNER, LEFT, RIGHT a FULL JOIN s příklady.", "programming", ["SQL", "JOIN", "database", "W3Schools"], [], [], "en"),
    ("Git rebase vs merge – Atlassian", "Atlassian tutoriál srovnávající git rebase a git merge.", "programming", ["Git", "rebase", "version control"], [], [], "en"),

    # ── fashion ─────────────────────────────────────────────────────────────
    ("Adidas Samba OG na Footshop", "Produktová stránka tenisek Adidas Samba OG, černobílá, 2 790 Kč.", "fashion", ["Adidas", "Adidas Samba", "sneakers", "footwear"], [], ["Czech Republic"], "cs"),
    ("Converse Chuck Taylor All Star", "Klasické plátěné boty Converse Chuck Taylor, vysoký střih.", "fashion", ["Converse", "Chuck Taylor", "sneakers", "footwear"], [], [], "en"),
    ("Levi's 501 Original Jeans", "Produktová stránka riflí Levi's 501 Original Fit, modré.", "fashion", ["Levi's", "501 jeans", "denim", "apparel"], [], [], "en"),
    ("The North Face Nuptse Jacket", "Zimní péřová bunda The North Face Nuptse 700, černá.", "fashion", ["The North Face", "winter jacket", "outerwear"], [], [], "en"),
    ("Zara Wool Blend Coat", "Dámský vlněný kabát Zara, béžová barva, kolekce podzim.", "fashion", ["Zara", "wool coat", "apparel"], [], [], "en"),

    # ── watches ─────────────────────────────────────────────────────────────
    ("Casio G-Shock GA-2100 Listing", "Inzerát hodinek Casio G-Shock GA-2100 'CasiOak', černé.", "watches", ["Casio", "G-Shock", "GA-2100", "wristwatch"], [], [], "en"),
    ("Seiko 5 SNK809 Automatic", "Automatické hodinky Seiko 5 SNK809 s textilním páskem.", "watches", ["Seiko", "Seiko 5", "automatic watch", "wristwatch"], [], [], "en"),
    ("Garmin Fenix 7 Sapphire Solar", "Sportovní GPS hodinky Garmin Fenix 7 se solárním nabíjením.", "watches", ["Garmin", "Fenix 7", "GPS watch", "smartwatch"], [], [], "en"),
    ("Orient Bambino V2 Dress Watch", "Společenské hodinky Orient Bambino V2 s klenutým sklíčkem.", "watches", ["Orient", "Bambino", "dress watch", "automatic watch"], [], [], "en"),

    # ── appliances ──────────────────────────────────────────────────────────
    ("Bosch Serie 6 Myčka SMV6", "Vestavná myčka nádobí Bosch Serie 6 SMV6, 14 sad nádobí, A++.", "appliances", ["Bosch", "dishwasher", "myčka nádobí"], [], ["Czech Republic"], "cs"),
    ("Samsung RB38 Lednice s mrazákem", "Kombinovaná chladnička Samsung RB38 No Frost, 390 l.", "appliances", ["Samsung", "refrigerator", "lednice"], [], [], "cs"),
    ("Philips Airfryer XXL HD9650", "Horkovzdušná fritéza Philips Airfryer XXL s kapacitou 1,4 kg.", "appliances", ["Philips", "airfryer", "horkovzdušná fritéza"], [], [], "cs"),
    ("Dyson V15 Detect Absolute", "Tyčový vysavač Dyson V15 Detect s laserovou detekcí prachu.", "appliances", ["Dyson", "vacuum cleaner", "vysavač"], [], [], "en"),
    ("Whirlpool Mikrovlnná trouba MWP", "Mikrovlnná trouba Whirlpool MWP 339 SB, 33 l, grill.", "appliances", ["Whirlpool", "microwave", "mikrovlnná trouba"], [], [], "cs"),

    # ── electronics ─────────────────────────────────────────────────────────
    ("Sony WH-1000XM5 Sluchátka", "Bezdrátová sluchátka Sony WH-1000XM5 s ANC, recenze.", "electronics", ["Sony", "WH-1000XM5", "headphones", "noise cancelling"], [], [], "en"),
    ("MacBook Air M3 13palcový", "Apple MacBook Air s čipem M3, 8GB RAM, 256GB SSD.", "electronics", ["Apple", "MacBook Air", "M3 chip", "laptop"], [], [], "en"),
    ("LG OLED C4 65 TV", "4K OLED televizor LG OLED65C4 s 120Hz a webOS.", "electronics", ["LG", "OLED TV", "4K television"], [], [], "en"),
    ("Samsung Galaxy S24 Ultra", "Smartphone Samsung Galaxy S24 Ultra s 200MP foťákem a S Pen.", "electronics", ["Samsung", "Galaxy S24", "smartphone", "Android"], [], [], "en"),

    # ── furniture ───────────────────────────────────────────────────────────
    ("IKEA KALLAX Regál", "Regál IKEA KALLAX 4x4 přihrádky, bílá, 147x147 cm.", "furniture", ["IKEA", "KALLAX", "shelf", "regál"], [], [], "cs"),
    ("IKEA POÄNG Křeslo", "Houpací křeslo IKEA POÄNG s ohýbaným rámem a polštářem.", "furniture", ["IKEA", "POÄNG", "armchair", "křeslo"], [], [], "cs"),
    ("JYSK Rozkládací pohovka GISTRUP", "Rozkládací rohová pohovka JYSK GISTRUP, šedá.", "furniture", ["JYSK", "sofa", "pohovka"], [], [], "cs"),

    # ── gaming ──────────────────────────────────────────────────────────────
    ("Cyberpunk 2077 na Steam", "Steam stránka hry Cyberpunk 2077 Ultimate Edition, sleva 50%.", "gaming", ["Cyberpunk 2077", "Steam", "RPG", "video game"], [], [], "en"),
    ("Elden Ring Shadow of the Erdtree", "DLC Elden Ring: Shadow of the Erdtree, recenze a hodnocení.", "gaming", ["Elden Ring", "DLC", "souls-like", "video game"], [], [], "en"),
    ("PlayStation 5 Slim Konzole", "Herní konzole PlayStation 5 Slim s diskovou mechanikou.", "gaming", ["PlayStation 5", "PS5", "console", "gaming"], [], [], "en"),
    ("Nintendo Switch OLED Model", "Nintendo Switch OLED s 7palcovým displejem, bílá.", "gaming", ["Nintendo Switch", "OLED", "handheld console"], [], [], "en"),

    # ── food ────────────────────────────────────────────────────────────────
    ("Recept na hovězí guláš", "Tradiční recept na český hovězí guláš s cibulí a paprikou.", "food", ["guláš", "beef stew", "Czech cuisine", "recipe"], [], ["Czech Republic"], "cs"),
    ("Domácí pizza Margherita recept", "Recept na domácí pizzu Margherita s rajčaty a mozzarellou.", "food", ["pizza", "Margherita", "Italian cuisine", "recipe"], [], [], "cs"),
    ("Starbucks Menu Pumpkin Spice", "Podzimní menu Starbucks s Pumpkin Spice Latte.", "food", ["Starbucks", "coffee", "latte", "menu"], [], [], "en"),

    # ── travel ──────────────────────────────────────────────────────────────
    ("Letenka Praha–Barcelona Ryanair", "Letenka Ryanair Praha do Barcelony, 1 290 Kč zpáteční.", "travel", ["Ryanair", "flight", "Barcelona", "letenka"], [], ["Barcelona", "Prague"], "cs"),
    ("Airbnb Apartmán Praha centrum", "Airbnb nabídka apartmánu v centru Prahy, 2 100 Kč/noc.", "travel", ["Airbnb", "apartment", "Prague", "accommodation"], [], ["Prague"], "cs"),
    ("Booking.com Hotel Vídeň", "Rezervace hotelu ve Vídni přes Booking.com, 4 hvězdy.", "travel", ["Booking.com", "hotel", "Vienna"], [], ["Vienna"], "en"),

    # ── finance ─────────────────────────────────────────────────────────────
    ("Revolut Mobilní Bankovnictví", "Aplikace Revolut s přehledem útrat a směnou měn.", "finance", ["Revolut", "banking app", "fintech"], [], [], "cs"),
    ("NVDA Akcie Graf TradingView", "Graf akcie NVIDIA (NVDA) na TradingView, růst o 12%.", "finance", ["NVIDIA", "NVDA", "stock", "trading"], [], [], "en"),
    ("ČSOB Smart Klíč Aplikace", "Mobilní bankovnictví ČSOB Smart s biometrickým přihlášením.", "finance", ["ČSOB", "banking app", "mobile banking"], [], ["Czech Republic"], "cs"),

    # ── photography ─────────────────────────────────────────────────────────
    ("Canon EOS R6 Mark II", "Bezzrcadlovka Canon EOS R6 Mark II, 24MP, recenze.", "photography", ["Canon", "EOS R6", "mirrorless camera"], [], [], "en"),
    ("Sony A7 IV Full Frame", "Fotoaparát Sony A7 IV s 33MP snímačem a 4K videem.", "photography", ["Sony", "A7 IV", "mirrorless camera", "full frame"], [], [], "en"),

    # ── music / health / books ──────────────────────────────────────────────
    ("Spotify Playlist Focus Flow", "Spotify playlist Focus Flow pro soustředění při práci.", "music", ["Spotify", "playlist", "focus music"], [], [], "en"),
    ("Strava Běžecká Aktivita 10K", "Záznam běhu 10 km v aplikaci Strava, tempo 5:30/km.", "health", ["Strava", "running", "fitness tracking"], [], [], "en"),
    ("Atomic Habits na Goodreads", "Kniha Atomic Habits od James Clear, hodnocení na Goodreads.", "books", ["Atomic Habits", "Goodreads", "self-help book"], ["James Clear"], [], "en"),

    # ── SUBJECT BRANCH test: stejný iPhone na 3 zdrojích ─────────────────────
    ("iPhone 15 Pro 256GB on Alza.cz", "Produktová stránka Apple iPhone 15 Pro 256GB Titan na Alza.cz, 32 990 Kč.", "electronics", ["Apple", "iPhone 15 Pro", "smartphone"], [], ["Czech Republic"], "cs"),
    ("iPhone 15 Pro 256GB on Mall.cz", "Apple iPhone 15 Pro 256GB Natural Titanium na Mall.cz, 32 490 Kč.", "electronics", ["Apple", "iPhone 15 Pro", "smartphone"], [], ["Czech Republic"], "cs"),
    ("iPhone 15 Pro – moje fotka", "Vlastní fotka krabičky Apple iPhone 15 Pro 256GB po koupi.", "electronics", ["Apple", "iPhone 15 Pro", "smartphone"], [], [], "cs"),

    # ── SUBJECT BRANCH test: Bosch myčka na 2 zdrojích ──────────────────────
    ("Bosch SMV6 Myčka on Datart.cz", "Vestavná myčka Bosch Serie 6 SMV6 na Datart.cz, 14 990 Kč.", "appliances", ["Bosch", "dishwasher", "SMV6"], [], ["Czech Republic"], "cs"),

    # ── MERGE test: near-identical Casio G-Shock ────────────────────────────
    ("Casio G-Shock GA-2100 Black", "Hodinky Casio G-Shock GA-2100-1A1 CasiOak, černé, recenze.", "watches", ["Casio", "G-Shock", "GA-2100", "wristwatch"], [], [], "en"),
]


def _hash(title: str) -> str:
    return hashlib.sha256(("seed:" + title).encode()).hexdigest()


async def main() -> None:
    print(f"Seeding {len(CAPTURES)} captures…")

    # 1. Batch-embed all summaries in ONE Voyage request (rate-limit safe)
    embed_texts = [f"{t[0]} {t[1]}" for t in CAPTURES]
    print("Computing embeddings (single batched Voyage request)…")
    embeddings = getEmbeddingsBatch(embed_texts)
    ok = sum(1 for e in embeddings if any(v != 0.0 for v in e[:5]))
    print(f"  → {ok}/{len(embeddings)} embeddings OK")

    # 2. Insert each capture + its topics/people/places
    base = datetime.utcnow()
    inserted = 0
    async with get_session() as session:
        for i, (title, summary, category, topics, people, places, lang) in enumerate(CAPTURES):
            cid = _hash(title)
            created = (base - timedelta(days=len(CAPTURES) - i, hours=i % 12)).isoformat()
            all_topics = topics + [category]  # category stored as a Topic (domain tag)

            await session.run(
                Q.UPSERT_CAPTURE,
                id=cid, title=title, summary=summary, raw_text=summary,
                image_hash=cid, language=lang, sentiment="neutral",
                confidence=0.9, created_at=created, embedding=embeddings[i],
            )
            for t in all_topics:
                await session.run(Q.UPSERT_TOPIC, name=t, capture_id=cid)
            for p in people:
                await session.run(Q.UPSERT_PERSON, name=p, capture_id=cid)
            for pl in places:
                await session.run(Q.UPSERT_PLACE, name=pl, capture_id=cid)
            inserted += 1

    print(f"Inserted {inserted} captures.")
    await close_driver()
    print("Done. Next: POST /relink, /group-subjects, /merge-duplicates")


if __name__ == "__main__":
    asyncio.run(main())
