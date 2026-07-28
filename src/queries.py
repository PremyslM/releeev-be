"""
Cypher query library.

All queries return Neo4j result records; callers are responsible
for session management.
"""
from __future__ import annotations

# ── Topic overlap ─────────────────────────────────────────────────────────────

TOPIC_OVERLAP = """
MATCH (c:Capture)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other:Capture)
WHERE c.id = $capture_id
  AND other.id <> $capture_id
  AND other.user_id = c.user_id
WITH other, collect(DISTINCT t.name) AS shared_names
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       shared_names,
       size(shared_names)  AS shared_topics,
       'TOPIC_OVERLAP'     AS link_type
ORDER BY shared_topics DESC
LIMIT $limit
"""

# ── Entity matching (same Person or Place) ───────────────────────────────────

ENTITY_MATCH = """
MATCH (c:Capture)-[:MENTIONS]->(e)<-[:MENTIONS]-(other:Capture)
WHERE c.id = $capture_id
  AND other.id <> $capture_id
  AND other.user_id = c.user_id
  AND (e:Person OR e:Place)
WITH other, COLLECT(DISTINCT e.name) AS shared_entities
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       shared_entities,
       SIZE(shared_entities)          AS entity_count,
       'ENTITY_MATCH'      AS link_type
ORDER BY entity_count DESC
LIMIT $limit
"""

# ── Temporal proximity (< 90 days) ───────────────────────────────────────────

TEMPORAL_PROXIMITY = """
MATCH (c:Capture), (other:Capture)
WHERE c.id = $capture_id
  AND other.id <> $capture_id
  AND other.user_id = c.user_id
  AND abs(duration.between(c.created_at, other.created_at).days) < $days
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       abs(duration.between(c.created_at, other.created_at).days) AS days_apart,
       'TEMPORAL_PROXIMITY' AS link_type
ORDER BY days_apart ASC
LIMIT $limit
"""

# ── Embedding (cosine) similarity via Neo4j vector index ─────────────────────

EMBEDDING_SIMILARITY = """
CALL db.index.vector.queryNodes(
    'capture_embedding',
    $k,
    $embedding
) YIELD node AS other, score
WHERE other.id <> $capture_id
  AND score >= $threshold
WITH other, score
// Only link within the same user's brain (no cross-user leakage).
MATCH (owner:Capture {id: $capture_id})
WHERE other.user_id = owner.user_id
WITH other, score
// Count how many topics the two captures share — used for corroboration.
// A high embedding score alone can be misleading (e.g. all e-shop product
// pages look similar); requiring topic overlap keeps only meaningful links.
OPTIONAL MATCH (src:Capture {id: $capture_id})-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other)
WITH other, score, collect(DISTINCT t.name) AS shared_names
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       score               AS similarity,
       shared_names        AS shared_names,
       size(shared_names)  AS shared_topics,
       'EMBEDDING_SIMILARITY' AS link_type
ORDER BY score DESC
"""

# ── Upsert helpers ────────────────────────────────────────────────────────────

# ── Users (auth) ──────────────────────────────────────────────────────────────
CREATE_USER = """
CREATE (u:User {id: $id, email: $email, password_hash: $password_hash,
                created_at: datetime()})
RETURN u.id AS id
"""

GET_USER_BY_EMAIL = """
MATCH (u:User {email: $email})
RETURN u.id AS id, u.email AS email, u.password_hash AS password_hash
"""

UPSERT_CAPTURE = """
MERGE (c:Capture {id: $id})
SET   c.user_id      = $user_id,
      c.title        = $title,
      c.summary      = $summary,
      c.raw_text     = $raw_text,
      c.image_hash   = $image_hash,
      c.language     = $language,
      c.sentiment    = $sentiment,
      c.confidence   = $confidence,
      c.capture_type = $capture_type,
      c.kind         = $kind,
      c.author       = $author,
      c.created_at   = datetime($created_at),
      c.embedding    = $embedding
RETURN c
"""

SET_CAPTURE_CONTEXT = """
MATCH (c:Capture {id: $id})
SET c.context = $context
"""

DELETE_CAPTURE = """
MATCH (c:Capture {id: $id}) WHERE c.user_id = $uid
DETACH DELETE c
"""

# Manual edit of the user-facing text (title / summary), with a refreshed embedding.
UPDATE_CAPTURE_TEXT = """
MATCH (c:Capture {id: $id}) WHERE c.user_id = $uid
SET c.title = $title, c.summary = $summary, c.embedding = $embedding
RETURN c
"""

# Full re-extraction: refresh every model-produced field + embedding in place.
UPDATE_CAPTURE_EXTRACTION = """
MATCH (c:Capture {id: $id}) WHERE c.user_id = $uid
SET c.title     = $title,
    c.summary   = $summary,
    c.raw_text  = $raw_text,
    c.language  = $language,
    c.kind      = $kind,
    c.author    = $author,
    c.embedding = $embedding
RETURN c
"""

# Assign all still-unowned captures to a user (one-off migration).
CLAIM_ORPHAN_CAPTURES = """
MATCH (c:Capture) WHERE c.user_id IS NULL
SET c.user_id = $user_id
RETURN count(c) AS n
"""

UPSERT_TOPIC = """
MERGE (t:Topic {name: $name})
ON CREATE SET t.capture_count = 1
ON MATCH  SET t.capture_count = t.capture_count + 1
WITH t
MATCH (c:Capture {id: $capture_id})
MERGE (c)-[:HAS_TOPIC]->(t)
"""

UPSERT_PERSON = """
MERGE (p:Person {name: $name})
ON CREATE SET p.capture_count = 1
ON MATCH  SET p.capture_count = p.capture_count + 1
WITH p
MATCH (c:Capture {id: $capture_id})
MERGE (c)-[:MENTIONS]->(p)
"""

UPSERT_PLACE = """
MERGE (pl:Place {name: $name})
ON CREATE SET pl.capture_count = 1
ON MATCH  SET pl.capture_count = pl.capture_count + 1
WITH pl
MATCH (c:Capture {id: $capture_id})
MERGE (c)-[:MENTIONS]->(pl)
"""

CREATE_LINK = """
MATCH (a:Capture {id: $source_id}), (b:Capture {id: $target_id})
MERGE (a)-[r:LINKED_TO {type: $link_type}]->(b)
SET   r.strength = $strength,
      r.reasons = $reasons,
      r.updated_at = datetime()
"""

GET_CAPTURE_BY_ID = """
MATCH (c:Capture {id: $id})
WHERE c.user_id = $uid
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
OPTIONAL MATCH (c)-[:MENTIONS]->(p:Person)
OPTIONAL MATCH (c)-[:MENTIONS]->(pl:Place)
RETURN c,
       collect(DISTINCT t.name)  AS topics,
       collect(DISTINCT p.name)  AS people,
       collect(DISTINCT pl.name) AS places
"""

GET_CAPTURE_BY_HASH = """
MATCH (c:Capture {image_hash: $image_hash})
WHERE c.user_id = $uid
RETURN c
ORDER BY c.created_at DESC
LIMIT 1
"""

GET_CAPTURE_CONNECTIONS = """
MATCH (c:Capture {id: $id})-[r:LINKED_TO]-(other:Capture)
WHERE other.user_id = c.user_id
// A pair may have two directed relationships (a→b and b→a); keep the strongest
WITH c, other, r ORDER BY r.strength DESC
WITH c, other, head(collect(r)) AS r
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other)
RETURN r.type       AS link_type,
       r.strength   AS strength,
       r.reasons    AS reasons,
       other.id     AS other_id,
       other.title  AS other_title,
       other.summary AS other_summary,
       collect(DISTINCT t.name) AS shared_topics
ORDER BY r.strength DESC
"""

FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes('capture_fulltext', $search_text)
YIELD node, score
WHERE node.user_id = $uid
RETURN node.id      AS capture_id,
       node.title   AS title,
       node.summary AS summary,
       node.created_at AS created_at,
       score
ORDER BY score DESC
LIMIT $limit
"""

# ── Semantic (embedding) search ───────────────────────────────────────────────
SEMANTIC_SEARCH = """
CALL db.index.vector.queryNodes('capture_embedding', $k, $embedding)
YIELD node, score
WHERE node.user_id = $uid
RETURN node.id      AS capture_id,
       node.title   AS title,
       node.summary AS summary,
       node.created_at AS created_at,
       score
ORDER BY score DESC
LIMIT $limit
"""

# ── Topic-based boost search ──────────────────────────────────────────────────
# Finds Captures directly connected to a Topic that matches ANY of the given keywords
TOPIC_BOOST_SEARCH = """
UNWIND $keywords AS keyword
MATCH (t:Topic)
WHERE toLower(t.name) CONTAINS toLower(keyword)
MATCH (c:Capture)-[:HAS_TOPIC]->(t)
WHERE c.user_id = $uid
RETURN DISTINCT
       c.id       AS capture_id,
       c.title    AS title,
       c.summary  AS summary,
       c.created_at AS created_at,
       0.4        AS score
LIMIT $limit
"""

# ── Get all captures for re-embedding ────────────────────────────────────────
GET_ALL_CAPTURES_FOR_REEMBED = """
MATCH (c:Capture) WHERE c.user_id = $uid
RETURN c.id AS id, c.title AS title, c.summary AS summary,
       c.embedding AS embedding
ORDER BY c.created_at ASC
"""

# All of a user's captures with their stored text — for the topic backfill.
GET_ALL_CAPTURES_TEXT = """
MATCH (c:Capture) WHERE c.user_id = $uid
RETURN c.id AS id, c.title AS title, c.summary AS summary, c.raw_text AS raw_text
ORDER BY c.created_at ASC
"""

UPDATE_EMBEDDING = """
MATCH (c:Capture {id: $id})
SET c.embedding = $embedding
"""

# ── Delete all LINKED_TO relationships (before full relink) ──────────────────
DELETE_ALL_LINKS = """
MATCH (a:Capture)-[r:LINKED_TO]->(b:Capture)
WHERE a.user_id = $uid
DELETE r
"""

# ── Get all captures with their embeddings (for relink) ──────────────────────
GET_ALL_CAPTURES_FOR_RELINK = """
MATCH (c:Capture) WHERE c.user_id = $uid
RETURN c.id AS id, c.embedding AS embedding
ORDER BY c.created_at ASC
"""

# ── Get all captures with title + text + embedding (dedup / grouping) ────────
GET_ALL_CAPTURES_FULL = """
MATCH (c:Capture) WHERE c.user_id = $uid
RETURN c.id AS id, c.title AS title, c.summary AS summary,
       c.raw_text AS raw_text, c.embedding AS embedding,
       coalesce(c.merged_count, 1) AS merged_count
ORDER BY c.created_at ASC
"""

# ── Merge near-duplicate captures: move all edges from DROP into KEEP ────────
# Run these in order, then delete the duplicate.
MERGE_MOVE_TOPICS = """
MATCH (b:Capture {id: $drop})-[:HAS_TOPIC]->(t:Topic)
MATCH (a:Capture {id: $keep})
MERGE (a)-[:HAS_TOPIC]->(t)
"""

MERGE_MOVE_MENTIONS = """
MATCH (b:Capture {id: $drop})-[:MENTIONS]->(e)
MATCH (a:Capture {id: $keep})
MERGE (a)-[:MENTIONS]->(e)
"""

MERGE_MOVE_LINKS_OUT = """
MATCH (b:Capture {id: $drop})-[r:LINKED_TO]->(x:Capture)
WHERE x.id <> $keep
MATCH (a:Capture {id: $keep})
MERGE (a)-[nr:LINKED_TO {type: r.type}]->(x)
ON CREATE SET nr.strength = r.strength, nr.reasons = r.reasons, nr.updated_at = datetime()
"""

MERGE_MOVE_LINKS_IN = """
MATCH (x:Capture)-[r:LINKED_TO]->(b:Capture {id: $drop})
WHERE x.id <> $keep
MATCH (a:Capture {id: $keep})
MERGE (x)-[nr:LINKED_TO {type: r.type}]->(a)
ON CREATE SET nr.strength = r.strength, nr.reasons = r.reasons, nr.updated_at = datetime()
"""

MERGE_FINALIZE = """
MATCH (a:Capture {id: $keep}), (b:Capture {id: $drop})
SET a.merged_count = coalesce(a.merged_count, 1) + coalesce(b.merged_count, 1),
    a.merged_ids   = coalesce(a.merged_ids, []) + [b.id] + coalesce(b.merged_ids, [])
DETACH DELETE b
"""

# ── Subject branches (group same-entity captures across sources) ─────────────
# A :Subject is the real-world thing (a product, a person, a place…). Captures
# that depict the same Subject hang off it via (Capture)-[:DEPICTS]->(Subject),
# so the graph shows "this washing machine, seen on Alza / eshop X / my photo".

SET_CAPTURE_SOURCE = """
MATCH (c:Capture {id: $id})
SET c.source = $source
"""

SET_CAPTURE_URL = """
MATCH (c:Capture {id: $id})
SET c.url = $url
"""

# Link preview: the page's own hero image / self-description, so the source page
# shows a real card instead of a bare domain.
SET_CAPTURE_PREVIEW = """
MATCH (c:Capture {id: $id})
SET c.og_image = $og_image,
    c.site_name = $site_name,
    c.page_description = $page_description
"""

# ── Capture location (place-anchored memory) ─────────────────────────────────
SET_CAPTURE_LOCATION = """
MATCH (c:Capture {id: $id})
SET c.lat = $lat, c.lon = $lon,
    c.place = $place, c.locality = $locality,
    c.captured_at = $captured_at
"""

# Group captures made at the same venue under a :Location node.
UPSERT_LOCATION = """
MERGE (l:Location {name: $name})
ON CREATE SET l.locality = $locality, l.lat = $lat, l.lon = $lon, l.capture_count = 1
ON MATCH  SET l.capture_count = coalesce(l.capture_count, 0) + 1
WITH l
MATCH (c:Capture {id: $id})
MERGE (c)-[:AT_LOCATION]->(l)
"""

UPSERT_SUBJECT = """
MERGE (s:Subject {id: $id})
ON CREATE SET s.name = $name, s.kind = $kind, s.created_at = datetime()
SET s.name = $name, s.kind = $kind
WITH s
UNWIND $capture_ids AS cid
MATCH (c:Capture {id: cid})
MERGE (c)-[:DEPICTS]->(s)
WITH s
MATCH (c2:Capture)-[:DEPICTS]->(s)
WITH s, count(c2) AS cnt
SET s.capture_count = cnt
"""

# Remove this user's Subjects + DEPICTS (rebuild from scratch)
CLEAR_SUBJECTS = """
MATCH (s:Subject)<-[:DEPICTS]-(c:Capture {user_id: $uid})
WITH DISTINCT s
DETACH DELETE s
"""

# A capture's subject + its sibling captures (other sources of the same thing)
GET_CAPTURE_SUBJECT = """
MATCH (c:Capture {id: $id})-[:DEPICTS]->(s:Subject)
MATCH (sib:Capture)-[:DEPICTS]->(s)
WHERE sib.user_id = c.user_id
RETURN s.id AS subject_id, s.name AS subject_name, s.kind AS subject_kind,
       collect({id: sib.id, title: sib.title, source: sib.source}) AS siblings
"""

# All subjects with their captures (for graph visualization)
GET_SUBJECTS_GRAPH = """
MATCH (s:Subject)<-[:DEPICTS]-(c:Capture)
WHERE c.user_id = $uid
RETURN s.id AS subject_id, s.name AS subject_name, s.kind AS subject_kind,
       collect(c.id) AS capture_ids
"""

# Find an existing Subject that any of the given captures already belongs to
GET_SUBJECT_FOR_CAPTURES = """
MATCH (c:Capture)-[:DEPICTS]->(s:Subject)
WHERE c.id IN $ids
RETURN s.id AS subject_id, s.name AS subject_name, s.kind AS subject_kind
LIMIT 1
"""

# ── Entity navigation (browsable hubs) ───────────────────────────────────────

LIST_TOPICS = """
MATCH (t:Topic)<-[:HAS_TOPIC]-(c:Capture)
WHERE c.user_id = $uid
RETURN t.name AS name, count(c) AS cnt
ORDER BY cnt DESC, name ASC LIMIT $limit
"""

LIST_PEOPLE = """
MATCH (p:Person)<-[:MENTIONS]-(c:Capture)
WHERE c.user_id = $uid
RETURN p.name AS name, count(c) AS cnt
ORDER BY cnt DESC, name ASC LIMIT $limit
"""

LIST_PLACES = """
MATCH (pl:Place)<-[:MENTIONS]-(c:Capture)
WHERE c.user_id = $uid
RETURN pl.name AS name, count(c) AS cnt
ORDER BY cnt DESC, name ASC LIMIT $limit
"""

LIST_SUBJECTS = """
MATCH (s:Subject)<-[:DEPICTS]-(c:Capture)
WHERE c.user_id = $uid
RETURN s.name AS name, count(c) AS cnt
ORDER BY cnt DESC, name ASC LIMIT $limit
"""

# ── Locations (place-anchored memories) ──────────────────────────────────────
LIST_LOCATIONS = """
MATCH (l:Location)<-[:AT_LOCATION]-(c:Capture)
WHERE c.user_id = $uid
RETURN l.name AS name, l.locality AS locality, l.lat AS lat, l.lon AS lon, count(c) AS cnt
ORDER BY cnt DESC, name ASC LIMIT $limit
"""

# ── Journal (captures grouped by day) ────────────────────────────────────────
LIST_JOURNAL_DAYS = """
MATCH (c:Capture)
WHERE c.user_id = $uid AND c.created_at IS NOT NULL
WITH date(c.created_at) AS d, count(c) AS n, collect(c.title) AS titles
RETURN toString(d) AS date, n AS count,
       head([t IN titles WHERE t IS NOT NULL AND t <> ""]) AS preview
ORDER BY d DESC LIMIT $limit
"""

GET_DAY_CAPTURES = """
MATCH (c:Capture)
WHERE c.user_id = $uid AND c.created_at IS NOT NULL AND date(c.created_at) = date($d)
RETURN c.id AS id, c.title AS title, c.summary AS summary,
       c.capture_type AS capture_type, c.kind AS kind, c.created_at AS created_at
ORDER BY c.created_at ASC
"""

# Captures made at the same venue (feeds the CO_LOCATED / same_place link signal).
LOCATION_MATCH = """
MATCH (c:Capture {id: $capture_id})-[:AT_LOCATION]->(l:Location)<-[:AT_LOCATION]-(other:Capture)
WHERE other.id <> $capture_id
  AND other.user_id = c.user_id
RETURN other.id      AS capture_id,
       other.title   AS title,
       other.summary AS summary
LIMIT $limit
"""

# Captures referencing a given entity. {label}/{rel} are injected from a
# whitelist (topic→Topic/HAS_TOPIC, person/place→Person/Place/MENTIONS).
ENTITY_CAPTURES_TMPL = """
MATCH (e:{label} {{name: $name}})<-[:{rel}]-(c:Capture)
WHERE c.user_id = $uid
RETURN c.id AS id, c.title AS title, c.summary AS summary
ORDER BY c.created_at DESC
LIMIT $limit
"""

# Entities co-occurring with the given one (shared captures) — for hopping.
ENTITY_RELATED_TMPL = """
MATCH (e:{label} {{name: $name}})<-[:{rel}]-(c:Capture)-[:HAS_TOPIC|MENTIONS]->(o)
WHERE c.user_id = $uid AND (o:Topic OR o:Person OR o:Place) AND o.name <> $name
RETURN labels(o)[0] AS lbl, o.name AS name, count(DISTINCT c) AS cnt
ORDER BY cnt DESC, name ASC
LIMIT 15
"""

# ── Ask (RAG): fetch full context for the retrieved captures ─────────────────
GET_CAPTURES_FOR_ASK = """
MATCH (c:Capture)
WHERE c.id IN $ids AND c.user_id = $uid
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
RETURN c.id           AS id,
       c.title        AS title,
       c.summary      AS summary,
       c.raw_text     AS raw_text,
       c.capture_type AS capture_type,
       c.created_at   AS created_at,
       collect(toLower(t.name)) AS topics
"""

# Most-recently-added captures — the "Recent logs" browse list in search.
RECENT_CAPTURES = """
MATCH (c:Capture)
WHERE c.user_id = $uid
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
WITH c, collect(toLower(t.name)) AS topics
RETURN c.id           AS id,
       c.title        AS title,
       c.summary      AS summary,
       c.capture_type AS capture_type,
       c.created_at   AS created_at,
       topics
ORDER BY c.created_at DESC
LIMIT $limit
"""

# ── Living topic documents ───────────────────────────────────────────────────
# A topic's captures for this user, oldest first so the compiled document reads
# as the story of how their thinking developed.
TOPIC_CAPTURES = """
MATCH (t:Topic {name: $topic})<-[:HAS_TOPIC]-(c:Capture)
WHERE c.user_id = $uid
RETURN c.id           AS id,
       c.title        AS title,
       c.summary      AS summary,
       c.kind         AS kind,
       c.capture_type AS capture_type,
       c.created_at   AS created_at
ORDER BY c.created_at ASC
LIMIT $limit
"""

# The compiled document is cached per (user, topic). Topic nodes are global, so
# it can't live on the Topic itself — TopicDoc is the user's own copy.
GET_TOPIC_DOC = """
MATCH (d:TopicDoc {user_id: $uid, topic: $topic})
RETURN d.text AS text, d.capture_count AS capture_count, d.updated_at AS updated_at
"""

UPSERT_TOPIC_DOC = """
MERGE (d:TopicDoc {user_id: $uid, topic: $topic})
SET d.text = $text, d.capture_count = $capture_count, d.updated_at = datetime()
"""

# The user's topics by how many of THEIR captures use them (t.capture_count is a
# global counter shared across users, so it can't be trusted for this).
USER_TOPICS = """
MATCH (c:Capture)-[:HAS_TOPIC]->(t:Topic)
WHERE c.user_id = $uid
WITH t.name AS name, count(c) AS cnt
WHERE cnt >= $min_captures
RETURN name, cnt
ORDER BY cnt DESC, name ASC
LIMIT $limit
"""

# ── Faceted search: fetch filterable fields for a set of candidate captures ──
GET_CAPTURE_FACETS = """
MATCH (c:Capture)
WHERE c.id IN $ids AND c.user_id = $uid
OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
RETURN c.id           AS id,
       c.language     AS language,
       c.source       AS source,
       c.capture_type AS capture_type,
       c.created_at   AS created_at,
       collect(toLower(t.name)) AS topics
"""
