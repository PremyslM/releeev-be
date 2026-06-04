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
WITH other, COUNT(t) AS shared_topics
WHERE shared_topics >= $min_shared
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       shared_topics,
       'TOPIC_OVERLAP'     AS link_type
ORDER BY shared_topics DESC
LIMIT $limit
"""

# ── Entity matching (same Person or Place) ───────────────────────────────────

ENTITY_MATCH = """
MATCH (c:Capture)-[:MENTIONS]->(e)<-[:MENTIONS]-(other:Capture)
WHERE c.id = $capture_id
  AND other.id <> $capture_id
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
RETURN other.id            AS capture_id,
       other.title         AS title,
       other.summary       AS summary,
       score               AS similarity,
       'EMBEDDING_SIMILARITY' AS link_type
ORDER BY score DESC
"""

# ── Upsert helpers ────────────────────────────────────────────────────────────

UPSERT_CAPTURE = """
MERGE (c:Capture {id: $id})
SET   c.title      = $title,
      c.summary    = $summary,
      c.raw_text   = $raw_text,
      c.image_hash = $image_hash,
      c.language   = $language,
      c.sentiment  = $sentiment,
      c.confidence = $confidence,
      c.created_at = datetime($created_at),
      c.embedding  = $embedding
RETURN c
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
      r.updated_at = datetime()
"""

GET_CAPTURE_BY_ID = """
MATCH (c:Capture {id: $id})
RETURN c
"""

GET_CAPTURE_BY_HASH = """
MATCH (c:Capture {image_hash: $image_hash})
RETURN c
ORDER BY c.created_at DESC
LIMIT 1
"""

GET_CAPTURE_CONNECTIONS = """
MATCH (c:Capture {id: $id})-[r:LINKED_TO]-(other:Capture)
RETURN c, r, other
"""

FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes('capture_fulltext', $search_text)
YIELD node, score
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
RETURN node.id      AS capture_id,
       node.title   AS title,
       node.summary AS summary,
       node.created_at AS created_at,
       score
ORDER BY score DESC
LIMIT $limit
"""

# ── Topic-based boost search ──────────────────────────────────────────────────
# Finds Captures directly connected to a Topic that matches the query keyword
TOPIC_BOOST_SEARCH = """
MATCH (t:Topic)
WHERE toLower(t.name) CONTAINS toLower($keyword)
MATCH (c:Capture)-[:HAS_TOPIC]->(t)
RETURN DISTINCT
       c.id       AS capture_id,
       c.title    AS title,
       c.summary  AS summary,
       c.created_at AS created_at,
       1.5        AS score
LIMIT $limit
"""
