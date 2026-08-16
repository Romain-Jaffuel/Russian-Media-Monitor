"""DuckDB storage layer."""
from pathlib import Path
import duckdb

DB_PATH = Path("data/russia.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id          VARCHAR PRIMARY KEY,
    source_name VARCHAR,
    feed_url    VARCHAR,
    url         VARCHAR UNIQUE,
    title       VARCHAR,
    author      VARCHAR,
    summary     TEXT,
    content     TEXT,
    language    VARCHAR,
    published_at TIMESTAMP,
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_html    TEXT,
    pays             VARCHAR,
    type_media       VARCHAR,
    statut_legal_ru  VARCHAR
);
-- ALTER plutot que dans le CREATE : la base existe deja en prod, et
-- CREATE TABLE IF NOT EXISTS ne retro-ajoute pas de colonne a une table
-- deja creee. 'press' par defaut pour tous les articles collectes avant
-- l'introduction des sources Telegram.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS source_kind VARCHAR DEFAULT 'press';
-- Vues de la video/emission parente, quand la plateforme les publie (YouTube,
-- RuTube). Repetee sur chaque segment d'une meme video : c'est une propriete
-- du parent, a agreger avec MAX par video et non a sommer.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS view_count BIGINT;
CREATE INDEX IF NOT EXISTS idx_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_source    ON articles(source_name);
CREATE INDEX IF NOT EXISTS idx_language  ON articles(language);
CREATE INDEX IF NOT EXISTS idx_type_media ON articles(type_media);
"""


def get_conn(db_path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        conn.execute(SCHEMA)
    return conn
