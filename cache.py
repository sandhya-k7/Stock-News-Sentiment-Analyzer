"""SQLite cache so a text is never scored twice per engine."""
import hashlib
import sqlite3

DB = "cache.sqlite"


def score_with_cache(texts, engine):
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS s(k TEXT PRIMARY KEY, v REAL)")
    keys = [hashlib.md5(f"{engine.name}|{t}".encode()).hexdigest() for t in texts]
    got = dict(con.execute(f"SELECT k,v FROM s WHERE k IN ({','.join('?' * len(keys))})", keys).fetchall()) if keys else {}
    todo = [i for i, k in enumerate(keys) if k not in got]
    if todo:
        for i, s in zip(todo, engine.score([texts[i] for i in todo])):
            got[keys[i]] = s
        con.executemany("INSERT OR REPLACE INTO s VALUES(?,?)", [(keys[i], got[keys[i]]) for i in todo])
        con.commit()
    con.close()
    return [got[k] for k in keys]
