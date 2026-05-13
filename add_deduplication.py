#!/usr/bin/env python3
import sqlite3
import hashlib
import sys
def get_table_columns(conn, table):
    """Get list of column names in a table"""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]
def backfill_game_hashes(conn):
    """Backfill hashes using whatever columns exist"""
    cols = get_table_columns(conn, 'games')
    
    # Check what columns we have
    has_date = 'date' in cols or 'game_date' in cols
    has_event = 'event' in cols
    has_site = 'site' in cols
    has_round = 'round' in cols
    
    date_col = 'date' if 'date' in cols else 'game_date' if 'game_date' in cols else None
    
    print(f"Found columns: date={date_col}, event={has_event}, site={has_site}, round={has_round}")
    
    # Get games needing hashes
    cursor = conn.execute("SELECT game_id, white, black, result, white_elo, black_elo, site, event, ply_count, winner, round FROM games WHERE game_hash IS NULL OR game_hash = ''")
    games = cursor.fetchall()
    
    if not games:
        print("All games already have hashes")
        return
    
    print(f"Backfilling hashes for {len(games):,} games...")
    
    updated = 0
    for  game_id, white, black, result, white_elo, black_elo, site, event, ply_count, winner, round in games:
        try:
            # Use whatever metadata we have
            white = white or 'Unknown'
            black = black or 'Unknown'
            
            # Compute hash from available data
            content = f"{white}|{black}|{result}|{white_elo}|{black_elo}|{site}|{event}|{ply_count}|{winner}|{round}"
            h = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            cursor.execute("UPDATE games SET game_hash = ?, dedup_processed_at = datetime('now') WHERE game_id = ?", (h, game_id))
            updated += 1
            
            if updated % 1000 == 0:
                print(f"  Processed {updated:,}/{len(games):,} games...")
                conn.commit()
                
        except Exception as e:
            print(f"  Warning: Failed to hash game {game_id}: {e}")
            continue
    
    conn.commit()
    print(f"Backfill complete: {updated:,} games updated")
if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'chessv2.db'
    conn = sqlite3.connect(db_path)
    backfill_game_hashes(conn)
    conn.close()