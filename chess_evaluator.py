#TODO make this typed more strongly now that it isn't in cells

#Functions to take games and evaluate a game.
#Optimization to consider: No need to re-evaluate the actual move or current position just take the values from the possible move exploration and pass them around

#Other optimization to consider just use SimpleEngine and limit thinking time to .2 seconds with multipv=218

# import clean up in the future too
from typing import Sequence
import multiprocess as mp
import hashlib

import time
import sqlite3
import queue
import threading

import chess.pgn
from stockfish import Stockfish

def push_headers(queue, game, count):
    #TODO: Update this to use parameterized insert
    query = "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

    game_id = count
    event_name = game.headers.get("Event", "Unknown")
    site_name = game.headers.get("Site", "Unknown")
    date = game.headers.get("Date", "Unknown")
    white = hashlib.sha256(game.headers.get("White", "Unknown").encode('utf-8')).hexdigest()
    black = hashlib.sha256(game.headers.get("Black", "Unknown").encode('utf-8')).hexdigest()
    result = game.headers.get("Result", "Unknown")
    white_elo =  1600 if not game.headers.get("WhiteElo", "1600").isdigit() else int(game.headers.get("WhiteElo", "1600"))
    black_elo =1600 if not game.headers.get("BlackElo", "1600").isdigit() else int(game.headers.get("BlackElo", "1600"))
    eco = game.headers.get("ECO") or "-"
    termination = game.headers.get("Termination") or "Unknown"
    time_control = game.headers.get("TimeControl") or "Unknown"
    end_time = game.headers.get("EndTime") or "Unknown"
    variant = game.headers.get("Variant") or "Standard"
    ply_count = len(list(game.mainline_moves()))


    white_elo_diff = white_elo - black_elo
    black_elo_diff = black_elo - white_elo

    winner = "Unknown"
    loser = "Unknown"
    winner_elo = None
    winner_elo_diff = None
    loser_elo = None
    loser_elo_diff = None
    
    if result == "1-0":
        winner = white
        loser = black
        winner_elo = white_elo
        winner_elo_diff = white_elo_diff
        loser_elo = black_elo
        loser_elo_deff = black_elo_diff
    elif result == "0-1":
        winner = black
        loser = white
        winner_elo = black_elo
        winner_elo_diff = black_elo_diff
        loser_elo = white_elo
        loser_elo_deff = white_elo_diff 
    elif result == "0.5-0.5":
        winner = None
        loser = None

    game_data = (game_id, count, event_name, site_name, date, '1', white, black, result, 
             white_elo, white_elo_diff , black_elo, black_elo_diff, '', '', winner, winner_elo, loser, loser_elo, winner_elo_diff, 
             eco, termination, time_control, date, end_time, variant, ply_count, 'NotUsed', 'NotUsed')
    queue.add_query(query, [game_data]) #wrapped in array for execute many optimization on possible moves

def parse_eval(eval):
    if eval.type == "cp":
        return (eval.value, None)
    elif eval.type == "mate":
        return (None, eval.value)
    else:
        return (None, None)

def get_possible_move(game_id, board, stockfish_move, fen_before):
    uci = stockfish_move.get("Move") or None
    if uci is None:
        return
    
    move = board.parse_uci(uci)
    san = board.san(move)
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    promotion = None if move.promotion is None else chess.Piece(move.promotion, board.turn).symbol()
    piece = board.piece_at(chess.parse_square(from_square)).symbol()

    eval = stockfish_move.get("Centipawn", "None") 
    mate_count = stockfish_move.get("Mate", "None") 

    board.push(move)
    move_no_pair = board.fullmove_number
    color = "White" if board.turn == chess.WHITE else "Black"
    move_no = board.ply()
    fen_after = board.fen()
    board.pop()
    return {
        'game_id': game_id, 
        'move_no': move_no, 
        'move_no_pair': move_no_pair, 
        'notation': san, 
        'move': uci, 
        'from_square': from_square, 
        'to_square': to_square, 
        'piece': piece, 
        'promotion': promotion,
        'color': color, 
        'fen_before': fen_before,
        'fen_after': fen_after,
        'eval': eval, 
        'mate_count': mate_count, 
        'white_win_perc': None, # not yet supported 
        'black_win_perc': None, # not yet supported  
        'draw_perc': None, # not yet supported    
    }
    
def push_possible_move_evals(queue, game_id, game, board, stockfish, fen):
    query = """
        INSERT INTO possible_move_evals (
            game_id, move_no, move_no_pair, notation, move, 
            from_square, to_square, piece, color, fen_before, fen_after,
            eval, mate_count, white_win_perc, black_win_perc, draw_perc
        )
        VALUES (
            :game_id, :move_no, :move_no_pair, :notation, :move,
            :from_square, :to_square, :piece, :color, :fen_before, :fen_after,
            :eval, :mate_count, :white_win_perc, :black_win_perc, :draw_perc
        );
    """
    moves = stockfish.get_top_moves(218) # maximum number of moves ever possible
    possible_moves_analysis = []
    for stockfish_move in moves:
        possible_moves_analysis.append(get_possible_move(game_id, board, stockfish_move, fen))

    if len(possible_moves_analysis) > 0:
        queue.add_query(query, possible_moves_analysis)
    
        

def push_move(queue, game_id, game, node, board, stockfish, moves_list):
    query = """
        INSERT INTO actual_moves (
            game_id, move_no, move_no_pair, player, notation, move, 
            from_square, to_square, piece, color, fen_before, time_remaining, 
            time_spent, game_to_position, white_win_perc_before, 
            black_win_perc_before, draw_perc_before, white_win_perc_after, 
            black_win_perc_after, draw_perc_after, static_eval_before, 
            static_eval_after, eval_before, mate_count_before, eval_after, 
            mate_count_after
        ) VALUES (:game_id, :move_no, :move_no_pair, :player, :notation, :move, 
            :from_square, :to_square, :piece, :color, :fen_before, :time_remaining, 
            :time_spent, :game_to_position, :white_win_perc_before, 
            :black_win_perc_before, :draw_perc_before, :white_win_perc_after, 
            :black_win_perc_after, :draw_perc_after, :static_eval_before, 
            :static_eval_after, :eval_before, :mate_count_before, :eval_after, 
            :mate_count_after);
        """
    move = node.move
    fen_before = board.fen()
    static_eval_before = stockfish.get_static_eval()
    
    perc_before = stockfish.get_wdl_stats(get_as_tuple=True)
    (white_win_perc_before, draw_perc_before, black_win_perc_before) = (None, None, None) if perc_before is None else perc_before
    (eval_before, mate_count_before) = stockfish.get_evaluation()
    
    push_possible_move_evals(queue, game_id, game, board, stockfish, fen_before)
    player = game.headers.get("White", "Unknown") if board.turn == chess.WHITE else game.headers.get("Black", "Unknown")
    uci = move.uci()
    san = board.san(move)
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    promotion = None if move.promotion is None else chess.Piece(move.promotion, board.turn).symbol()
    piece = board.piece_at(chess.parse_square(from_square)).symbol()
    
    board.push(move)
    moves_list.append(uci)
    
    stockfish.make_moves_from_current_position([uci])
    static_eval_after = stockfish.get_static_eval()
    perc_after = stockfish.get_wdl_stats(get_as_tuple=True)
    (white_win_perc_after, draw_perc_after, black_win_perc_after) = (None, None, None) if perc_after is None else perc_after
    (eval_after, mate_count_after) = stockfish.get_evaluation()

    move_no_pair = board.fullmove_number
    color = "White" if board.turn == chess.WHITE else "Black"
    move_no = board.ply()
    fen_after = board.fen()
    time_remaining = node.clock() or None
    time_spent =  node.emt() or None

    wdl_totals = 1000.0
    actual_move_data = {
        "game_id": game_id,
        "player": hashlib.sha256(player.encode("UTF-8")).hexdigest(),
        'move_no': move_no, 
        'move_no_pair': move_no_pair, 
        'notation': san, 
        'move': uci, 
        'from_square': from_square, 
        'to_square': to_square, 
        'piece': piece, 
        'promotion': promotion,
        'color': color, 
        'fen_before': fen_before,
        'fen_after': fen_after,
        "time_remaining": time_remaining,
        "time_spent": time_spent,
        "game_to_position": ' '.join(moves_list), 
        "white_win_perc_before": white_win_perc_before/wdl_totals,
        "black_win_perc_before": black_win_perc_before/wdl_totals,
        "draw_perc_before": draw_perc_before/wdl_totals,
        "white_win_perc_after": white_win_perc_before/wdl_totals,
        "black_win_perc_after": black_win_perc_before/wdl_totals,
        "draw_perc_after": draw_perc_before/wdl_totals,
        "static_eval_before": None, #static_eval_before, 
        "static_eval_after": None, #static_eval_after, 
        "eval_before":eval_before, 
        "mate_count_before":mate_count_before, 
        "eval_after":eval_after, 
        "mate_count_after":mate_count_after
    }
    
    queue.add_query(query, [actual_move_data]) #wrapped in array for execute many optimization on possible moves
    
    return moves_list

def evaluate_game(queue, stockfish, game, game_id):
    print(f'Evaluating game: {game_id}')
    stockfish.send_ucinewgame_command()
    board = game.board()
    stockfish.set_fen_position(board.fen())
    moves_list = []
    push_headers(queue, game, game_id)
    for node in game.mainline():
        moves_list = push_move(queue, game_id, game, node, board, stockfish, moves_list)
    print(f"Completed processing: {game_id} - Writes not guaranteed")