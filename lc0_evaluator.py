#!/usr/bin/env python3
"""
lc0_evaluator.py

Compatible with both standalone use and lc0_processor_with_parquet.py
Accepts parameters that the processor passes.
"""

import chess
import chess.engine
import chess.pgn
from pathlib import Path
import hashlib
import time
import sys

class LC0Evaluator:
    """LC0 evaluator configured for Michael's Windows setup"""
    
    def __init__(self, 
                 engine_path=None,
                 weights_path=None,
                 backend='cudnn-fp16',
                 threads=4,
                 minibatch_size=256,
                 max_batch_size=256,  # Accept this for compatibility
                 **kwargs):  # Accept any other params
        
        # Use provided paths or defaults from Michael's setup
        self.engine_path = engine_path or r"C:\Users\micha\Personal\Coding\chess-clone\lc0\lc0.exe"
        self.weights_path = Path(weights_path or r"C:\Users\micha\Personal\School\DEng\dissertation\mutation\ChessGameProcessor\weights\791556.pb.gz")
        self.backend = backend
        
        # Use max_batch_size if provided, otherwise minibatch_size
        self.minibatch_size = max_batch_size or minibatch_size
        
        # Verify files exist
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"LC0 engine not found: {self.engine_path}")
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        
        # Start engine
        print(f"Starting LC0: {self.engine_path}")
        print(f"Weights: {self.weights_path.name}")
        
        self.engine = chess.engine.SimpleEngine.popen_uci(
            self.engine_path,
            setpgrp=False
        )
        
        # Configure engine
        self.engine.configure({
            "WeightsFile": str(self.weights_path.absolute()),
            "Backend": self.backend,
            "Threads": threads,
            "MinibatchSize": self.minibatch_size,
            "MaxPrefetch": 32,
            "LogFile": ""
        })
        
        self._network_hash = hashlib.md5(self.weights_path.read_bytes()).hexdigest()[:12]
        
        self.stats = {'positions_evaluated': 0, 'total_time': 0.0}
        print(f"LC0 initialized: {self._network_hash}")
        print(f"Backend: {self.backend}")
        print(f"Minibatch size: {self.minibatch_size}")
    
    @property
    def network_info(self):
        return {
            'engine': 'lc0',
            'backend': self.backend,
            'weights_file': self.weights_path.name,
            'weights_hash': self._network_hash,
        }
    
    def evaluate_position(self, board):
        """Evaluate position with 0 nodes (pure NN)"""
        start = time.time()
        
        info = self.engine.analyse(
            board, 
            chess.engine.Limit(nodes=0),
            info=chess.engine.INFO_ALL
        )
        
        score = info.get('score')
        if score:
            cp = score.white().score(mate_score=10000)
            if cp is None:
                cp = 10000 if score.white().mate() > 0 else -10000
        else:
            cp = 0
        
        wdl = info.get('wdl', [333, 334, 333])
        pv = info.get('pv', [])
        best_move = pv[0].uci() if pv else None
        
        self.stats['positions_evaluated'] += 1
        self.stats['total_time'] += time.time() - start
        
        return {
            'ev': cp,
            'wdl': wdl,
            'best_move': best_move,
            'engine': 'lc0',
            'network': self._network_hash
        }
    
    def get_top_moves(self, board, num_moves=10):
        """Get top N moves"""
        self.engine.configure({"MultiPV": num_moves})
        info = self.engine.analyse(board, chess.engine.Limit(nodes=0), multipv=num_moves, info=chess.engine.INFO_ALL)
        self.engine.configure({"MultiPV": 1})
        
        moves = []
        for line in info:
            if 'pv' in line and line['pv']:
                move = line['pv'][0]
                score = line.get('score')
                cp = score.white().score(mate_score=10000) if score else 0
                if cp is None:
                    cp = 10000 if score.white().mate() > 0 else -10000
                moves.append({'Move': move.uci(), 'Centipawn': cp, 'Mate': score.white().mate() if score else None})
        return moves
    
    def close(self):
        if hasattr(self, 'engine'):
            self.engine.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# For backward compatibility with scripts that do: from lc0_evaluator_micha import LC0Evaluator
# they can just do: from lc0_evaluator import LC0Evaluator

if __name__ == '__main__':
    print("="*70)
    print("LC0 Evaluator Test")
    print("="*70)
    
    with LC0Evaluator() as evaluator:
        board = chess.Board()
        result = evaluator.evaluate_position(board)
        print(f"Score: {result['ev']} cp")
        print(f"WDL: {result['wdl']}")
        print(f"Best: {result['best_move']}")