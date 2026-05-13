#!/usr/bin/env python3
"""
lc0_evaluator_fixed.py
Windows-compatible LC0 evaluator using python-chess engine wrapper.
This works with the LC0 Windows binary (lc0.exe) instead of a pip package.
"""
import chess
import chess.engine
import chess.pgn
from pathlib import Path
import hashlib
import time
class LC0Evaluator:
    """
    LC0 evaluator using python-chess UCI interface.
    Works with lc0.exe on Windows.
    """
    
    def __init__(self, engine_path="C:\\lc0\\lc0.exe", weights_path="weights\\703810.pb.gz", 
                 backend="cudnn-fp16", threads=4):
        """
        Initialize LC0 engine.
        
        Args:
            engine_path: Path to lc0.exe
            weights_path: Path to network weights (.pb.gz)
            backend: Backend to use ('cudnn-fp16', 'cudnn', 'blas')
            threads: CPU threads for search
        """
        self.engine_path = engine_path
        self.weights_path = Path(weights_path)
        self.backend = backend
        
        # Verify files exist
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"LC0 engine not found: {engine_path}\n"
                                   f"Download from: https://github.com/LeelaChessZero/lc0/releases")
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}\n"
                                   f"Download from: https://lczero.org/play/networks")
        
        # Start engine
        self.engine = chess.engine.SimpleEngine.popen_uci(
            engine_path,
            setpgrp=False  # Windows compatibility
        )
        
        # Configure engine
        self.engine.configure({
            "WeightsFile": str(self.weights_path.absolute()),
            "Backend": backend,
            "Threads": threads,
            "MinibatchSize": 256,
            "MaxPrefetch": 0,
            "LogFile": ""  # Disable logging
        })
        
        self._network_hash = self._get_network_hash()
        
        # Statistics
        self.stats = {
            'positions_evaluated': 0,
            'total_time': 0.0,
        }
    
    def _get_network_hash(self):
        """Get hash of network weights file"""
        return hashlib.md5(self.weights_path.read_bytes()).hexdigest()[:12]
    
    @property
    def network_info(self):
        return {
            'engine': 'lc0',
            'backend': self.backend,
            'weights_file': self.weights_path.name,
            'weights_hash': self._network_hash,
        }
    
    def evaluate_position(self, board):
        """
        Evaluate position.
        Uses 0 nodes (pure NN forward pass) for speed.
        """
        start = time.time()
        
        # Analyse with 1 node minimum to avoid LC0 returning invalid 'a1a1' moves
        try:
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(nodes=1),
                info=chess.engine.INFO_ALL
            )
        except chess.engine.EngineError:
            # Terminal position or engine glitch — return neutral eval
            self.stats['positions_evaluated'] += 1
            self.stats['total_time'] += time.time() - start
            return {
                'ev': 0,
                'wdl': [333, 334, 333],
                'best_move': None,
                'engine': 'lc0',
                'network': self._network_hash
            }
        
        # Extract score
        score = info.get('score')
        if score:
            # Convert to centipawns from pov of white
            cp = score.white().score(mate_score=10000)
            if cp is None:  # Mate
                cp = 10000 if score.white().mate() > 0 else -10000
        else:
            cp = 0
        
        # Extract WDL if available (some LC0 versions provide this)
        wdl = info.get('wdl', [333, 334, 333])  # Default neutral
        
        # Get principal variation (best move)
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
        """
        Get top N moves with evaluations.
        Uses MultiPV analysis.
        """
        # Set MultiPV option
        self.engine.configure({"MultiPV": num_moves})
        
        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=0),
            multipv=num_moves,
            info=chess.engine.INFO_ALL
        )
        
        # Reset MultiPV
        self.engine.configure({"MultiPV": 1})
        
        moves = []
        for line in info:
            if 'pv' in line and line['pv']:
                move = line['pv'][0]
                score = line.get('score')
                cp = score.white().score(mate_score=10000) if score else 0
                if cp is None:
                    cp = 10000 if score.white().mate() > 0 else -10000
                
                moves.append({
                    'Move': move.uci(),
                    'Centipawn': cp,
                    'Mate': score.white().mate() if score else None,
                })
        
        return moves
    
    def close(self):
        """Close engine"""
        if hasattr(self, 'engine'):
            self.engine.quit()
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
# Test script
if __name__ == '__main__':
    import sys
    
    weights = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\micha\Personal\School\DEng\dissertation\mutation\ChessGameProcessor\weights\791556.pb.gz"
    engine = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\micha\Personal\Coding\chess-clone\lc0\lc0.exe"
    
    print(f"Testing LC0: {engine}")
    print(f"Weights: {weights}")
    
    with LC0Evaluator(engine_path=engine, weights_path=weights) as evaluator:
        board = chess.Board()
        
        # Test single position
        start = time.time()
        result = evaluator.evaluate_position(board)
        elapsed = time.time() - start
        
        print(f"\nSingle position eval: {elapsed*1000:.1f}ms")
        print(f"Score: {result['ev']} cp")
        print(f"WDL: {result['wdl']}")
        print(f"Best move: {result['best_move']}")
        
        # Test batch speed
        print("\nTesting batch speed...")
        start = time.time()
        for i in range(100):
            board = chess.Board()
            # Make some random moves to get different positions
            if board.legal_moves:
                board.push(list(board.legal_moves)[i % board.legal_moves.count()])
            evaluator.evaluate_position(board)
        elapsed = time.time() - start
        
        print(f"100 positions: {elapsed:.2f}s")
        print(f"Speed: {100/elapsed:.1f} pos/sec")
        print(f"\nExpected on RTX 4090: 800-1200 pos/sec")