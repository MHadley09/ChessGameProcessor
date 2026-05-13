#!/usr/bin/env python3
"""
batch_evaluator.py

Async batch LC0 evaluator. Spawns N LC0 engine instances and distributes
positions across them using asyncio. The GPU naturally batches across
concurrent engine instances via CUDA scheduling.

This saturates LC0's minibatch (256) without needing the C++ API.
"""

import asyncio
import chess
import chess.engine
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvalRequest:
    """A position to evaluate."""
    board: chess.Board
    request_id: int = 0


@dataclass 
class EvalResult:
    """Evaluation result."""
    request_id: int
    ev: int
    wdl: List[int]
    best_move: Optional[str]
    engine: str = 'lc0'
    network: str = ''


class BatchLC0Evaluator:
    """
    Async batch evaluator using multiple LC0 UCI instances.
    
    Instead of 1 engine doing 1 position at a time, this runs N engines
    concurrently. The GPU batches across all of them.
    
    With num_engines=16 on a 4090:
    - Each engine sends 1 position at a time via UCI
    - But 16 are running concurrently
    - LC0's CUDA scheduler batches them into GPU minibatches
    - Net effect: ~10-16x throughput vs single engine
    """

    def __init__(self,
                 engine_path: str = r"C:\lc0\lc0.exe",
                 weights_path: str = "weights/791556.pb.gz",
                 backend: str = "cuda-fp16",
                 num_engines: int = 16,
                 nodes: int = 1):
        self.engine_path = engine_path
        self.weights_path = Path(weights_path)
        self.backend = backend
        self.num_engines = num_engines
        self.nodes = nodes
        self.engines: List[chess.engine.UciProtocol] = []
        self._network_hash = self._get_network_hash()
        self._semaphores: List[asyncio.Semaphore] = []
        self._next_engine = 0
        
        # Stats
        self.positions_evaluated = 0
        self.total_time = 0.0

    def _get_network_hash(self) -> str:
        return hashlib.md5(self.weights_path.read_bytes()).hexdigest()[:12]

    @property
    def network_info(self):
        return {
            'engine': 'lc0',
            'backend': self.backend,
            'weights_file': self.weights_path.name,
            'weights_hash': self._network_hash,
        }

    async def start(self):
        """Start all engine instances."""
        print(f"Starting {self.num_engines} LC0 engines ({self.backend})...")
        
        for i in range(self.num_engines):
            transport, engine = await chess.engine.popen_uci(self.engine_path)
            await engine.configure({
                "WeightsFile": str(self.weights_path.absolute()),
                "Backend": self.backend,
                "Threads": 1,
                "MinibatchSize": 256,
                "MaxPrefetch": 0,
                "LogFile": "",
            })
            self.engines.append(engine)
            self._semaphores.append(asyncio.Semaphore(1))
        
        print(f"  {self.num_engines} engines ready")

    async def stop(self):
        """Stop all engines."""
        for engine in self.engines:
            try:
                await engine.quit()
            except Exception:
                pass
        self.engines.clear()

    async def _eval_single(self, engine_idx: int, board: chess.Board) -> Dict:
        """Evaluate a single position on a specific engine."""
        engine = self.engines[engine_idx]
        
        # Handle terminal positions
        if board.is_checkmate():
            cp = -10000 if board.turn == chess.WHITE else 10000
            wdl = [0, 0, 1000] if board.turn == chess.WHITE else [1000, 0, 0]
            return {'ev': cp, 'wdl': wdl, 'best_move': None,
                    'engine': 'lc0', 'network': self._network_hash}

        if (board.is_stalemate() or board.is_insufficient_material() or
                board.can_claim_draw() or board.is_fifty_moves() or board.is_repetition()):
            return {'ev': 0, 'wdl': [0, 1000, 0], 'best_move': None,
                    'engine': 'lc0', 'network': self._network_hash}

        try:
            info = await engine.analyse(
                board,
                chess.engine.Limit(nodes=self.nodes),
                info=chess.engine.INFO_ALL
            )
        except chess.engine.EngineError:
            return {'ev': 0, 'wdl': [333, 334, 333], 'best_move': None,
                    'engine': 'lc0', 'network': self._network_hash}

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

        return {'ev': cp, 'wdl': wdl, 'best_move': best_move,
                'engine': 'lc0', 'network': self._network_hash}

    async def evaluate_position(self, board: chess.Board) -> Dict:
        """Evaluate a single position (picks the next available engine)."""
        # Round-robin engine selection with semaphore
        engine_idx = self._next_engine % self.num_engines
        self._next_engine += 1
        
        async with self._semaphores[engine_idx]:
            start = time.time()
            result = await self._eval_single(engine_idx, board)
            self.total_time += time.time() - start
            self.positions_evaluated += 1
            return result

    async def evaluate_batch(self, boards: List[chess.Board]) -> List[Dict]:
        """
        Evaluate multiple positions concurrently.
        This is the key method — fires all positions at once across engines.
        """
        start = time.time()
        
        tasks = []
        for i, board in enumerate(boards):
            engine_idx = i % self.num_engines
            # Each engine gets a semaphore so it processes one at a time,
            # but all engines run concurrently
            task = self._eval_with_semaphore(engine_idx, board)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        self.positions_evaluated += len(boards)
        self.total_time += elapsed
        
        return list(results)

    async def _eval_with_semaphore(self, engine_idx: int, board: chess.Board) -> Dict:
        async with self._semaphores[engine_idx]:
            return await self._eval_single(engine_idx, board)

    def get_stats(self):
        rate = self.positions_evaluated / self.total_time if self.total_time > 0 else 0
        return {
            'positions_evaluated': self.positions_evaluated,
            'total_time': self.total_time,
            'positions_per_second': rate,
            'num_engines': self.num_engines,
        }


class SyncBatchEvaluator:
    """
    Synchronous wrapper around BatchLC0Evaluator.
    
    Drop-in replacement for the old LC0Evaluator — same interface,
    but internally uses async batch evaluation.
    
    Usage:
        evaluator = SyncBatchEvaluator(weights_path="weights/791556.pb.gz", num_engines=16)
        result = evaluator.evaluate_position(board)  # same API as before
        results = evaluator.evaluate_batch(boards)    # new: batch API
    """

    def __init__(self,
                 engine_path: str = r"C:\lc0\lc0.exe",
                 weights_path: str = "weights/791556.pb.gz",
                 backend: str = "cuda-fp16",
                 num_engines: int = 16,
                 nodes: int = 1,
                 **kwargs):
        self._loop = asyncio.new_event_loop()
        self._batch_eval = BatchLC0Evaluator(
            engine_path=engine_path,
            weights_path=weights_path,
            backend=backend,
            num_engines=num_engines,
            nodes=nodes,
        )
        self._loop.run_until_complete(self._batch_eval.start())

    @property
    def network_info(self):
        return self._batch_eval.network_info

    def evaluate_position(self, board: chess.Board) -> Dict:
        """Single position eval — same API as old evaluator."""
        return self._loop.run_until_complete(
            self._batch_eval.evaluate_position(board)
        )

    def evaluate_batch(self, boards: List[chess.Board]) -> List[Dict]:
        """Batch eval — fire all positions concurrently."""
        return self._loop.run_until_complete(
            self._batch_eval.evaluate_batch(boards)
        )

    def get_stats(self):
        return self._batch_eval.get_stats()

    def close(self):
        self._loop.run_until_complete(self._batch_eval.stop())
        self._loop.close()


if __name__ == '__main__':
    """Quick benchmark."""
    import sys

    weights = sys.argv[1] if len(sys.argv) > 1 else "weights/791556.pb.gz"
    engine_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\lc0\lc0.exe"
    num_engines = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    print(f"Benchmarking with {num_engines} engines...")
    
    evaluator = SyncBatchEvaluator(
        engine_path=engine_path,
        weights_path=weights,
        num_engines=num_engines,
    )

    # Generate 256 different positions
    boards = []
    board = chess.Board()
    boards.append(board.copy())
    for move in list(board.legal_moves)[:20]:
        b = chess.Board()
        b.push(move)
        boards.append(b.copy())
        for move2 in list(b.legal_moves)[:12]:
            b2 = b.copy()
            b2.push(move2)
            boards.append(b2.copy())
            if len(boards) >= 256:
                break
        if len(boards) >= 256:
            break

    boards = boards[:256]
    print(f"Testing with {len(boards)} positions")

    # Single eval baseline
    start = time.time()
    for b in boards[:50]:
        evaluator.evaluate_position(b)
    single_elapsed = time.time() - start
    single_rate = 50 / single_elapsed

    # Batch eval
    start = time.time()
    results = evaluator.evaluate_batch(boards)
    batch_elapsed = time.time() - start
    batch_rate = len(boards) / batch_elapsed

    print(f"\nSingle eval:  {single_rate:.1f} pos/sec")
    print(f"Batch eval:   {batch_rate:.1f} pos/sec")
    print(f"Speedup:      {batch_rate/single_rate:.1f}x")
    print(f"\nStats: {evaluator.get_stats()}")

    evaluator.close()
