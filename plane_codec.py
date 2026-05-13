"""
plane_codec.py - CNN Plane Codec Utilities for Chess

47-plane encoding: current position + last 2 moves
Planes 0-11: current position pieces (P,N,B,R,Q,K white then black)
Planes 12-23: t-1 pieces
Planes 24-35: t-2 pieces  
Planes 36-37: last move from/to for t-1
Planes 38-39: last move from/to for t-2
Plane 40: side to move (1.0 for white, 0.0 for black)
Planes 41-44: castling rights (WK, WQ, BK, BQ)
Plane 45: en passant target square
Plane 46: fifty-move counter / 100.0
"""

import numpy as np
import chess
import zlib
from typing import List, Dict, Tuple, Optional, Any


def board_to_planes(board: chess.Board, 
                    history: List[Tuple[chess.Square, chess.Square]] = None) -> np.ndarray:
    """
    Convert a chess.Board to 47x8x8 plane representation.
    
    Args:
        board: Current chess.Board position
        history: List of (from_square, to_square) moves for t-1, t-2, etc.
                [t-1_move, t-2_move, ...]
    
    Returns:
        np.ndarray: Shape (47, 8, 8) float32 planes
    """
    planes = np.zeros((47, 8, 8), dtype=np.float32)
    
    # Piece planes for current position (0-11)
    for piece_type in range(1, 7):  # PAWN to KING
        for color in [chess.WHITE, chess.BLACK]:
            plane_idx = (piece_type - 1) + (6 * color)
            for square in board.pieces(piece_type, color):
                row = 7 - (square // 8)
                col = square % 8
                planes[plane_idx, row, col] = 1.0
    
    # Piece planes for t-1 (12-23)
    if history and len(history) >= 1:
        temp_board = board.copy()
        temp_board.pop()  # Undo last move to get t-1
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                plane_idx = 12 + (piece_type - 1) + (6 * color)
                for square in temp_board.pieces(piece_type, color):
                    row = 7 - (square // 8)
                    col = square % 8
                    planes[plane_idx, row, col] = 1.0
    
    # Piece planes for t-2 (24-35)
    if history and len(history) >= 2:
        temp_board = board.copy()
        temp_board.pop()  # t-1
        temp_board.pop()  # t-2
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                plane_idx = 24 + (piece_type - 1) + (6 * color)
                for square in temp_board.pieces(piece_type, color):
                    row = 7 - (square // 8)
                    col = square % 8
                    planes[plane_idx, row, col] = 1.0
    
    # Last move planes (36-39)
    if history and len(history) >= 1:
        from_sq, to_sq = history[0]  # t-1 move
        if from_sq is not None:
            planes[36, 7 - (from_sq // 8), from_sq % 8] = 1.0
            planes[37, 7 - (to_sq // 8), to_sq % 8] = 1.0
    
    if history and len(history) >= 2:
        from_sq, to_sq = history[1]  # t-2 move
        if from_sq is not None:
            planes[38, 7 - (from_sq // 8), from_sq % 8] = 1.0
            planes[39, 7 - (to_sq // 8), to_sq % 8] = 1.0
    
    # Side to move (40)
    planes[40, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Castling rights (41-44)
    planes[41, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[42, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[43, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[44, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # En passant square (45)
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        planes[45, row, col] = 1.0
    
    # Fifty-move counter (46) - normalized
    planes[46, :, :] = min(board.halfmove_clock, 100) / 100.0
    
    return planes


def pack_planes(planes: np.ndarray) -> bytes:
    """
    Pack float planes into compressed binary blob for database storage.
    
    Uses np.packbits to convert to uint8, then zlib.compress.
    Typical size: 90-130 bytes for 47x8x8 position.
    
    Args:
        planes: np.ndarray of shape (47, 8, 8), float32, values 0.0-1.0
    
    Returns:
        bytes: Compressed binary representation
    """
    # Convert to uint8 (0 or 1, or scaled)
    # For binary planes, multiply by 255; for float planes like fifty-move, keep as is
    planes_uint8 = (planes * 255).astype(np.uint8)
    
    # Pack bits to reduce from 47*8*8=3008 bytes to 376 bytes
    packed = np.packbits(planes_uint8.flatten())
    
    # Compress with zlib
    compressed = zlib.compress(packed.tobytes())
    
    return compressed


def unpack_planes(blob: bytes) -> np.ndarray:
    """
    Unpack compressed binary blob back to 47x8x8 float planes.
    
    Args:
        blob: Compressed bytes from pack_planes()
    
    Returns:
        np.ndarray: Shape (47, 8, 8), float32
    """
    # Decompress
    decompressed = zlib.decompress(blob)
    
    # Unpack bits
    unpacked = np.unpackbits(np.frombuffer(decompressed, dtype=np.uint8))
    
    # Reshape to 47x8x8 and convert back to float32
    planes = unpacked.reshape((47, 8, 8)).astype(np.float32) / 255.0
    
    return planes


def planes_to_torch(planes: np.ndarray) -> 'torch.Tensor':
    """
    Convert numpy planes to PyTorch tensor format (CHW).
    
    Args:
        planes: np.ndarray of shape (47, 8, 8)
    
    Returns:
        torch.Tensor: Shape (47, 8, 8) float32, channel-first
    """
    import torch
    return torch.from_numpy(planes).float()


class PlaneCodec:
    """
    Legacy codec class for backward compatibility.
    Wraps the functional API.
    """
    
    def __init__(self, board_size: int = 8, num_piece_types: int = 6):
        self.board_size = board_size
        self.num_piece_types = num_piece_types
        self.total_planes = 12 * 3 + 4 + 1 + 4 + 1 + 1  # 47
    
    def encode_position(self, position_data: Dict) -> np.ndarray:
        """Legacy method - use board_to_planes instead"""
        import chess
        board = chess.Board(position_data.get('fen', chess.STARTING_FEN))
        history = position_data.get('history', [])
        return board_to_planes(board, history)
    
    def decode_planes(self, planes: np.ndarray) -> Dict:
        """Not implemented - planes are one-way for training"""
        raise NotImplementedError("Decoding planes back to board is not supported")
    
    def get_plane_index(self, piece_type: int, color: int) -> int:
        """Get plane index for piece type and color"""
        return (piece_type - 1) + (6 * color)
    
    def unpack_bitboard(self, bitboard: int) -> np.ndarray:
        """Unpack 64-bit integer to 8x8 board"""
        board = np.zeros((8, 8), dtype=np.uint8)
        for i in range(64):
            if bitboard & (1 << i):
                board[7 - (i // 8), i % 8] = 1
        return board
    
    def pack_bitboard(self, board: np.ndarray) -> int:
        """Pack 8x8 board to 64-bit integer"""
        bitboard = 0
        for i in range(64):
            if board[7 - (i // 8), i % 8]:
                bitboard |= (1 << i)
        return bitboard


# Backwards compatibility functions
def decode_plane(plane_data: bytes, plane_index: int) -> np.ndarray:
    """Decode single plane - deprecated, use unpack_planes for full position"""
    planes = unpack_planes(plane_data)
    return planes[plane_index]


def encode_plane(plane: np.ndarray, plane_index: int) -> bytes:
    """Encode single plane - deprecated, use pack_planes for full position"""
    planes = np.zeros((47, 8, 8), dtype=np.float32)
    planes[plane_index] = plane
    return pack_planes(planes)


def get_piece_planes(codec: PlaneCodec, position: Dict) -> List[np.ndarray]:
    """Get all piece planes from position"""
    planes = codec.encode_position(position)
    return [planes[i] for i in range(36)]  # First 36 are piece planes


if __name__ == "__main__":
    # Test
    import chess
    board = chess.Board()
    history = []
    planes = board_to_planes(board, history)
    print(f"Planes shape: {planes.shape}")
    print(f"Non-zero planes: {np.count_nonzero(planes.sum(axis=(1,2)))}")
    
    packed = pack_planes(planes)
    print(f"Packed size: {len(packed)} bytes")
    
    unpacked = unpack_planes(packed)
    print(f"Round-trip matches: {np.allclose(planes, unpacked, atol=1/255)}")
