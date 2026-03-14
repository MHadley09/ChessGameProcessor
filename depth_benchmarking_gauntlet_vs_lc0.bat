@echo off
SET outfile="out_benchmark.txt"
echo Cutechess started at %date% %time%. Output is redirected to %outfile%
echo Cutechess started at %date% %time% > %outfile%

cutechess-cli.exe -tournament gauntlet -rounds 40 -games 2 -repeat -concurrency 8 -pgnout out_benchmark.pgn -recover ^
-resign movecount=20 score=1000 -draw movenumber=50 movecount=40 score=20 ^
-openings file="books/book-ply8-unifen-Q-0.25-0.40.pgn" order=random plies=8 policy=round ^
-engine name="Lc0" cmd="lc0" ^
-engine name="Stockfish Baseline" cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 28" depth=28 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 24" depth=24 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 20" depth=20 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 16" depth=16 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 14" depth=14 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 12" depth=12 cmd="stockfish-windows-x86-64" ^
-engine name="Stockfish Depth 8" depth=8 cmd="stockfish-windows-x86-64" ^
-each proto=uci tc=120+2 >> %outfile%
echo Tournament ended at %date% %time%.
echo Tournament ended at %date% %time%. >> %outfile%
pause