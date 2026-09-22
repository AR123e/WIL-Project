import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("src", exist_ok=True)
os.makedirs("eval", exist_ok=True)

FIDE_MD = ''' # FIDE Laws of Chess - Basic Rules of Play (for beginners)

Source: FIDE Handbook, "FIDE Laws of Chess taking effect from 1 January 2023"
https://handbook.fide.com/chapter/e012023
(Basic Rules of Play only - Articles 1-5. Competetive rules such clocks and arbiter procedure are excluded as out of scope for a beginner-facing KB.)

## Article 1: The Nature and Objectives of the Game of Chess

1.1     The game of chess is played between two opponents who move their pieces on a square board called a ‘chessboard’.

1.2     The player with the light-coloured pieces (White) makes the first move, then the players move alternately, with the player with the dark-coloured pieces (Black) making the next move.

1.3     A player is said to ‘have the move’ when his/her opponent’s move has been ‘made’.

1.4     The objective of each player is to place the opponent’s king ‘under attack’ in such a way that the opponent has no legal move.

1.4.1    The player who achieves this goal is said to have ‘checkmated’ the opponent’s king and to have won the game. Leaving one’s own king under attack, exposing one’s own king to attack and also ’capturing’ the opponent’s king is not allowed.

1.4.2    The opponent whose king has been checkmated has lost the game.

1.5     If the position is such that neither player can possibly checkmate the opponent’s king, the game is drawn (see Article 5.2.2).

 

## Article 2: The Initial Position of the Pieces on the Chessboard

2.1     The chessboard is composed of an 8 x 8 grid of 64 equal squares alternately light (the ‘white’ squares) and dark (the ‘black’ squares).

The chessboard is placed between the players in such a way that the near corner square to the right of the player is white.

2.2     At the beginning of the game White has 16 light-coloured pieces (the ‘white’ pieces); Black has 16 dark-coloured pieces (the ‘black’ pieces).
These pieces are as follows: a king (K), a queen (Q), two rooks (R), two bishops (B), and eight pawns, for each side	 	 	 

 

2.3     The initial position of the pieces on the chessboard is set up with each side's pawns on the second rank (closest to that player), rooks in the corners, then knight, then bishops, with the queen on her own colour and the king on the remaining central square

 

2.4     The eight vertical columns of squares are called ‘files’. The eight horizontal rows of squares are called ‘ranks’. A straight line of squares of the same colour, running from one edge of the board to an adjacent edge, is called a ‘diagonal’.

 

## Article 3: The Moves of the Pieces

3.1     It is not permitted to move a piece to a square occupied by a piece of the same colour.

3.1.1    If a piece moves to a square occupied by an opponent’s piece the latter is captured and removed from the chessboard as part of the same move.

3.1.2    A piece is said to attack an opponent’s piece if the piece could make a capture on that square according to Articles 3.2 to 3.8.

3.1.3    A piece is considered to attack a square even if this piece is constrained from moving to that square because it would then leave or place the king of its own colour under attack.

3.2     The bishop may move to any square along a diagonal on which it stands.


3.3     The rook may move to any square along the file or the rank on which it stands.


3.4     The queen may move to any square along the file, the rank or a diagonal on which it stands.


3.5     When making these moves, the bishop, rook or queen may not move over any intervening pieces.

3.6     The knight may move to one of the squares nearest to that on which it stands but not on the same rank, file or diagonal (the familiar "L-shape")


3.7     The pawn:
3.7.1    The pawn may move forward to the square immediately in front of it on the same file, provided that this square is unoccupied, or

3.7.2    on its first move the pawn may move as in 3.7.1 or alternatively it may advance two squares along the same file, provided that both squares are unoccupied, or

3.7.3    the pawn may move to a square occupied by an opponent’s piece diagonally in front of it on an adjacent file, capturing that piece.


3.7.3.1-2    A pawn occupying a square on the same rank as and on an adjacent file to an opponent’s pawn which has just advanced two squares in one move from its original square may capture this opponent’s pawn as though the latter had been moved only one square.
This capture is only legal on the move following this advance and is called an ‘en passant’ capture.


3.7.3.3-7.3.5    When a player, having the move, plays a pawn to the rank furthest from its starting position, he/she must exchange that pawn as part of the same move for a new queen, rook, bishop or knight of the same colour on the intended square of arrival. This is called the square of ‘promotion’.
The player's choice is not restricted to pieces that have been captured previously.
This exchange of a pawn for another piece is called promotion, and the effect of the new piece is immediate.

3.8     There are two different ways of moving the king:

3.8.1    by moving to an adjoining square; or
3.8.2    by ‘castling’. This is a move of the king and either rook of the same colour along the player’s first rank, counting as a single move of the king and executed as follows: the king is transferred from its original square two squares towards the rook on its original square, then that rook is transferred to the square the king has just crossed.

3.8.2.1    The right to castle has been lost:

1) If the king has already moved, or

2) With a rook that has already moved.

3.8.2.2    Castling is prevented temporarily:

3) If the square on which the king stands, or the square which it must cross, or the square which it is to occupy, is attacked by one or more of the opponent's pieces, or

4) If there is any piece between the king and the rook with which castling is to be effected.

3.9     The king in check:
3.9.1    The king is said to be 'in check' if it is attacked by one or more of the opponent's pieces, even if such pieces are constrained from moving to the square occupied by the king because they would then leave or place their own king in check.
3.9.2    No piece can be moved that will either expose the king of the same colour to check or leave that king in check.

3.10   Legal and illegal moves; illegal positions:

3.10.1     A move is legal when all the relevant requirements of Articles 3.1 – 3.9 have been fulfilled.

3.10.2    A move is illegal when it fails to meet the relevant requirements of Articles 3.1 – 3.9.

3.10.3    A position is illegal when it cannot have been reached by any series of legal moves.

 

## Article 4: The Act of Moving the Pieces

4.1     Each move must be played with one hand only.


4.3     if the player having the move touches on the chessboard, with the intention of moving or capturing: one or more of his/her own pieces, he/she must move the first piece touched that can be moved.
 

## Article 5: The Completion of the Game

5.1.1    The game is won by the player who has checkmated his/her opponent’s king. 

5.1.2    The game is lost by the player who declares he/she resigns (this immediately ends the game), unless the position is such that the opponent cannot checkmate the player’s king by any possible series of legal moves. In this case the result of the game is a draw.

5.2.1    The game is drawn when the player to move has no legal move and his/her king is not in check. The game is said to end in ‘stalemate’.

5.2.2    The game is drawn when a position has arisen in which neither player can checkmate the opponent’s king with any series of legal moves. The game is said to end in a ‘dead position’. 

5.2.3    The game is drawn upon agreement between the two players during the game, provided both players have made at least one move.

## Glossary (selected beginner-relevant terms)

check: Where a king is attacked by one or more of the opponent’s pieces.

checkmate: Where the king is attacked and cannot parry the threat.

stalemate: where the player to has no legal move and is not in
castling: a move of the king and a rook together, used to get the king
en passant: a special pawn capture available only on the move immediately after an opponent's pawn advances two squares.
promotion: replacing a pawn that reaches the far rank with a queen, rook, bishop, or knight.
draw: a game that ends with neither side winning.
minor piece: a bishop or knight.
file/ rank / diagonal: the vertical/ horizontal/ diagonal lines of squares on the board 
'''
WIKIBOOKS_MD = ''' ## Chess Opening Principles and a Few Common Openings (for beginners)

Source: Wikibooks,"Chess/Basic Openings" (CC BY-SA),
https://en.wikibooks.org/wiki/Chess/Basic_Openings

## Aims of an opening

Before looking at some of the more common openings played today it is important to consider what you are trying to achieve in the opening of a chess game.

In some cases, one player will sacrifice a pawn, or in some cases even more, to accomplish the goals listed below. Such an opening is called a gambit.

Development
You should attempt to move your pieces away from their starting positions to squares in which they can participate more fully in the game. Obviously a Knight at c3 is more effective both defensively and aggressively than a knight on b1. Avoid moving pieces more than once in the opening as this allows your opponent time to develop another piece while you are wasting time. Also avoid moving your queen in the opening, it can too easily be chased around the board by other pieces which aids your opponent's development while wasting moves for you.

Control the Centre

The squares in the centre of the board are critical for two reasons. The first is that pieces in the centre are able to move to more squares than pieces on the edges. (Note a knight on a3 can move to only 4 squares, whereas a knight on c3 can move to 8.) Secondly, if you control the centre it is easier to move pieces from Kingside to Queenside quickly.

One of the great truths in chess is that attacks along one wing are destined to fail if the centre is not sufficiently controlled. This is because the defender with a strong centre will generally be able to muster enough defence on that side and at the same time mount a counterattack in the centre and/or on the other wing. So if you plan on mounting an attack make sure you control the centre.

Traditionally it was thought that the ideal situation is one where you have a pawn majority in the centre, especially with pawns on d4 and e4 (for white). However it has been shown that an equally valid strategy is to control the centre with pieces and make minimal pawn moves. The thinking is that central pawn moves often lead to permanent positions and can block attacks. "Fianchettoing" a bishop by moving it to b2 or g2 after b3 or g3 allows the bishop to keep a watchful eye on the centre without fixing a pawn in the centre.


Protect the King

It is not always necessary although highly advisable to protect the king through castling. That being said, there are other ways to go about protecting the king than castling although they are less efficient and will often result in pins.

## A Few Common Openings (illustrative, not exhaustive)

**Italian Game**: 1.e4 e5 2.Nf3 Nc6 3.Bc4. A classical opening that develops a piece towards the centre and eyes the f7 square early. two common replies are 3...Bc5 (the "Giuoco Piano", Italian for "Quiet Game") and 3...Nf6.

**Ruy Lopez (Spanish Opening)**: 1.e4 e5 2.Nf3 Nc6 3.Bb5. one of the oldest and most respected openings, developing the bishop to pressure the knight defending e5.

**Sicilian Defence**: 1.e4 c5. The most common and highest-scoring reply to 1.e4 at all levels. Instead of mirroring White's central pawn, Black creates an assymetrical position and often gets active piece play in return.

**Queen's Gambit**: 1.d4 d5 2.c4. White offers a wing pawn to gain faster development and central influence. Despite the name, white does not usually end up a pawn down for long if Black accepts it (2...dxc4).

## Note on scope

Specific named openings and move sequences are more advanced than absolute first principles, but they are included here because beginners are commonly curious about "why is [opening] considered good", and because they let us test whether the RAG system retrieves the right level of detail depending on how a question is phrased.
'''

PREPARE_KB_PY = ''' """
"""
import json
import re
from pathlib import Path
RAW_DIR = Path(__file__).parent.parent/"data"/"raw"
OUT_PATH = Path(__file__).parent.parent/"data"/"processed"/"chunks.json" 
DOCS = [
      {
	"doc_id": "fide_basic_rules",
	"path": RAW_DIR/ "fide_basic_rules.md",
	"title": "FIDE Laws of Chess - Basic Rules of play",
      },
      {
	"doc_id": "wikibooks_opening_principles",
	"path": RAW_DIR/ "wikibooks_opening_principles.md",
	"title": "Chess Opening Principles (Wikibooks)",
      },
]
def split_sections(text: str):
	parts = re.split(r"\n(?=##)", text)
	sections=[]
	for part in parts:
		part=part.strip()
		if not part or part.startswith("# "):
			if part.startswith("## "):
				pass
			else:
				continue
		lines = part.split("\n",1)
		header = lines[0].lstrip("#").strip()
		body = lines[1].strip() if len(lines)>1 else ""
		if body:
			sections.append((header,body))
	return sections

def split_paragraphs(body: str):
	paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
	return paras

def main():
	chunks=[]
	chunk_counter = 0
	for doc in DOCS:
		text = doc["path"].read_text()
		sections = split_sections(text)
		for section_title, section_body in sections:
			paragraphs = split_paragraphs(section_body)
			for para in paragraphs:
				if len(para)<40:
					continue
				chunk_counter+=1
				chunks.append(
					{
						"chunk_id": f"{doc['doc_id']}_{chunk_counter:03d}",
						"doc_id": doc["doc_id"],
						"doc_title": doc["title"],
						"section": section_title,
						"text": para,	
					}
				) 
	OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUT_PATH.write_text(json.dumps(chunks, indent=2))
	print(f"Wrote {len(chunks)} chunks to {OUT_PATH}")
if __name__ == "__main__":
	main() 
'''

RETRIEVE_PY = '''"""
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"

def _tokensize(text: str):
	return re.findall(r"[a-z0-9]+", text.lower())
class ChessRetriever:
	def __init__(self, chunks_path: Path = CHUNKS_PATH):
		self.chunks = json.loads(Path(chunks_path).read_text())
		corpus = [_tokensize(c["text"]) for c in self.chunks]
		self.bm25 = BM25Okapi(corpus)
	def retrieve(self, query: str, k: int = 3):
		scores = self.bm25.get_scores(_tokensize(query))
		ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
		results=[]
		for i in ranked:
			results.append({**self.chunks[i], "score": float(scores[i])})
		return results
if __name__=="__main__":
	retriever = ChessRetriever()
	demo_q = "How does a pawn capture another piece?"
	for r in retriever.retrieve(demo_q, k=3):
		print(f"[{r['score']:.2f}] {r['chunk_id']} ({r['section']})")
		print(" ", r["text"][:100].replace("\n"," "))
'''
files={
	"data/raw/fide_basic_rules.md": FIDE_MD,
	"data/raw/wikibooks_opening_principles.md": WIKIBOOKS_MD,
	"src/prepare_kb.py": PREPARE_KB_PY,
	"src/retrieve.py": RETRIEVE_PY,
}

for path, content in files.items():
	with open(path, "w", encoding="utf-8") as f:
		f.write(content)
	print("Wrote", path)
print("\nDone. Now run:")
print(" pythom src\\prepare_kb.py")
print(" python src\\retrive.py")