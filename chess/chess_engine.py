import copy


class Board():

    def __init__(self):
        self.white_team = Team(1)
        self.black_team = Team(2)


    def play_turn(self, team):
        moves =


    def get_all_pieces(self):
        return self.white_team.get_all_pieces() + self.black_team.get_all_pieces()


    def get_all_living_pieces(self):
        return self.white_team.get_living_pieces() + self.black_team.get_living_pieces()


    def get_all_living_pieces_of_team(self, num):
        if num == 1:
            return self.white_team.get_living_pieces()
        else:
            return self.black_team.get_living_pieces()


    def execute_move(self, m):
        new_board = copy.deepcopy(self)

        pieces = new_board.get_all_living_pieces()
        piece = [i for i in pieces if i.x == m['piece'].x and i.y == m['piece'].y][0]
        target = m['piece_taken']
        if target:
            target = [i for i in pieces if i.x == target.x and i.y == target.y][0]

        p = m['p']
        piece.x = p[0]
        piece.y = p[1]
        if target:
            target.get_taken()
        return new_board




class Team():
    def __init__(self, team):
        self.team = team
        self.pieces = dict()

        if self.team == 1:
            self.pieces[0] = Pawn(0, 1, 1, team)
            self.pieces[1] = Pawn(1, 1, 1, team)
            self.pieces[2] = Pawn(2, 1, 1, team)
            self.pieces[3] = Pawn(3, 1, 1, team)

            self.pieces[4] = Pawn(4, 1, 1, team)
            self.pieces[5] = Pawn(5, 1, 1, team)
            self.pieces[6] = Pawn(6, 1, 1, team)
            self.pieces[7] = Pawn(7, 1, 1, team)

            self.pieces[8] = Rook(0, 0, 1, team)
            self.pieces[9] = Knight(1, 0, 1, team)
            self.pieces[10] = Bishop(2, 0, 1, team)
            self.pieces[11] = Queen(3, 0, 1, team)

            self.pieces[12] = King(4, 0, 1, team)
            self.pieces[13] = Bishop(5, 0, 1, team)
            self.pieces[14] = Knight(6, 0, 1, team)
            self.pieces[15] = Rook(7, 0, 1, team)

            self.king_index = 12

        else:
            self.pieces[0] = Pawn(0, 6, 1, team)
            self.pieces[1] = Pawn(1, 6, 1, team)
            self.pieces[2] = Pawn(2, 6, 1, team)
            self.pieces[3] = Pawn(3, 6, 1, team)

            self.pieces[4] = Pawn(4, 6, 1, team)
            self.pieces[5] = Pawn(5, 6, 1, team)
            self.pieces[6] = Pawn(6, 6, 1, team)
            self.pieces[7] = Pawn(7, 6, 1, team)

            self.pieces[8] = Rook(0, 7, 1, team)
            self.pieces[9] = Knight(1, 7, 1, team)
            self.pieces[10] = Bishop(2, 7, 1, team)
            self.pieces[11] = Queen(3, 7, 1, team)

            self.pieces[12] = King(4, 7, 1, team)
            self.pieces[13] = Bishop(5, 7, 1, team)
            self.pieces[14] = Knight(6, 7, 1, team)
            self.pieces[15] = Rook(7, 7, 1, team)

            self.king_index = 12



    def get_all_pieces(self):
        return [j for _, j in self.pieces.items()]


    def get_living_pieces(self):
        return [j for _, j in self.pieces.items() if j.status != 0]


    def get_possible_moves(self, b):
        moves = []
        for _, i in self.pieces.items():
            moves.extend(i.get_possible_moves(b))

        return [i for i in moves if not self.is_in_check(b, i)]


    def is_in_check(self, b, m = None):
        in_check = False

        if m:
            b = b.execute(m)

        opposing_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        for i in opposing_pieces:
            possible_moves = i.get_possible_moves(b)

            for mp in possible_moves:
                if mp['p'][0] == self.pieces[self.king_index].x and mp['p'][0] == self.pieces[self.king_index].x:
                    in_check = True
        return in_check


class Piece():
    def __init__(self, x, y, status, team):
        self.x = x
        self.y = y
        self.status = status
        self.team = team


    def get_possible_moves(self, b):
        return []


    def get_taken(self):
        self.status = 0


    def analyze_moves_for_square(self, x_diff, y_diff, opposing_team_pieces, same_team_pieces):
        moves = []
        same_team_piece_on_square = False
        opponent_team_piece_on_square = False
        piece_to_take = None
        next_square_loc = (self.x + x_diff, self.y + y_diff)

        for i in opposing_team_pieces:
            if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                opponent_team_piece_on_square = True
                piece_to_take = i
        if not opponent_team_piece_on_square:
            for i in same_team_pieces:
                if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                    same_team_piece_on_square = True
        if not same_team_piece_on_square:
            moves.append({'piece': self, 'piece_taken': piece_to_take, 'p': next_square_loc})
        return moves


class Pawn(Piece):

    def get_possible_moves(self, b):
        living_pieces = b.get_all_living_pieces()
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)

        if self.team == 1:
            direction = 1
        else:
            direction = -1

        moves = []

        piece_on_next_square = False
        next_square_loc = (self.x, self.y + (1 * direction))
        for i in living_pieces:
            if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                piece_on_next_square = True

        if not piece_on_next_square:
            moves.append({'piece': self, 'piece_taken': None, 'p': next_square_loc})

        piece_on_next_square = False
        next_square_loc = (self.x, self.y + (2 * direction))
        for i in living_pieces:
            if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                piece_on_next_square = True

        if not piece_on_next_square:
            moves.append({'piece': self, 'piece_taken': None, 'p': next_square_loc})


        opponent_on_diagonal = False
        piece_to_take = None
        next_square_loc = (self.x + 1, self.y + (1 * direction))
        for i in opposing_team_pieces:
            if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                opponent_on_diagonal = True
                piece_to_take = i

        if not opponent_on_diagonal:
            moves.append({'piece': self, 'piece_taken': piece_to_take, 'p': next_square_loc})


        opponent_on_diagonal = False
        piece_to_take = None
        next_square_loc = (self.x - 1, self.y + (1 * direction))
        for i in opposing_team_pieces:
            if i.x == next_square_loc[0] and i.y == next_square_loc[1]:
                opponent_on_diagonal = True
                piece_to_take = i

        if not opponent_on_diagonal:
            moves.append({'piece': self, 'piece_taken': piece_to_take, 'p': next_square_loc})


        return [i for i in moves if is_on_board(i['p'])]



class Rook(Piece):
    def get_possible_moves(self, b):

        moves = []
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        same_team_pieces = b.get_all_living_pieces_of_team(self.team)

        d1 = True
        d2 = True
        d3 = True
        d4 = True

        for i in range(1, 9):

            if d1:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d1 = False
                moves.extend(new_moves)

            if d2:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d2 = False
                moves.extend(new_moves)

            if d3:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d3 = False
                moves.extend(new_moves)

            if d4:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d4 = False
                moves.extend(new_moves)

        return [i for i in moves if is_on_board(i['p'])]






class Knight(Piece):
    def get_possible_moves(self, b):
        moves = []
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        same_team_pieces = b.get_all_living_pieces_of_team(self.team)
        moves.extend(self.analyze_moves_for_square(2, 1, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(1, 2, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(-2, 1, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(-1, 2, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(2, -1, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(1, -2, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(-2, -1, opposing_team_pieces, same_team_pieces))
        moves.extend(self.analyze_moves_for_square(-1, -2, opposing_team_pieces, same_team_pieces))
        return [i for i in moves if is_on_board(i['p'])]


class Bishop(Piece):
    def get_possible_moves(self, b):

        moves = []
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        same_team_pieces = b.get_all_living_pieces_of_team(self.team)

        d1 = True
        d2 = True
        d3 = True
        d4 = True

        for i in range(1, 9):

            if d1:
                new_moves = self.analyze_moves_for_square(i, i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d1 = False
                moves.extend(new_moves)

            if d2:
                new_moves = self.analyze_moves_for_square(-i, i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d2 = False
                moves.extend(new_moves)

            if d3:
                new_moves = self.analyze_moves_for_square(i, -i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d3 = False
                moves.extend(new_moves)

            if d4:
                new_moves = self.analyze_moves_for_square(-i, -i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d4 = False
                moves.extend(new_moves)



        return [i for i in moves if is_on_board(i['p'])]


class King(Piece):
    def get_possible_moves(self, b):
        moves = []
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        same_team_pieces = b.get_all_living_pieces_of_team(self.team)

        new_moves = self.analyze_moves_for_square(1, 1, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(-1, 1, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(1, -1, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(-1, -1, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(1, 0, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(1, 0, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(1, 0, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        new_moves = self.analyze_moves_for_square(1, 0, opposing_team_pieces, same_team_pieces)
        moves.extend(new_moves)

        return [i for i in moves if is_on_board(i['p'])]


    def is_in_check(self):
        pass


class Queen(Piece):
    def get_possible_moves(self, b):
        moves = []
        opposing_team_pieces = b.get_all_living_pieces_of_team(1 if self.team == 2 else 2)
        same_team_pieces = b.get_all_living_pieces_of_team(self.team)

        d1 = True
        d2 = True
        d3 = True
        d4 = True
        d5 = True
        d6 = True
        d7 = True
        d8 = True

        for i in range(1, 9):

            if d1:
                new_moves = self.analyze_moves_for_square(i, i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d1 = False
                moves.extend(new_moves)

            if d2:
                new_moves = self.analyze_moves_for_square(-i, i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d2 = False
                moves.extend(new_moves)

            if d3:
                new_moves = self.analyze_moves_for_square(i, -i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d3 = False
                moves.extend(new_moves)

            if d4:
                new_moves = self.analyze_moves_for_square(-i, -i, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d4 = False
                moves.extend(new_moves)

            if d5:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d5 = False
                moves.extend(new_moves)

            if d6:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d6 = False
                moves.extend(new_moves)

            if d7:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d7 = False
                moves.extend(new_moves)

            if d8:
                new_moves = self.analyze_moves_for_square(i, 0, opposing_team_pieces, same_team_pieces)
                if not new_moves or new_moves[0]['piece_taken']:
                    d8 = False
                moves.extend(new_moves)

        return [i for i in moves if is_on_board(i['p'])]


def is_on_board(p):
    if p[0] >= 0 and p[0] <=7 and p[1] >= 0 and p[1] <=7:
        return True
    else:
        return False