from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

class TicTacToe(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.current_player = 1
        self.board = [0] * 9
        self.num_moves = 0

    def possible_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, move):
        self.board[move] = self.current_player
        self.num_moves += 1

    def unmake_move(self, move):
        self.board[move] = 0
        self.num_moves -= 1

    def lose(self):
        possible_combinations = [
            [1, 2, 3], [4, 5, 6], [7, 8, 9],  # rows
            [1, 4, 7], [2, 5, 8], [3, 6, 9],  # columns
            [1, 5, 9], [3, 5, 7]              # diagonals
        ]

        for combination in possible_combinations:
            if all(self.board[i - 1] == self.current_player for i in combination):
                return True

        return False

    def is_over(self):
        return self.num_moves >= 9 or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0

    def show(self):
        print('\n' + '\n'.join([' '.join([['.', 'O', 'X'][self.board[3 * j + i]]
                                         for i in range(3)]) for j in range(3)]))

if __name__ == "__main__":
    algorithm = Negamax(depth=9)

    TicTacToe([Human_Player(), AI_Player(algorithm)]).play()
