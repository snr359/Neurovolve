import random
import numpy

class tttGame:
    players_per_game = 2
    input_size = 18
    output_size = 9

    def __init__(self):
        self.cells = None
        self.turn = None
        
    def initialize(self):
        # initialize game cells. Each game cell is -1 if empty, and has player's index if marked
        self.cells = numpy.array([-1] * 9)

        # initialize turn counter
        self.turn = int(random.random() > 0.5)

    def check_winner(self):
        winner = None
        for i in range(2):
            if self.cells[0+3*i] != -1 and (self.cells[0+3*i] == self.cells[1+3*i] == self.cells[2+3*i]):
                winner = self.cells[0+3*i]
                break
            elif self.cells[0+i] != -1 and (self.cells[0+i] == self.cells[3+i] == self.cells[6+i]):
                winner = self.cells[0]
                break
        if winner is None:
            if self.cells[0] != -1 and (self.cells[0] == self.cells[4] == self.cells[8]):
                winner = self.cells[0]
            elif self.cells[2] != -1 and (self.cells[2] == self.cells[4] == self.cells[6]):
                winner = self.cells[2]
        return winner
    
    def get_action(self, player):
        # the input format is 18 bools.
        # in every ith two bools, the first of the two is 1 if the going player has a mark there, and the second is 1
        # if the opposing player has a mark there
        # 'turn' is which player is going
        input = [0]*(9*2)
        for i in range(9):
            if self.cells[i] == self.turn:
                input[i*2] = 1
            elif self.cells[i] == (self.turn+1) % 2:
                input[i*2 + 1] = 1
        output = player.get_output(input)
        # get the choice from the output
        empty = self.cells < 0
        choice = numpy.argmax(output * empty)
        return choice
    
    def play_game(self, players):
        # check correct number of players
        assert(len(players) == 2)

        winner = None

        while True:
            # get the player's action
            action = self.get_action(players[self.turn])
    
            # mark the cell with the player's mark
            self.cells[action] = self.turn
    
            # check win/draw condition
            winner = self.check_winner()
            if winner is not None:
                break
            elif -1 not in self.cells:
                break
            # if no win/draw, advance turn
            else:
                self.turn = (self.turn+1) % 2

        win_status = (int(winner==0), int(winner==1))
        return win_status