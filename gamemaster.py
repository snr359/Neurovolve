import random
import math

class GameMaster:
    def __init__(self):
        self.game = None

    def load_game(self, game_name):
        if game_name == 'ttt':
            import game_ttt
            self.game = game_ttt.tttGame

        else:
            print('ERROR: game name {0} not recognized'.format(game_name))

    def evaluate_players(self, players, games_per_player):
        num_players = len(players)

        wins = [0]*num_players
        losses = [0]*num_players

        matchings = []

        # round-robin-esque matchups

        # TODO: this only works for 2 player games so far
        assert(self.game.players_per_game == 2)

        for offset in range(games_per_player):
            i = offset % num_players
            j = (num_players - 1 + offset) % num_players
            for _ in range(int(math.ceil(num_players/2))):
                matchings.append((i, j))
                i = (i + 1) % num_players
                j = (j - 1) % num_players
                if i == j:
                    i = (i + 1) % num_players

        # play the game with all the matchings
        # TODO: multiprocess this
        for m in matchings:
            new_game = self.game()
            new_game.initialize()
            players_this_game = []
            for player_index in m:
                players_this_game.append(players[player_index])
            result = new_game.play_game(players_this_game)
            for i in range(len(result)):
                player_index = m[i]
                if result[i] == 1:
                    wins[player_index] += 1
                else:
                    losses[player_index] += 1

        for i,p in enumerate(players):
            p.wins = wins[i]
            p.losses = losses[i]