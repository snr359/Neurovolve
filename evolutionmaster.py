import gamemaster
import popi
import math
import random

class EvolutionMaster:
    class Species:
        def __init__(self):
            self.members = []
            self.representative = None

    def __init__(self):
        self.game_name = None
        self.game_master = None
        self.popi_type = None

    def setup_game(self, game_name):
        self.game_name = game_name
        self.game_master = gamemaster.GameMaster()
        self.game_master.load_game(self.game_name)

    def k_tournament(self, population, k):
        tourney = random.sample(population, k)
        winner = max(tourney, key=lambda p:p.fitness)
        return winner

    def sharing(self, popi1, popi2, compatibility_threshold):
        distance = popi1.similarity(popi2)
        if distance > compatibility_threshold:
            return 0
        else:
            return 1

    def random_search(self, pop_size, max_evals):
        generations = int(math.ceil(max_evals / pop_size))
        best_popi = None
        for g in range(generations):
            population = []

            if best_popi is not None:
                population.append(best_popi)

            for i in range(len(population), pop_size):
                new_popi = self.popi_type()
                new_popi.input_size = self.game_master.game.input_size
                new_popi.output_size = self.game_master.game.output_size
                new_popi.hidden_layer_sizes = [20,30,18]
                new_popi.initialize_network()
                population.append(new_popi)

            self.game_master.evaluate_players(population, 30)
            for p in population:
                p.fitness = p.wins / (p.wins + p.losses)

            best_popi = max(population, key=lambda p:p.fitness)

            progress = (g / generations) * 100
            print('Random search {0}% done'.format(progress))

        return best_popi

    def basic_evolution(self, pop_size, lam, max_evals, tournament_size):
        population = []
        for i in range(pop_size):
            new_popi = self.popi_type()
            new_popi.input_size = self.game_master.game.input_size
            new_popi.output_size = self.game_master.game.output_size
            new_popi.hidden_layer_sizes = [20,30,18]
            new_popi.initialize_network()
            population.append(new_popi)

        self.game_master.evaluate_players(population, 30)
        for p in population:
            p.fitness = p.wins / (p.wins + p.losses)

        evals = pop_size

        progress = (evals / max_evals)*100
        print('Evolution {0}% done'.format(progress))

        while evals < max_evals:
            children = []
            for _ in range(lam):
                parent1 = max(random.sample(population, tournament_size), key=lambda p: p.fitness)
                parent2 = max(random.sample(population, tournament_size), key=lambda p: p.fitness)
                new_child = parent1.recombine(parent2)
                children.append(new_child)

            population.extend(children)
            random.shuffle(population)
            self.game_master.evaluate_players(population, 30)
            for p in population:
                p.fitness = p.wins / (p.wins + p.losses)

            evals += (pop_size + lam)

            progress = (evals / max_evals) * 100
            print('Evolution {0}% done'.format(progress))

            population.sort(key=lambda p: p.fitness, reverse=True)
            population = population[:pop_size]

        best_popi = max(population, key=lambda p: p.fitness)
        return best_popi

    def NEAT_evolution(self, pop_size, lam, max_evals, tournament_size):
        population = []
        innovation_number = 0
        for i in range(pop_size):
            new_popi = self.popi_type()
            new_popi.input_size = self.game_master.game.input_size
            new_popi.output_size = self.game_master.game.output_size
            new_popi.initialize_inputs_outputs()
            for _ in range(random.randint(3,15)):
                if random.random() > 0.5:
                    new_popi.mutate_add_connection(innovation_number)
                    innovation_number += 1
                else:
                    new_popi.mutate_add_node(innovation_number)
                    innovation_number += 2

            population.append(new_popi)

        self.game_master.evaluate_players(population, 30)
        for p in population:
            p.fitness = p.wins / (p.wins + p.losses)

        evals = pop_size

        progress = (evals / max_evals)*100
        print('Evolution {0}% done'.format(progress))

        while evals < max_evals:
            children = []
            for _ in range(lam):
                parent1 = max(random.sample(population, tournament_size), key=lambda p: p.fitness)
                parent2 = max(random.sample(population, tournament_size), key=lambda p: p.fitness)
                new_child = parent1.recombine(parent2)
                children.append(new_child)

            population.extend(children)
            random.shuffle(population)
            self.game_master.evaluate_players(population, 30)
            for p in population:
                p.fitness = p.wins / (p.wins + p.losses)

            evals += (pop_size + lam)

            progress = (evals / max_evals) * 100
            print('Evolution {0}% done'.format(progress))

            population.sort(key=lambda p: p.fitness, reverse=True)
            population = population[:pop_size]

        best_popi = max(population, key=lambda p: p.fitness)
        return best_popi

    def speciated_NEAT_evolution(self, pop_size, lam, max_evals, tournament_size, distance_threshold):
        population = []
        innovation_number = 0
        species_list = []

        for i in range(pop_size):
            new_popi = self.popi_type()
            new_popi.input_size = self.game_master.game.input_size
            new_popi.output_size = self.game_master.game.output_size
            new_popi.initialize_inputs_outputs()
            for _ in range(random.randint(3,15)):
                if random.random() > 0.5:
                    new_popi.mutate_add_connection(innovation_number)
                    innovation_number += 1
                else:
                    new_popi.mutate_add_node(innovation_number)
                    innovation_number += 2

            population.append(new_popi)

        self.game_master.evaluate_players(population, 7)
        for p in population:
            p.fitness = p.wins / (p.wins + p.losses)

        evals = pop_size

        progress = (evals / max_evals)*100
        print('Evolution {0}% done'.format(progress))

        # sort the popis into species
        first_species = self.Species()
        first_species.members = [population[0]]
        first_species.representative = population[0]

        species_list.append(first_species)

        while evals < max_evals:
            children = []
            for _ in range(lam):
                parent1 = self.k_tournament(population, tournament_size)
                parent2 = self.k_tournament(population, tournament_size)
                new_child = parent1.recombine(parent2)
                children.append(new_child)

                # try to place the child in a species
                placed_in_species = False
                for s in species_list:
                    if new_child.similarity(s.representative) <= distance_threshold:
                        s.members.append(new_child)
                        placed_in_species = True
                        break

                # if the child has not yet been placed into a species, make a new one
                if not placed_in_species:
                    new_species = self.Species()
                    new_species.members.append(new_child)
                    new_species.representative = new_child
                    species_list.append(new_species)

            population.extend(children)
            random.shuffle(population)
            self.game_master.evaluate_players(population, 7)
            for p in population:
                p.fitness = p.wins / (p.wins + p.losses)
                
            # do fitness sharing
            for s in species_list:
                for m in s.members:
                    m.fitness /= len(s.members)

            evals += (pop_size + lam)

            progress = (evals / max_evals) * 100
            print('Evolution {0}% done'.format(progress))

            population.sort(key=lambda p: p.fitness, reverse=True)
            survival_threshold_fitness = population[pop_size].fitness

            # do truncation survival for each species
            for s in species_list:
                s.members = list(m for m in s.members if m.fitness >= survival_threshold_fitness)
                if len(s.members) > 0:
                    s.representative = random.choice(s.members)

            # eliminate extinct species
            species_list = list(s for s in species_list if len(s.members) > 0)

            population = []
            for s in species_list:
                population.extend(s.members)

        best_popi = max(population, key=lambda p: p.fitness)
        return best_popi

def main():
    evolution_master = EvolutionMaster()
    evolution_master.popi_type = popi.NEATPopi
    evolution_master.setup_game('ttt')
    best = evolution_master.speciated_NEAT_evolution(500, 500, 10000, 10, 3)

    print(best.wins)
    print(best.losses)

if __name__ == '__main__':
    main()