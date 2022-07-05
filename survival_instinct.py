# ------------------ dependencies: --------------------
import argparse 
import math
import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# -------------------- parameters: --------------------
parser = argparse.ArgumentParser()

parser.add_argument("dim", nargs='?', help="Environment size", default=10, type=int)
parser.add_argument("nfood", nargs='?', help="Number of pieces of food", default=40, type=int)
parser.add_argument("npoison", nargs='?', help="Number of poisons", default=0, type=int)
parser.add_argument("energy", nargs='?', help="Starting energy of individuals", default=10, type=int)
parser.add_argument("nsteps", nargs='?', help="Number of moves", default=5, type=int)

parser.add_argument("popsize", nargs='?', help="Population size", default=30, type=int)
parser.add_argument("ngen", nargs='?', help="Number of generations", default=50, type=int)
parser.add_argument("mut_prob", nargs='?', help="Mutation probability", default=0.05, type=float)

args = parser.parse_args()

dim = args.dim
nfood = args.nfood
npoison = args.npoison
energy = args.energy
nsteps = args.nsteps

npop = args.popsize
ngen = args.ngen
mut_p = args.mut_prob


# --------------------- artists: -------------------------
food_img = "icons/cheese.png"
poison_img = "incons/poison.png"
ind_img = "icons/mouse.png"


# --------------------- functions: -----------------------
def convert_to_dec(number, base):
    """ 
    given a number <number> in a certain base <base> represented as a list, it returns the number as a decimal integer.
    """
    return sum(val*(base**i) for i, val in enumerate(reversed(number)))

def convert_dec_to(decimal, base):
    """ 
    given a decimal integer <decimal>, it returns a list representing the number in the base <base>.
    """
    if decimal == 0:
        return 0
    number = []
    while decimal:
        decimal, r = divmod(decimal, base)
        number.insert(0,r)
    return number

def roulette_sampling(lista_prob):
    """
    given a list of probabilities normalized to 1 in input, it selects a position in the list with probability given by the list element at that position.
    """
    i = random.random()
    cum = 0
    for j, p in enumerate(lista_prob):
        cum += p
        if i < cum:
            return j


def imscatter(x, y, image, ax=None, zoom=1):
    """
    it scatters the points whose coordinates are given by the lists <x> and <y> representing them with the icon <image>.
    """
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    #ax.autoscale()
    return artists
# ------------------------------------------------------


# ---------------------- classes: ----------------------
class Individual(list):
        
    def __init__(self, moves = None, position = None, energy = energy):
        if not moves:
            if npoison != 0:
                super(Individual, self).__init__( [random.randint(0,3) for i in range(3**4)] )
            else:
                super(Individual, self).__init__( [random.randint(0,3) for i in range(2**4)] )
        else:
            super(Individual, self).__init__(moves)

        if not position:
            self.position = [random.randint(0,dim-1) , random.randint(0,dim-1)]
        else:
            self.position = position

        self.energy = energy

    def get_fitness(self):
        """
        if returns the energy of the individual.
        """
        return self.energy

    def score(self):
        """
        it returns a normalized measure of the individual survival instinct.
        """
        if npoison != 0:       
            configs = [[0 for i in range( 4 - len(convert_dec_to(n,3)) )] + convert_dec_to(n,3) for n in range(1, len(self)-1) if n != 40]
            score = 0
            for c in configs:
                if 1 in c and 2 not in c:
                    if c[self[convert_to_dec(c,3)]] == 1:
                        score += 1
                return score/14.
        else:
            configs = [[0 for i in range( 4 - len(convert_dec_to(n,2)) )] + convert_dec_to(n,2) for n in range(1, len(self)-1)]
            return sum( 1 for c in configs if c[self[convert_to_dec(c,2)]] == 1 )/14.

    def mate(self, other, nodes = None, offspring_pos = None):
        """
        it returns an individual obtained from a two-point-crossover of the individuals <self> and <other>; 
        crossover nodes and offspring position are generated randomly if not given through the arguments <nodes> and <offspring_pos>.
        """
        if not nodes:
            node1 = random.randint(1,len(self)-2)
            node2 = random.randint(node1+1,len(self)-1)
        else:
            node1, node2 = nodes
        new_genome = self[:node1] + other[node1:node2] + self[node2:]
        if not offspring_pos:
            offspring_pos = [ random.randint(0,dim-1) for i in range(2) ]
        return Individual( moves = new_genome, position = offspring_pos, energy = int((self.energy+other.energy)/2) )

    def mutate(self, prob):
        """ 
        it randomly mutates a gene in the individual's genome, with probability equal to <prob>.
        """
        if random.random() <= prob:
            i = random.randint(0, len(self)-1)
            self[i] = random.randint(0,3)
            
    def get_case(self, food, poison=None): 
        """
        it returns a four-dimensional list cofidying the presence of food and poison in the four cases around the individual; 
        with the convention that 0-1-2-3 stand for up-right-down-left, the list contains 1 in position i if that position there is food, 2 if poison, 0 elsewhere.
        """
        neighbours = [ [self.position[0], (self.position[1]+1)%dim] , [(self.position[0]+1)%dim,self.position[1]] , [self.position[0],(self.position[1]-1)%dim] , [(self.position[0]-1)%dim,self.position[1]] ]
        if poison:
            return [ 1 if pos in food else 2 if pos in poison else 0 for pos in neighbours ]
        else:
            return [ 1 if pos in food else 0 for pos in neighbours ]

    def move_eat(self, food, poison=None):
        """
        it moves the individual to one of the neighboring cases; 
        if there is food in there the individual's energy is increased by 1 and the function returns the value 1, if poison it is decreased by 2.
        """
        kwarg1 = {'poison':poison if poison else None}
        config = self.get_case(food, **kwarg1)
        kwarg2 ={'base':3 if poison else 2}
        i = convert_to_dec(config, **kwarg2)

        if self[i] == 0:
            self.position[1] = (self.position[1] + 1)%dim
        elif self[i] == 1:
            self.position[0] = (self.position[0] + 1)%dim
        elif self[i] == 2:
            self.position[1] = (self.position[1] - 1)%dim
        elif self[i] == 3:
            self.position[0] = (self.position[0] - 1)%dim

        if self.position in food:
            self.energy += 1
            return 1
        elif poison and self.position in poison:
            self.energy -= 2
            return -1
        else:
            self.energy -= 1
            return 0



class Environment:

    def __init__(self):
        self.food = []
        if npoison != 0:
            self.poison = []
        self.pop = []

        self.distribute()
        self.gen_pop()
        
    def distribute(self):
        """
        it (re)distributes food and poison at random positions in the environment.
        """
        self.food = [ [random.randint(0,dim-1) for i in range(2)] for n in range(nfood) ]

        if npoison != 0:
            poison = []
            for i in range(npoison):
                while(True):
                    poison_pos = [ random.randint(0,dim-1) for i in range(2) ]
                    if poison_pos not in self.food:
                        poison.append(poison_pos)
                        break
            self.poison = poison

    def remove_food(self, pos):
        """
        given a position in the environment expressed by a 2 elements list <pos> in input, it removes a piece of food from that position, if any.
        """
        if pos in self.food:
            del self.food[self.food.index(pos)]

    def remove_poison(self, pos):
        """
        given a position in the environment expressed by a 2 elements list <pos> in input, it removes a poison from that position, if any.
        """
        if pos in self.poison:
            del self.poison[self.poison.index(pos)]

    def best_individuals(self, frac):
        """
        it returns a list containing a fraction <frac> of the highest fitness individuals in the population.
        """
        pop = self.pop
        pop.sort(key = lambda i : i.get_fitness(), reverse = True)
        return pop[:int(len(self.pop)*frac)]

    def gen_pop(self, nind = npop, mut_prob = None, parents = None):  
        """
        if <mut_prob> is not given, it fills self.pop with <nind> random individuals located at random position;
        if <mut_prob> is given, it replaces self.pop with <nind> individuals generated from mating individuals selected from the set <parents> with probability proportional to energy; if not given, <parents> is set equal to self.pop
        """
        pop = []
        for i in range(nind):
            while(True):
                ind_pos = [ random.randint(0,dim-1) for i in range(2) ]
                
                if ind_pos not in self.food:
                    if npoison != 0:
                        if ind_pos not in self.poison:
                            pass
                        else:
                            continue
                    if not mut_prob:
                        pop.append( Individual(position = ind_pos) )
                        break
                    else:
                        if not parents:
                            parents = self.pop
                        energies = [ind.get_fitness() for ind in parents]
                        reproduction_probs = [ float(en)/sum(energies) for en in energies ]
                        i = roulette_sampling(reproduction_probs)
                        parent1 = parents[i]
                        while(True):
                            j = roulette_sampling(reproduction_probs)
                            if j != i:
                                parent2 = parents[j]
                                break
                        pop.append( parent1.mate(parent2, offspring_pos = ind_pos) )
                        pop[-1].mutate(mut_prob)
                        break

        self.pop = pop

    def show_environment(self, ax, pause=None):
        """
        it plots current environment configuration on the axis <ax> pausing for a time <pause>.
        """
        ax.clear()
        ax.set_xlim(-0.5,(dim-1)+0.5)
        ax.set_ylim(-0.5,(dim-1)+0.5)
        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.grid()
        food_x = [ x[0] if x not in self.food[i+1:] and x not in self.food[:i] else x[0]+0.08 if x not in self.food[i+1:] and x in self.food[:i] else x[0]-0.08 for i, x in enumerate(self.food)]
        food_y = [ x[1] if x not in self.food[i+1:] and x not in self.food[:i] else x[1]+0.08 if x not in self.food[i+1:] and x in self.food[:i] else x[1]-0.08 for i, x in enumerate(self.food)]
        imscatter( food_x, food_y, food_img, zoom=0.04, ax=ax)
        if npoison != 0:
            poison_x = [ x[0] if x not in self.poison[i+1:] and x not in self.poison[:i] else x[0]+0.08 if x not in self.poison[i+1:] and x in self.poison[:i] else x[0]-0.08 for i, x in enumerate(self.poison)]
            poison_y = [ x[1] if x not in self.poison[i+1:] and x not in self.poison[:i] else x[1]+0.08 if x not in self.poison[i+1:] and x in self.poison[:i] else x[1]-0.08 for i, x in enumerate(self.poison)]
            imscatter( poison_x, poison_y, poison_img, zoom=0.04, ax=ax)
        ind_x = [ x[0] if x not in [ind.position for ind in self.pop[i+1:]] and x not in [ind.position for ind in self.pop[:i]] else x[0]-0.15 if x not in [ind.position for ind in self.pop[i+1:]] and x in [ind.position for ind in self.pop[:i]] else x[0]+0.1 for i, x in enumerate([ind.position for ind in self.pop])]
        ind_y = [ x[1] if x not in [ind.position for ind in self.pop[i+1:]] and x not in [ind.position for ind in self.pop[:i]] else x[1]-0.15 if x not in [ind.position for ind in self.pop[i+1:]] and x in [ind.position for ind in self.pop[:i]] else x[1]+0.1 for i, x in enumerate([ind.position for ind in self.pop])]
        imscatter( ind_x, ind_y, ind_img, zoom=0.03, ax=ax)
        for i, x in enumerate(zip(ind_x,ind_y)):
            ax.annotate(self.pop[i].energy, xy=(x[0], x[1]), xytext=(x[0], x[1]+0.3))
        plt.draw()
        if pause:
            plt.pause(pause) 


    def evolution(self, ax, show_steps=False):
        """
        it evolves environment through a generation; if <show_steps> = True it animates evolution.
        """

        self.show_environment(ax, 0.01)

        for step in range(nsteps):
            if len(self.pop) >= 2:
                for i, ind in enumerate(self.pop):
                    kwarg = {'poison':self.poison if npoison != 0 else None}
                    movement = ind.move_eat(self.food, **kwarg)
                    if movement == 1:
                        self.remove_food(ind.position)
                    elif movement == -1:
                        self.remove_poison(ind.position)
                    if ind.energy <= 0:
                        del self.pop[i]
                    if show_steps and step == 0 and i < 15:
                        self.show_environment(ax, 0.01) 
            else:
                break

        self.show_environment(ax, 0.01)
        if len(self.pop) >= 2:
            return True
        else: 
            return False
# -------------------------------------------------------





# ------------------------ main: -----------------------
fig, axes = plt.subplots(1,2)
fig.set_size_inches(13,6.5)

axes[0].set_xlim(0,ngen)
axes[0].set_ylim(0,1)
axes[0].set_yticks(np.arange(0,1.1,0.1))
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Average survival instinct score")

env = Environment()

for gen in range(ngen):
    print(f"Generation {gen+1}.", end=' ')

    kwarg = {'show_steps':True if gen == 0 else False}
    if not env.evolution(axes[1], **kwarg):
        print("Only one individual survived: the population went extinct.")
        break
    else:
        scores = [ind.score() for ind in env.pop]
        best_ind_learning = max(scores)
        pop_learning = mean(scores)
        print("Survival istinct scores:", end=" ")
        print("best", end=" ")
        print("%.3f" % best_ind_learning, end="; ")
        print("average", end=" ")
        print("%.3f" % pop_learning)

        axes[0].scatter(gen, pop_learning, marker=".", c='blue', clip_on=False)
        plt.draw()
        plt.pause(0.01)

        env.distribute()
        env.gen_pop(nind = npop, mut_prob = mut_p)

plt.show()

