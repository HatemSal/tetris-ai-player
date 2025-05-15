import random, math, copy, itertools
import tetris_base as tb
import numpy as np

POP_SIZE = 500
TRAIN_GAME_STEPS = 400
TRAIN_GENERATION_STEPS = 1
LOWER_RANGE = -10
UPPER_RANGE= 10
MUT_RATE = 0.1

def extract_feats(board, move_info):
    
    (_, max_h, lines_removed, new_holes, blocking_blocks,piece_sides, floor_sides, wall_sides) = move_info
    
    return [
        max_h,
        lines_removed,
        new_holes,
        blocking_blocks,
        piece_sides,
        floor_sides,
        wall_sides
    ]

def rate_move(board, piece, next_piece, col, weights):
    total_holes_before, blocking_before = tb.calc_initial_move_info(board)
    
    best_score = -math.inf
    best_rotation = 0
    
    for rotation in range(len(tb.PIECES[piece["shape"]])):
        rotation_info  = tb.calc_move_info(board, copy.deepcopy(piece), col, rotation, total_holes_before, blocking_before)
        if not rotation_info[0]:
            continue
        
        features = extract_feats(board, rotation_info)
        
        score = 0
        for i in range(len(features)):
            score += features[i]*weights[i]
            
        if score > best_score:
            best_score = score
            best_rotation = rotation
    
    return best_score, best_rotation

def ai_play_game(weights, max_steps=TRAIN_GAME_STEPS):
    
    board= tb.get_blank_board()
    score = 0
    falling_piece = tb.get_new_piece()
    next_piece = tb.get_new_piece()
    steps = 0
    
    while steps < max_steps:
        best_move = [-math.inf, None, None] # Score, x/col , rotation
        
        for col in range(tb.BOARDWIDTH):
            move_score, rotation = rate_move(board, falling_piece, next_piece, col, weights)
            
            if move_score > best_move[0]:
                best_move = [move_score, col, rotation]
        
        if best_move[1] is None:
            break
    
        best_col, best_rot = best_move[1], best_move[2]
        falling_piece['x'], falling_piece['rotation']  = best_col, best_rot
        
        while tb.is_valid_position(board, falling_piece, adj_Y=1):
            falling_piece['y'] += 1
        
        tb.add_to_board(board, falling_piece)
        
        score+= 1
        
        lines = tb.remove_complete_lines(board)
        if   lines == 1: score += 40
        elif lines == 2: score += 120
        elif lines == 3: score += 300
        elif lines == 4: score += 1200
        
        falling_piece, next_piece = next_piece, tb.get_new_piece()
        steps+=1
        
        if not tb.is_valid_position(board, falling_piece):
            break
   
    return score


def initialize_pop(pop_size = POP_SIZE):
    population = []
    for i in range(pop_size):
        chromosome = [random.uniform(LOWER_RANGE,UPPER_RANGE) for _ in range(7)]
        population.append(chromosome)
    
    return population

def calc_fitness(population):
    fitness_scores = []
    for chromosome in population:
        score = ai_play_game(chromosome)
        if not math.isfinite(score):
            score = 0
        fitness_scores.append(score)
    return fitness_scores

def selection(population, fitness_scores):
    fitness_scores_np = np.array(fitness_scores)
    n = len(fitness_scores)
    highest_chromosomes_idx = fitness_scores_np.argsort()[-(n//2):][::-1]
    population_np = np.array(population)
    population_filtered = population_np[highest_chromosomes_idx]
    return list(population_filtered)

def point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def crossover(population):
    n = len(population)
    offsprings = []
    for i in range(0,n,2):
        parent1 = population[i]
        parent2 = population[ i+1 if i+1 < n else random.randrange(n)]
        child1, child2 = point_crossover(parent1, parent2)
        offsprings.append(child1)
        offsprings.append(child2)
    return offsprings

def mutation(offsprings, mut_rate = MUT_RATE):
    n_genes = len(offsprings[0])
    n_chroms = len(offsprings)
    num_mutated = int(n_genes*n_chroms*MUT_RATE)
    for i in range(num_mutated):
        chrom_idx = random.randrange(n_chroms)
        mut_idx = random.randrange(n_genes)
        offsprings[chrom_idx][mut_idx] = random.uniform(LOWER_RANGE,UPPER_RANGE)

def train(generations = TRAIN_GENERATION_STEPS):
    population = initialize_pop()
    for i in range(generations):
        fitness_scores = calc_fitness(population)
        print(f"Average Score : {sum(fitness_scores)/len(fitness_scores)}")
        population_filtered = selection(population, fitness_scores)
        offsprings = crossover(population_filtered)
        mutation(offsprings)
        population = offsprings + population_filtered
    
    fitness_scores = calc_fitness(population)
    best_match_idx = np.argmax(fitness_scores)
    return population[best_match_idx]
    


if __name__ == "__main__":
    best_weights = train()
    print(f"Final Score: {ai_play_game(best_weights)}")
    
    

        
    