import tetris_ai as ta 

def simulate_test_run():
    with open('best_weights.txt', "r") as f:
        line = f.read()
        weights = list(map(float, line.strip().split(",")))
    scores = []
    for i in range(10):
        score = ta.ai_game_simulation(weights)
        scores.append(score)
    average_final_score = sum(scores) / len(scores)
    with open('test_run.txt',"w") as f:
        f.write(f"Average final score for the test run: {average_final_score}")
        
if __name__ == "__main__":
    simulate_test_run()