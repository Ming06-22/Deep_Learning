import torch
import random

class MazeGame:
    def __init__(self, size):
        self.size = size
        self.maze = torch.zeros(size, size)
        self.player_position = (0, 0)
        self.exit_position = (size - 1, size - 1)
        self.generate_maze()

    def check(self):
        def bfs():
            nonlocal record, queue
            
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            temp = set()
            for x, y in queue:
                for a, b in directions:
                    if (x + a, y + b) not in record and 0 <= x + a < self.size and 0 <= y + b < self.size and self.maze[x + a][y + b] != 1:
                        record.add((x + a, y + b))
                        temp.add((x + a, y + b))

            queue = temp

        record = set([(0, 0)])
        queue = set([(0, 0)])
        while queue:
            bfs()
            if (self.size - 1, self.size - 1) in queue:
                return True
        return False
    
    def generate_maze(self):
        while True:
            for i in range(self.size):
                for j in range(self.size):
                    # 生成隨機迷宮，0 表示通道，1 表示障礙物
                    self.maze[i, j] = random.choice([0, 1])

                    # 確保起點和終點是通道
                    self.maze[0, 0] = 0
                    self.maze[self.size - 1, self.size - 1] = 0
                    
            if self.check():
                break

    def move_player(self, direction):
        x, y = self.player_position
        if direction == 'up' and x > 0 and self.maze[x - 1, y] == 0:
            self.player_position = (x - 1, y)
        elif direction == 'down' and x < self.size - 1 and self.maze[x + 1, y] == 0:
            self.player_position = (x + 1, y)
        elif direction == 'left' and y > 0 and self.maze[x, y - 1] == 0:
            self.player_position = (x, y - 1)
        elif direction == 'right' and y < self.size - 1 and self.maze[x, y + 1] == 0:
            self.player_position = (x, y + 1)

    def check_game_status(self):
        if self.player_position == self.exit_position:
            return 'Win'
        elif self.maze[self.player_position[0], self.player_position[1]] == 1:
            return 'Hit obstacle'
        else:
            return 'Continue'

    def display_game(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.player_position:
                    print('P', end=' ')
                elif (i, j) == self.exit_position:
                    print('E', end=' ')
                elif self.maze[i, j] == 1:
                    print('#', end=' ')
                else:
                    print('.', end=' ')
            print()

maze_game = MazeGame(10)

while True:
    maze_game.display_game()
    print("Enter your move (up, down, left, right):")
    move = input().strip().lower()
    maze_game.move_player(move)
    status = maze_game.check_game_status()
    if status == 'Win':
        print("Congratulations! You win!")
        break
    elif status == 'Hit obstacle':
        print("Oops! You hit an obstacle. Game over!")
        break
    else:
        print("Continue exploring...")