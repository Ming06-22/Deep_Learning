import torch
import random
import heapq

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
                    if (x + a, y + b) not in record and 0 <= x + a < self.size and 0 <= y + b < self.size and self.maze[x + a][y + b] == 0:
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

    def shortest_path(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        record = {(0, 0): (0, 1, 0)}
        queue = [(0, (0, 0), (-1, -1))]
        while queue:
            v, (x, y), (mx, my) = heapq.heappop(queue)
            record[(x, y)] = (v, mx, my)
            for a, b in directions:
                if 0 <= a + x < self.size and 0 <= b + y < self.size and self.maze[x + a][y + b] == 0:
                    if (a + x, b + y) not in record or v < record[a + x, b + y][0]:
                        if (a + x, b + y) == (self.size - 1, self.size - 1):
                            queue = []
                            break
                        heapq.heappush(queue, (v + 1, (a + x, b + y), (x, y)))
                        record[(a + x, b + y)] = (v + 1, x, y)
                    
        coordinates = [(self.size - 1, self.size - 1)]
        while True:
            coordinates.append((x, y))
            x, y = mx, my
            _, mx, my = record[(x, y)]
            if (mx, my) == (-1, -1):
                break
        coordinates.append((0, 0))

        result = []
        for i in range(len(coordinates) - 1, 0, -1):
            x, y = coordinates[i]
            a, b = coordinates[i - 1]
            if x > a:
                result.append("up")
            elif x < a:
                result.append("down")
            elif y > b:
                result.append("left")
            else:
                result.append("right")
        return result
    
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
shortest_path = maze_game.shortest_path()
i = 0

while True:
    maze_game.display_game()
    move = shortest_path[i]
    i += 1
    print(f"Your move: {move}")
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