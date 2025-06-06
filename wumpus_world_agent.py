import pygame
import random
import math
import numpy as np
from collections import deque
import time
from enum import Enum

# Initialize Pygame and Mixer
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    VICTORY = "victory"

class SoundManager:
    def __init__(self):
        self.sounds = {}
        self.sound_enabled = True
        self.generate_sounds()
    
    def generate_sounds(self):
        """Generate simple sound effects using pygame"""
        try:
            # Generate different tones for various events
            sample_rate = 22050
            
            # Footstep sound
            duration = 0.1
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                freq = 150 + (i / frames) * 50
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.3 * (1 - i / frames)
            self.sounds['footstep'] = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            
            # Arrow shoot sound
            duration = 0.3
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                freq = 800 - (i / frames) * 600
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.4 * (1 - i / frames)
            self.sounds['arrow'] = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            
            # Wumpus scream
            duration = 0.8
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                freq = 80 + np.sin(i * 0.01) * 40
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.6 * (1 - i / frames)
            self.sounds['scream'] = pygame.sndarray.make_sound((arr * 32767).ast(np.int16))
            
            # Gold pickup
            duration = 0.5
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                freq1 = 523.25  # C5
                freq2 = 659.25  # E5
                freq3 = 783.99  # G5
                wave = (np.sin(2 * np.pi * freq1 * i / sample_rate) +
                       np.sin(2 * np.pi * freq2 * i / sample_rate) +
                       np.sin(2 * np.pi * freq3 * i / sample_rate)) / 3
                arr[i] = wave * 0.5 * (1 - i / frames)
            self.sounds['gold'] = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            
            # Death sound
            duration = 1.0
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                freq = 440 * (1 - i / frames)
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.7 * (1 - i / frames)
            self.sounds['death'] = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            
            # Victory fanfare
            duration = 1.5
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            melody = [523.25, 587.33, 659.25, 783.99, 880.00]  # C-D-E-G-A
            for i in range(frames):
                note_duration = frames // len(melody)
                note_index = min(i // note_duration, len(melody) - 1)
                freq = melody[note_index]
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.6 * (1 - i / frames)
            self.sounds['victory'] = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            
        except Exception as e:
            print(f"Sound generation failed: {e}")
            self.sound_enabled = False
    
    def play_sound(self, sound_name):
        if self.sound_enabled and sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except:
                pass

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_particle(self, x, y, color, velocity, life_time, size=3):
        self.particles.append({
            'x': x, 'y': y, 'vx': velocity[0], 'vy': velocity[1],
            'color': color, 'life': life_time, 'max_life': life_time,
            'size': size
        })
    
    def update(self, dt):
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['life'] -= dt
            particle['vy'] += 200 * dt  # Gravity
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            color = (*particle['color'], int(255 * alpha))
            size = int(particle['size'] * alpha)
            if size > 0:
                try:
                    pygame.draw.circle(screen, particle['color'], 
                                     (int(particle['x']), int(particle['y'])), size)
                except:
                    pass
    
    def create_explosion(self, x, y, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.add_particle(x, y, color, (vx, vy), random.uniform(0.5, 1.5))

class WumpusWorld:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.agent_pos = (0, 0)
        self.agent_dir = "right"
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.score = 0
        self.steps = 0
        self.world = self.generate_world()
        self.percepts = self.get_percepts()
        self.action_history = []
        self.game_state = GameState.PLAYING
        
    def generate_world(self):
        world = [[{"pit": False, "wumpus": False, "gold": False} 
                 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place pits (20% chance per cell, except start)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) != (0, 0) and random.random() < 0.2:
                    world[i][j]["pit"] = True
        
        # Place Wumpus
        wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while wumpus_pos == (0, 0):
            wumpus_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        world[wumpus_pos[0]][wumpus_pos[1]]["wumpus"] = True
        
        # Place gold
        gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        while gold_pos == (0, 0) or gold_pos == wumpus_pos:
            gold_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        world[gold_pos[0]][gold_pos[1]]["gold"] = True
        
        return world
    
    def get_percepts(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        percepts = {
            "stench": False, "breeze": False, "glitter": False,
            "bump": False, "scream": False
        }
        
        # Check adjacent cells
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.world[nx][ny]["wumpus"] and self.wumpus_alive:
                    percepts["stench"] = True
                if self.world[nx][ny]["pit"]:
                    percepts["breeze"] = True
        
        if cell["gold"]:
            percepts["glitter"] = True
        
        return percepts
    
    def move_forward(self):
        self.steps += 1
        self.score -= 1
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        direction_map = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1)
        }
        
        dx, dy = direction_map[self.agent_dir]
        new_x, new_y = x + dx, y + dy
        
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)
            self.percepts = self.get_percepts()
            self.action_history.append(("move", (new_x, new_y)))
            return True
        else:
            self.percepts["bump"] = True
            self.action_history.append(("bump", None))
            return False
    
    def turn_left(self):
        self.steps += 1
        self.score -= 1
        dirs = ["up", "left", "down", "right"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()
        self.action_history.append(("turn_left", self.agent_dir))
    
    def turn_right(self):
        self.steps += 1
        self.score -= 1
        dirs = ["up", "right", "down", "left"]
        idx = dirs.index(self.agent_dir)
        self.agent_dir = dirs[(idx + 1) % 4]
        self.percepts = self.get_percepts()
        self.action_history.append(("turn_right", self.agent_dir))
    
    def shoot_arrow(self):
        self.steps += 1
        self.score -= 10
        if not self.has_arrow:
            return False
        
        self.has_arrow = False
        x, y = self.agent_pos
        
        direction_map = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1)
        }
        
        dx, dy = direction_map[self.agent_dir]
        
        # Check if arrow hits wumpus
        for i in range(1, self.grid_size):
            check_x, check_y = x + dx * i, y + dy * i
            if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                break
            if self.world[check_x][check_y]["wumpus"] and self.wumpus_alive:
                self.wumpus_alive = False
                self.percepts["scream"] = True
                return True
        
        return False
    
    def grab_gold(self):
        self.steps += 1
        x, y = self.agent_pos
        if self.world[x][y]["gold"]:
            self.has_gold = True
            self.world[x][y]["gold"] = False
            self.percepts["glitter"] = False
            self.score += 1000
            return True
        return False
    
    def is_game_over(self):
        x, y = self.agent_pos
        cell = self.world[x][y]
        if cell["pit"] or (cell["wumpus"] and self.wumpus_alive):
            self.score -= 1000
            self.game_state = GameState.GAME_OVER
            return "lose"
        if self.has_gold and self.agent_pos == (0, 0):
            self.game_state = GameState.VICTORY
            return "win"
        return "continue"

class KnowledgeBase:
    def __init__(self, world_size):
        self.world_size = world_size
        self.wumpus_alive = True
        self.percepts = {}  # pos -> percepts
        self.safe_cells = {(0, 0)}
        self.pit_cells = set()
        self.wumpus_cells = set()
        self.no_pit_cells = set()
        self.no_wumpus_cells = set()
    
    def add_percept(self, pos, percept):
        self.percepts[pos] = percept
        
        # If no breeze, no adjacent pits
        if not percept["breeze"]:
            for adj_pos in self.get_adjacent_cells(pos):
                self.no_pit_cells.add(adj_pos)
        
        # If no stench, no adjacent wumpus
        if not percept["stench"]:
            for adj_pos in self.get_adjacent_cells(pos):
                self.no_wumpus_cells.add(adj_pos)
    
    def get_adjacent_cells(self, pos):
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= adj_pos[0] < self.world_size and 0 <= adj_pos[1] < self.world_size:
                adjacent.append(adj_pos)
        return adjacent
    
    def infer(self):
        # Inference based on percepts
        for pos, percept in self.percepts.items():
            adjacent = self.get_adjacent_cells(pos)
            
            # Breeze inference
            if percept["breeze"]:
                unknown_adjacent = [adj for adj in adjacent 
                                   if adj not in self.no_pit_cells and adj not in self.pit_cells]
                if len(unknown_adjacent) == 1:
                    self.pit_cells.add(unknown_adjacent[0])
            
            # Stench inference
            if percept["stench"] and self.wumpus_alive:
                unknown_adjacent = [adj for adj in adjacent 
                                   if adj not in self.no_wumpus_cells and adj not in self.wumpus_cells]
                if len(unknown_adjacent) == 1:
                    self.wumpus_cells.add(unknown_adjacent[0])
                
                # If all but one adjacent cells are confirmed safe, the remaining must be wumpus
                safe_adjacent = [adj for adj in adjacent if adj in self.no_wumpus_cells]
                if len(safe_adjacent) == len(adjacent) - 1:
                    for adj in adjacent:
                        if adj not in self.no_wumpus_cells and adj not in self.wumpus_cells:
                            self.wumpus_cells.add(adj)
    
    def is_safe(self, pos):
        return (pos in self.no_pit_cells and 
                (pos in self.no_wumpus_cells or not self.wumpus_alive))
    
    def is_dangerous(self, pos):
        return pos in self.pit_cells or (pos in self.wumpus_cells and self.wumpus_alive)

class LogicAgent:
    def __init__(self, world_size=4):
        self.world_size = world_size
        self.knowledge_base = KnowledgeBase(world_size)
        self.visited = set()
        self.visited.add((0, 0))  # Starting position is safe
        self.safe_cells = {(0, 0)}
        self.dangerous_cells = set()
        self.path_history = [(0, 0)]
        self.current_pos = (0, 0)
        self.current_dir = "right"
        self.has_gold = False
        self.wumpus_killed = False
        self.wumpus_pos = None
        self.pit_probabilities = {}
        self.wumpus_probabilities = {}
        self.frontier_cells = set()
        self.backtrack_path = []
        
    def update_knowledge(self, pos, percepts):
        self.current_pos = pos
        self.knowledge_base.add_percept(pos, percepts)
        self.visited.add(pos)
        
        # Update safe cells based on percepts
        if not percepts["breeze"] and not percepts["stench"]:
            for adj_pos in self.get_adjacent_cells(pos):
                if self.is_valid_pos(adj_pos):
                    self.safe_cells.add(adj_pos)
                    if adj_pos in self.dangerous_cells:
                        self.dangerous_cells.remove(adj_pos)
        
        # If scream heard, wumpus is dead
        if percepts["scream"]:
            self.wumpus_killed = True
            self.knowledge_base.wumpus_alive = False
        
        # Update frontier cells
        self.update_frontier()
        
        # Update probabilities
        self.update_probabilities()
        
        # Try to infer wumpus location
        self.infer_wumpus_location()
    
    def is_valid_pos(self, pos):
        return 0 <= pos[0] < self.world_size and 0 <= pos[1] < self.world_size
    
    def get_adjacent_cells(self, pos):
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_pos = (pos[0] + dx, pos[1] + dy)
            if self.is_valid_pos(adj_pos):
                adjacent.append(adj_pos)
        return adjacent
    
    def update_frontier(self):
        """Update the frontier cells (unvisited adjacent to visited)"""
        new_frontier = set()
        for cell in self.visited:
            for adj in self.get_adjacent_cells(cell):
                if adj not in self.visited and adj not in self.dangerous_cells:
                    new_frontier.add(adj)
        self.frontier_cells = new_frontier
    
    def update_probabilities(self):
        """Update pit and wumpus probabilities for frontier cells"""
        # Reset probabilities
        for i in range(self.world_size):
            for j in range(self.world_size):
                pos = (i, j)
                self.pit_probabilities[pos] = 0.0
                self.wumpus_probabilities[pos] = 0.0
        
        # Calculate pit probabilities based on breezes
        for pos in self.knowledge_base.percepts:
            if self.knowledge_base.percepts[pos]["breeze"]:
                unknown_adjacent = [adj for adj in self.get_adjacent_cells(pos) 
                                  if adj not in self.knowledge_base.no_pit_cells]
                if unknown_adjacent:
                    prob = 1.0 / len(unknown_adjacent)
                    for adj in unknown_adjacent:
                        self.pit_probabilities[adj] = max(self.pit_probabilities[adj], prob)
        
        # Calculate wumpus probabilities based on stenches
        if self.knowledge_base.wumpus_alive:
            for pos in self.knowledge_base.percepts:
                if self.knowledge_base.percepts[pos]["stench"]:
                    unknown_adjacent = [adj for adj in self.get_adjacent_cells(pos) 
                                      if adj not in self.knowledge_base.no_wumpus_cells]
                    if unknown_adjacent:
                        prob = 1.0 / len(unknown_adjacent)
                        for adj in unknown_adjacent:
                            self.wumpus_probabilities[adj] = max(self.wumpus_probabilities[adj], prob)
    
    def infer_wumpus_location(self):
        """Try to determine wumpus location definitively"""
        if self.wumpus_killed:
            return
        
        # Look for cells that must contain the wumpus
        possible_locations = []
        for pos in self.frontier_cells:
            if self.wumpus_probabilities[pos] >= 0.99:
                possible_locations.append(pos)
        
        if len(possible_locations) == 1:
            self.wumpus_pos = possible_locations[0]
            self.dangerous_cells.add(self.wumpus_pos)
    
    def infer_safe_moves(self):
        self.knowledge_base.infer()
        
        # Update safe and dangerous cells based on inference
        for i in range(self.world_size):
            for j in range(self.world_size):
                pos = (i, j)
                if self.knowledge_base.is_safe(pos):
                    self.safe_cells.add(pos)
                    if pos in self.dangerous_cells:
                        self.dangerous_cells.remove(pos)
                elif self.knowledge_base.is_dangerous(pos):
                    self.dangerous_cells.add(pos)
    
    def find_path_to_target(self, target):
        """Find path to target using A* algorithm with safe cells"""
        if target == self.current_pos:
            return []
        
        def heuristic(pos):
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        
        open_set = {self.current_pos}
        came_from = {}
        g_score = {pos: float('inf') for pos in self.safe_cells}
        g_score[self.current_pos] = 0
        f_score = {pos: float('inf') for pos in self.safe_cells}
        f_score[self.current_pos] = heuristic(self.current_pos)
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score[pos])
            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            open_set.remove(current)
            
            for neighbor in self.get_adjacent_cells(current):
                if neighbor not in self.safe_cells:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        return None

    
    
    def choose_action(self, env):
        # First, grab gold if present
        if env.percepts["glitter"] and not self.has_gold:
            return "grab_gold"
        
        # If we have gold, return to start
        if self.has_gold:
            path = self.find_path_to_target((0, 0))
            if path and len(path) > 0:
                return self.get_move_action(env, path[0])
            return "stay"
        
        # Update knowledge with current percepts
        self.infer_safe_moves()
        
        # Try to shoot wumpus if we detect stench and have arrow
        if (env.percepts["stench"] and env.has_arrow and not self.wumpus_killed and 
            self.wumpus_pos and self.can_shoot_wumpus(env, self.wumpus_pos)):
            return self.align_and_shoot(env, self.wumpus_pos)
        
        # If we feel a breeze, move away from dangerous directions
        if env.percepts["breeze"]:
            safe_moves = self.get_safe_moves_away_from_breeze()
            if safe_moves:
                return random.choice(safe_moves)
        
        # Find unexplored safe cells
        unexplored_safe = self.safe_cells - self.visited
        
        if unexplored_safe:
            # Choose closest unexplored safe cell
            target = min(unexplored_safe, key=lambda p: self.manhattan_distance(p, self.current_pos))
            path = self.find_path_to_target(target)
            if path and len(path) > 0:
                return self.get_move_action(env, path[0])
        
        # If no safe unexplored cells, try to explore frontier cells
        if self.frontier_cells:
            target = self.choose_frontier_target()
            if target:
                path = self.find_path_to_target(target)
                if path and len(path) > 0:
                    return self.get_move_action(env, path[0])
        
        # If stuck, try to backtrack
        if not self.backtrack_path:
            self.backtrack_path = self.find_backtrack_path()
        
        if self.backtrack_path:
            next_pos = self.backtrack_path.pop(0)
            return self.get_move_action(env, next_pos)
        
        # If we get here, there are truly no safe moves
        # Check if we're completely surrounded by dangers
        all_adjacent = self.get_adjacent_cells(self.current_pos)
        safe_adjacent = [pos for pos in all_adjacent if pos in self.safe_cells]
        
        if not safe_adjacent:
            return "stay"  # Signal that we're stuck
        
        # As a last resort, try moving to the least dangerous adjacent cell
        least_dangerous = min(all_adjacent, 
                             key=lambda p: self.pit_probabilities.get(p, 0) + 
                                          (0 if self.wumpus_killed else self.wumpus_probabilities.get(p, 0)))
        
        return self.get_move_action(env, least_dangerous)
    
    def get_safe_moves_away_from_breeze(self):
        """Determine safe moves away from breezy areas"""
        safe_moves = []
        x, y = self.current_pos
        
        # Check all adjacent cells
        adjacent = self.get_adjacent_cells((x, y))
        
        # Find safe adjacent cells that don't have breeze
        for adj_pos in adjacent:
            if adj_pos in self.safe_cells:
                # Check if this direction is away from potential pits
                if not self.is_potential_pit_direction(adj_pos):
                    safe_moves.append(self.get_move_action(None, adj_pos))
        
        return safe_moves
    
    def is_potential_pit_direction(self, target_pos):
        """Check if moving to target_pos would take us toward potential pits"""
        x, y = self.current_pos
        tx, ty = target_pos
        
        # Get all adjacent cells to target that we haven't visited
        adjacent_to_target = self.get_adjacent_cells(target_pos)
        unvisited_adjacent = [pos for pos in adjacent_to_target if pos not in self.visited]
        
        # If any unvisited adjacent cells have high pit probability, this is dangerous
        for pos in unvisited_adjacent:
            if self.pit_probabilities.get(pos, 0) > 0.5:  # Threshold for considering dangerous
                return True
        
        return False
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def choose_frontier_target(self):
        """Choose the safest frontier cell based on probability"""
        best_cell = None
        best_score = -float('inf')
        
        for cell in self.frontier_cells:
            # Calculate combined risk (pit or wumpus)
            pit_risk = self.pit_probabilities.get(cell, 0.0)
            wumpus_risk = 0.0 if self.wumpus_killed else self.wumpus_probabilities.get(cell, 0.0)
            total_risk = pit_risk + wumpus_risk - pit_risk * wumpus_risk
            
            # Calculate potential information gain
            info_gain = len(self.get_adjacent_cells(cell)) / 4.0
            
            # Score balances risk vs information gain
            score = info_gain * (1 - total_risk)
            
            if score > best_score:
                best_score = score
                best_cell = cell
        
        # Only venture into frontier if risk is acceptable
        if best_score > 0.3:  # Empirical threshold
            return best_cell
        return None
    
    def find_backtrack_path(self):
        """Find a path back to a previously visited cell"""
        # Try to find a path to the closest visited cell with unexplored neighbors
        for visited_cell in sorted(self.visited, key=lambda p: self.manhattan_distance(p, self.current_pos)):
            if any(adj not in self.visited for adj in self.get_adjacent_cells(visited_cell)):
                path = self.find_path_to_target(visited_cell)
                if path:
                    return path
        return []
    
    def can_shoot_wumpus(self, env, wumpus_pos):
        x, y = self.current_pos
        wx, wy = wumpus_pos
        
        # Check if wumpus is in line of sight
        if x == wx:  # Same row
            return True
        elif y == wy:  # Same column
            return True
        return False
    
    def align_and_shoot(self, env, wumpus_pos):
        x, y = self.current_pos
        wx, wy = wumpus_pos
        
        target_dir = None
        if x == wx:  # Same row
            if wy > y:
                target_dir = "right"
            else:
                target_dir = "left"
        elif y == wy:  # Same column
            if wx > x:
                target_dir = "down"
            else:
                target_dir = "up"
        
        if target_dir == self.current_dir:
            return "shoot_arrow"
        else:
            return self.get_turn_action(target_dir)
    
    def get_move_action(self, env, target_pos):
        x, y = self.current_pos
        tx, ty = target_pos
        
        # Determine required direction
        if tx < x:
            target_dir = "up"
        elif tx > x:
            target_dir = "down"
        elif ty < y:
            target_dir = "left"
        else:
            target_dir = "right"
        
        # Turn if necessary
        if self.current_dir != target_dir:
            return self.get_turn_action(target_dir)
        else:
            return "move_forward"
    
    def get_turn_action(self, target_dir):
        dirs = ["up", "right", "down", "left"]
        current_idx = dirs.index(self.current_dir)
        target_idx = dirs.index(target_dir)
        
        # Calculate shortest turn
        diff = (target_idx - current_idx) % 4
        if diff == 1 or diff == -3:
            return "turn_right"
        else:
            return "turn_left"

class WumpusVisualizer:
    def __init__(self):
        self.screen_width = 900
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Wumpus World - Logic-Based Agent")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game components
        self.world = None
        self.agent = None
        self.sound_manager = SoundManager()
        self.particle_system = ParticleSystem()
        
        # Visual settings
        self.cell_size = 120
        self.grid_offset_x = 100
        self.grid_offset_y = 100
        
        # Colors
        self.colors = {
            'background': (10, 10, 30),
            'grid': (40, 40, 80),
            'safe': (20, 80, 20),
            'visited': (30, 60, 30),
            'dangerous': (80, 20, 20),
            'agent': (100, 150, 255),
            'pit': (0, 0, 0),
            'wumpus': (200, 50, 50),
            'gold': (255, 215, 0),
            'text': (255, 255, 255),
            'button': (60, 60, 120),
            'button_hover': (80, 80, 140),
            'logic': (180, 180, 220)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # UI elements
        self.buttons = {}
        self.create_ui_elements()
        
        # Animation variables
        self.time = 0
        self.agent_animation_offset = 0
        self.breathing_effect = 0
        
        # Game state
        self.game_state = GameState.MENU
        self.auto_play = False
        self.auto_play_speed = 0.5
        self.last_auto_step = 0
        self.agent_stuck = False
        
        self.reset_game()
    
    def create_ui_elements(self):
        button_width, button_height = 120, 40
        start_x = self.screen_width - 150
        
        self.buttons = {
            'play': pygame.Rect(start_x, 50, button_width, button_height),
            'step': pygame.Rect(start_x, 100, button_width, button_height),
            'reset': pygame.Rect(start_x, 150, button_width, button_height),
            'auto': pygame.Rect(start_x, 200, button_width, button_height),
            'sound': pygame.Rect(start_x, 250, button_width, button_height),
            'new_game': pygame.Rect(self.screen_width//2 - 100, 400, 200, 50),
            'quit': pygame.Rect(self.screen_width//2 - 100, 480, 200, 50)
        }
    
    def reset_game(self):
        self.world = WumpusWorld()
        self.agent = LogicAgent()
        self.game_state = GameState.PLAYING
        self.particle_system.particles.clear()
        self.agent_stuck = False
        self.create_ui_elements()  # Recreate buttons to ensure proper state
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if self.game_state == GameState.PLAYING:
                    if event.key == pygame.K_SPACE:
                        self.step_game()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        self.auto_play = not self.auto_play
                    elif event.key == pygame.K_s:
                        self.sound_manager.sound_enabled = not self.sound_manager.sound_enabled
                elif self.game_state == GameState.MENU:
                    if event.key == pygame.K_RETURN:
                        self.game_state = GameState.PLAYING
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if self.game_state == GameState.MENU:
                    if self.buttons['new_game'].collidepoint(mouse_pos):
                        self.game_state = GameState.PLAYING
                    elif self.buttons['quit'].collidepoint(mouse_pos):
                        self.running = False
                
                elif self.game_state == GameState.PLAYING:
                    if self.buttons['step'].collidepoint(mouse_pos):
                        self.step_game()
                    elif self.buttons['reset'].collidepoint(mouse_pos):
                        self.reset_game()
                    elif self.buttons['auto'].collidepoint(mouse_pos):
                        self.auto_play = not self.auto_play
                    elif self.buttons['sound'].collidepoint(mouse_pos):
                        self.sound_manager.sound_enabled = not self.sound_manager.sound_enabled
                
                elif self.game_state in [GameState.GAME_OVER, GameState.VICTORY]:
                    if self.buttons['new_game'].collidepoint(mouse_pos):
                        self.reset_game()
                    elif self.buttons['quit'].collidepoint(mouse_pos):
                        self.running = False
    
    def step_game(self):
        if self.game_state != GameState.PLAYING:
            return
        
        # Update agent knowledge
        self.agent.update_knowledge(self.world.agent_pos, self.world.percepts)
        
        # Choose and execute action
        action = self.agent.choose_action(self.world)
        old_pos = self.world.agent_pos
        
        # Check if agent is stuck (no possible moves)
        if action == "stay":
            all_adjacent = self.agent.get_adjacent_cells(self.world.agent_pos)
            safe_adjacent = [pos for pos in all_adjacent if pos in self.agent.safe_cells]
            
            if not safe_adjacent and not self.world.has_gold:
                self.agent_stuck = True
                self.game_state = GameState.GAME_OVER
                return
        
        if action == "move_forward":
            moved = self.world.move_forward()
            if moved:
                self.sound_manager.play_sound('footstep')
                # Add movement particles
                x, y = old_pos
                screen_x = self.grid_offset_x + y * self.cell_size + self.cell_size // 2
                screen_y = self.grid_offset_y + x * self.cell_size + self.cell_size // 2
                for _ in range(5):
                    self.particle_system.add_particle(
                        screen_x, screen_y, (100, 150, 255), 
                        (random.uniform(-30, 30), random.uniform(-30, 30)), 0.5, 2
                    )
        
        elif action in ["turn_left", "turn_right"]:
            if action == "turn_left":
                self.world.turn_left()
            else:
                self.world.turn_right()
            self.agent.current_dir = self.world.agent_dir
        
        elif action == "shoot_arrow":
            hit = self.world.shoot_arrow()
            self.sound_manager.play_sound('arrow')
            if hit:
                self.sound_manager.play_sound('scream')
                # Create explosion effect
                for i in range(self.world.grid_size):
                    for j in range(self.world.grid_size):
                        if self.world.world[i][j]["wumpus"]:
                            screen_x = self.grid_offset_x + j * self.cell_size + self.cell_size // 2
                            screen_y = self.grid_offset_y + i * self.cell_size + self.cell_size // 2
                            self.particle_system.create_explosion(screen_x, screen_y, (200, 50, 50))
            self.agent.wumpus_killed = not self.world.wumpus_alive
        
        elif action == "grab_gold":
            grabbed = self.world.grab_gold()
            if grabbed:
                self.sound_manager.play_sound('gold')
                x, y = self.world.agent_pos
                screen_x = self.grid_offset_x + y * self.cell_size + self.cell_size // 2
                screen_y = self.grid_offset_y + x * self.cell_size + self.cell_size // 2
                self.particle_system.create_explosion(screen_x, screen_y, (255, 215, 0), 30)
            self.agent.has_gold = self.world.has_gold
        
        # Check game state
        result = self.world.is_game_over()
        if result == "win":
            self.sound_manager.play_sound('victory')
            self.game_state = GameState.VICTORY
        elif result == "lose":
            self.sound_manager.play_sound('death')
            self.game_state = GameState.GAME_OVER
    

    def calculate_risk_threshold(self):
        """Dynamic risk assessment based on current game state"""
        base_risk = 0.3
        
        # Increase risk tolerance if gold is located
        if self.gold_location and not self.has_gold:
            base_risk *= 1.5
        
        # Decrease risk tolerance if return path is uncertain
        if self.has_gold and not self.safe_path_to_start():
            base_risk *= 0.5
        
        return base_risk


    def update(self, dt):
        self.time += dt
        self.agent_animation_offset = math.sin(self.time * 3) * 3
        self.breathing_effect = math.sin(self.time * 2) * 0.1 + 1
        
        # Auto-play logic
        if self.auto_play and self.game_state == GameState.PLAYING:
            if time.time() - self.last_auto_step > self.auto_play_speed:
                self.step_game()
                self.last_auto_step = time.time()
        
        # Update particle system
        self.particle_system.update(dt)
    
    def draw_grid(self):
        # Draw background grid with glow effect
        for i in range(self.world.grid_size + 1):
            # Vertical lines
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (x, self.grid_offset_y), 
                           (x, self.grid_offset_y + self.world.grid_size * self.cell_size), 2)
            
            # Horizontal lines
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['grid'], 
                           (self.grid_offset_x, y), 
                           (self.grid_offset_x + self.world.grid_size * self.cell_size, y), 2)
    
    def draw_cells(self):
        for i in range(self.world.grid_size):
            for j in range(self.world.grid_size):
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size
                
                # Cell background
                color = self.colors['background']
                if (i, j) in self.agent.visited:
                    color = self.colors['visited']
                elif (i, j) in self.agent.safe_cells:
                    color = self.colors['safe']
                elif (i, j) in self.agent.dangerous_cells:
                    color = self.colors['dangerous']
                
                pygame.draw.rect(self.screen, color, 
                               (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))
                
                # Draw world objects
                center_x = x + self.cell_size // 2
                center_y = y + self.cell_size // 2
                
                # Pit
                if self.world.world[i][j]["pit"]:
                    radius = int(30 * self.breathing_effect)
                    pygame.draw.circle(self.screen, self.colors['pit'], 
                                     (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, (50, 50, 50), 
                                     (center_x, center_y), radius, 3)
                    
                    # Add text
                    text = self.font_small.render("PIT", True, self.colors['text'])
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.screen.blit(text, text_rect)
                
                # Wumpus
                if self.world.world[i][j]["wumpus"]:
                    if self.world.wumpus_alive:
                        radius = int(35 * self.breathing_effect)
                        pygame.draw.circle(self.screen, self.colors['wumpus'], 
                                         (center_x, center_y), radius)
                        
                        # Eyes
                        eye_offset = 10
                        pygame.draw.circle(self.screen, (255, 255, 255), 
                                         (center_x - eye_offset, center_y - 5), 4)
                        pygame.draw.circle(self.screen, (255, 255, 255), 
                                         (center_x + eye_offset, center_y - 5), 4)
                        pygame.draw.circle(self.screen, (0, 0, 0), 
                                         (center_x - eye_offset, center_y - 5), 2)
                        pygame.draw.circle(self.screen, (0, 0, 0), 
                                         (center_x + eye_offset, center_y - 5), 2)
                        
                        text = self.font_small.render("WUMPUS", True, self.colors['text'])
                    else:
                        radius = 25
                        pygame.draw.circle(self.screen, (100, 100, 100), 
                                         (center_x, center_y), radius)
                        text = self.font_small.render("DEAD", True, self.colors['text'])
                    
                    text_rect = text.get_rect(center=(center_x, center_y + 25))
                    self.screen.blit(text, text_rect)
                
                # Gold
                if self.world.world[i][j]["gold"]:
                    glow_radius = int(40 * (1 + 0.3 * math.sin(self.time * 4)))
                    pygame.draw.circle(self.screen, (*self.colors['gold'], 50), 
                                     (center_x, center_y), glow_radius)
                    pygame.draw.circle(self.screen, self.colors['gold'], 
                                     (center_x, center_y), 20)
                    
                    # Sparkle effect
                    for angle in range(0, 360, 45):
                        sparkle_x = center_x + math.cos(math.radians(angle + self.time * 50)) * 15
                        sparkle_y = center_y + math.sin(math.radians(angle + self.time * 50)) * 15
                        pygame.draw.circle(self.screen, (255, 255, 255), 
                                         (int(sparkle_x), int(sparkle_y)), 2)
                    
                    text = self.font_small.render("GOLD", True, (0, 0, 0))
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.screen.blit(text, text_rect)
                
                # Draw pit probability
                pit_prob = self.agent.pit_probabilities.get((i, j), 0)
                if pit_prob > 0:
                    prob_text = self.font_small.render(f"P:{pit_prob:.2f}", True, self.colors['logic'])
                    self.screen.blit(prob_text, (x + 5, y + 5))
                
                # Draw wumpus probability
                wumpus_prob = self.agent.wumpus_probabilities.get((i, j), 0)
                if wumpus_prob > 0:
                    prob_text = self.font_small.render(f"W:{wumpus_prob:.2f}", True, self.colors['logic'])
                    self.screen.blit(prob_text, (x + 5, y + self.cell_size - 25))
    
    def draw_agent(self):
        x, y = self.world.agent_pos
        screen_x = self.grid_offset_x + y * self.cell_size + self.cell_size // 2
        screen_y = self.grid_offset_y + x * self.cell_size + self.cell_size // 2 + self.agent_animation_offset
        
        # Agent body
        pygame.draw.circle(self.screen, self.colors['agent'], 
                         (int(screen_x), int(screen_y)), 25)
        pygame.draw.circle(self.screen, (255, 255, 255), 
                         (int(screen_x), int(screen_y)), 25, 3)
        
        # Direction indicator
        angle_map = {"up": -90, "right": 0, "down": 90, "left": 180}
        angle = math.radians(angle_map[self.world.agent_dir])
        
        arrow_length = 20
        end_x = screen_x + math.cos(angle) * arrow_length
        end_y = screen_y + math.sin(angle) * arrow_length
        
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (screen_x, screen_y), (end_x, end_y), 4)
        
        # Arrow head
        head_angle1 = angle + math.radians(150)
        head_angle2 = angle + math.radians(-150)
        head_length = 8
        
        head1_x = end_x + math.cos(head_angle1) * head_length
        head1_y = end_y + math.sin(head_angle1) * head_length
        head2_x = end_x + math.cos(head_angle2) * head_length
        head2_y = end_y + math.sin(head_angle2) * head_length
        
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (end_x, end_y), (head1_x, head1_y), 4)
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (end_x, end_y), (head2_x, head2_y), 4)
    
    def draw_ui(self):
        # Only draw the sidebar UI if we're in the playing state
        if self.game_state == GameState.PLAYING:
            # Draw sidebar
            sidebar_rect = pygame.Rect(self.screen_width - 200, 0, 200, self.screen_height)
            pygame.draw.rect(self.screen, (20, 20, 50), sidebar_rect)
            
            # Draw buttons
            for name, rect in self.buttons.items():
                if name in ['new_game', 'quit']:
                    continue
                
                color = self.colors['button_hover'] if rect.collidepoint(pygame.mouse.get_pos()) else self.colors['button']
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                pygame.draw.rect(self.screen, (100, 100, 150), rect, 2, border_radius=5)
                
                # Button labels
                if name == 'play':
                    label = "Pause" if self.auto_play else "Play"
                elif name == 'sound':
                    label = "Sound ON" if self.sound_manager.sound_enabled else "Sound OFF"
                elif name == 'auto':
                    label = "Auto: ON" if self.auto_play else "Auto: OFF"
                else:
                    label = name.replace('_', ' ').title()
                
                text = self.font_small.render(label, True, self.colors['text'])
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            
            # Draw game info
            info_y = 350
            info_text = [
                f"Score: {self.world.score}",
                f"Steps: {self.world.steps}",
                f"Gold: {'Yes' if self.world.has_gold else 'No'}",
                f"Arrow: {'Yes' if self.world.has_arrow else 'No'}",
                f"Wumpus: {'Alive' if self.world.wumpus_alive else 'Dead'}"
            ]
            
            for text in info_text:
                rendered = self.font_small.render(text, True, self.colors['text'])
                self.screen.blit(rendered, (self.screen_width - 190, info_y))
                info_y += 30
            
            # Draw percepts
            percept_y = 500
            percept_text = "Percepts:"
            rendered = self.font_small.render(percept_text, True, self.colors['text'])
            self.screen.blit(rendered, (self.screen_width - 190, percept_y))
            percept_y += 30
            
            for name, value in self.world.percepts.items():
                if value:
                    text = f"- {name.capitalize()}"
                    rendered = self.font_small.render(text, True, self.colors['text'])
                    self.screen.blit(rendered, (self.screen_width - 180, percept_y))
                    percept_y += 25
            
            # Draw agent reasoning
            reasoning_y = 600
            reasoning_text = "Agent Reasoning:"
            rendered = self.font_small.render(reasoning_text, True, self.colors['text'])
            self.screen.blit(rendered, (self.screen_width - 190, reasoning_y))
            reasoning_y += 25
            
            # Show safe cells count
            safe_text = f"Safe Cells: {len(self.agent.safe_cells)}"
            rendered = self.font_small.render(safe_text, True, self.colors['text'])
            self.screen.blit(rendered, (self.screen_width - 190, reasoning_y))
            reasoning_y += 20
            
            # Show dangerous cells count
            danger_text = f"Dangerous Cells: {len(self.agent.dangerous_cells)}"
            rendered = self.font_small.render(danger_text, True, self.colors['text'])
            self.screen.blit(rendered, (self.screen_width - 190, reasoning_y))
    
    def draw_menu(self):
        # Dark background
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("WUMPUS WORLD", True, (255, 255, 255))
        subtitle = self.font_medium.render("Logic-Based Agent Implementation", True, (200, 200, 255))
        
        title_rect = title.get_rect(center=(self.screen_width//2, 150))
        subtitle_rect = subtitle.get_rect(center=(self.screen_width//2, 210))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, subtitle_rect)
        
        # Instructions
        instructions = [
            "Navigate through the cave to find the gold.",
            "Avoid pits and the deadly Wumpus!",
            "Use your arrow wisely to kill the Wumpus.",
            "Return to the start with the gold to win!",
            "",
            "Agent Logic:",
            "- Uses knowledge base to track safe/dangerous cells",
            "- Infers pit/Wumpus locations from percepts",
            "- Calculates probabilities for frontier cells",
            "- Prefers safe moves away from breezes"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_small.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.screen_width//2 - 250, 250 + i * 30))
        
        # Draw menu buttons
        for name in ['new_game', 'quit']:
            rect = self.buttons[name]
            color = self.colors['button_hover'] if rect.collidepoint(pygame.mouse.get_pos()) else self.colors['button']
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, (100, 100, 150), rect, 2, border_radius=5)
            
            label = name.replace('_', ' ').title()
            text = self.font_medium.render(label, True, self.colors['text'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_game_over(self, victory=False):
        # Dark background
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Result message
        if victory:
            title = self.font_large.render("VICTORY!", True, (100, 255, 100))
            message = f"You retrieved the gold with score: {self.world.score}"
        else:
            title = self.font_large.render("GAME OVER", True, (255, 100, 100))
            if self.agent_stuck:
                message = "Agent is stuck with no safe moves!"
            else:
                message = f"You died with score: {self.world.score}"
        
        title_rect = title.get_rect(center=(self.screen_width//2, 200))
        msg_text = self.font_medium.render(message, True, (255, 255, 255))
        msg_rect = msg_text.get_rect(center=(self.screen_width//2, 260))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(msg_text, msg_rect)
        
        # Draw menu buttons
        button_width, button_height = 200, 50
        button_y = 350
        
        new_game_rect = pygame.Rect(self.screen_width//2 - button_width//2, button_y, 
                                  button_width, button_height)
        quit_rect = pygame.Rect(self.screen_width//2 - button_width//2, button_y + 70, 
                               button_width, button_height)
        
        # New Game button
        new_game_color = self.colors['button_hover'] if new_game_rect.collidepoint(pygame.mouse.get_pos()) else self.colors['button']
        pygame.draw.rect(self.screen, new_game_color, new_game_rect, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 150), new_game_rect, 2, border_radius=5)
        new_game_text = self.font_medium.render("New Game", True, self.colors['text'])
        new_game_text_rect = new_game_text.get_rect(center=new_game_rect.center)
        self.screen.blit(new_game_text, new_game_text_rect)
        
        # Quit button
        quit_color = self.colors['button_hover'] if quit_rect.collidepoint(pygame.mouse.get_pos()) else self.colors['button']
        pygame.draw.rect(self.screen, quit_color, quit_rect, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 150), quit_rect, 2, border_radius=5)
        quit_text = self.font_medium.render("Quit", True, self.colors['text'])
        quit_text_rect = quit_text.get_rect(center=quit_rect.center)
        self.screen.blit(quit_text, quit_text_rect)
        
        # Update buttons dictionary for click handling
        self.buttons = {
            'new_game': new_game_rect,
            'quit': quit_rect
        }

    def draw(self):
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw game elements
        if self.game_state == GameState.MENU:
            self.draw_menu()
        elif self.game_state in [GameState.GAME_OVER, GameState.VICTORY]:
            self.draw_grid()
            self.draw_cells()
            self.draw_agent()
            self.particle_system.draw(self.screen)
            self.draw_ui()
            self.draw_game_over(self.game_state == GameState.VICTORY)
        else:
            self.draw_grid()
            self.draw_cells()
            self.draw_agent()
            self.particle_system.draw(self.screen)
            self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    game = WumpusVisualizer()
    game.run()