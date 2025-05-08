import pygame
import sys
import math
import random
import csv
import time

class RealmOfFear:
    def __init__(self, screen):
        self.score = 0

        self.screen = screen
        self.WIDTH, self.HEIGHT = screen.get_width(), screen.get_height()
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.CYAN = (0, 255, 255)

        # Load images and adjust sizes
        try:
            self.player_img = pygame.image.load(r"Game1/Assets/rohan.png").convert_alpha()
            self.enemy_img = pygame.image.load(r"Game1/Assets/rghosty.png").convert_alpha()
            self.trap_img = pygame.image.load(r"Game1/Assets/bomb.png").convert_alpha()
            self.background_img = pygame.image.load(r"Game1/Assets/fearbackdrop.jpeg").convert()
        except Exception as e:
            print("Error loading game assets:", e)
            sys.exit()
        self.player_img = pygame.transform.scale(self.player_img, (100, 100))
        self.enemy_img = pygame.transform.scale(self.enemy_img, (220, 180))
        self.trap_img = pygame.transform.scale(self.trap_img, (60, 80))
        self.background_img = pygame.transform.scale(self.background_img, (self.WIDTH, self.HEIGHT))
        self.power_up_imgs = {
            'speed_boost': pygame.transform.scale(pygame.image.load(r"Game1\Assets\speed.png").convert_alpha(), (40, 40)),
            'shield':      pygame.transform.scale(pygame.image.load(r"Game1\Assets\shield.png").convert_alpha(), (40, 40)),
            'trap_disabler': pygame.transform.scale(pygame.image.load(r"Game1\Assets\key.png").convert_alpha(), (40, 40)),
            'enemy_freeze':  pygame.transform.scale(pygame.image.load(r"Game1\Assets\freeze.png").convert_alpha(), (40, 40))
        }

        
        # Load sounds
        pygame.mixer.init()
        self.background_music1      = pygame.mixer.Sound(r"Game1\Sounds\fear.mp3") #changeee kiya h
        self.background_music      = pygame.mixer.Sound(r"Game1\Sounds\fear_intro-trimmed.mp3")  #changeee kiya h
        self.background_music1.set_volume(0.5) #changeee kiya h
        self.background_music.set_volume(0.5) #changeee kiya h
        self.background_music.play(loops=-1)  #changeee kiya h
        self.power_up_sound        = pygame.mixer.Sound(r"Game1\Sounds\powerup.wav")
        self.trap_collision_sound  = pygame.mixer.Sound(r"Game1\Sounds\trap_sound.wav")
        #self.game_over_sound       = pygame.mixer.Sound(r"Game1\Sounds\gameover.wav") #changeee kiya h

        # Cube & game props
        self.cube_size = 40
        self.player_x  = random.randint(0, self.WIDTH - self.cube_size)
        self.player_y  = random.randint(0, self.HEIGHT - self.cube_size)
        self.enemy_x   = random.randint(0, self.WIDTH - self.cube_size)
        self.enemy_y   = random.randint(0, self.HEIGHT - self.cube_size)
        self.enemy_speed = 3

        # Traps
        self.num_traps = 5
        self.traps = []
        for _ in range(self.num_traps):
            self.traps.append([
                random.randint(0, self.WIDTH - self.cube_size),
                random.randint(0, self.HEIGHT - self.cube_size),
                random.choice([-2, 2]), random.choice([-2, 2])
            ])

        # Power‑ups
        self.power_ups = []
        self.power_up_types = ['speed_boost', 'shield', 'trap_disabler', 'enemy_freeze']
        self.power_up_duration = 300

        # Safe‑area tracking
        self.safe_distance = 150    # minimum distance to be considered safe
        self.safe_time = 0.0        # cumulative safe‑zone time

        # Control flags
        self.game_over = False
        self.paused    = False
        self.clock     = pygame.time.Clock()

        # Power‑up states
        self.player_shield      = False
        self.player_speed_boost = False
        self.trap_disabled      = False
        self.enemy_frozen       = False
        self.power_up_timer     = 0

        # Input‑freq tracking
        self.last_input_time       = time.time()
        self.input_count           = 0
        self.input_window          = 1.0
        self.current_input_frequency = 0.0

        # CSV logging
        self.csv_file   = "game_data.csv"
        self.start_time = time.time()

    def start_logging(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Time', 'Player_X', 'Player_Y', 'Enemy_X', 'Enemy_Y', 'Distance_to_Nearest_Enemy',
                'Time_in_Safe_Areas', 'Player_Speed', 'Player_Direction', 'Input_Frequency', 'Events'
            ])
    
    def show_intro_image(self):
        self.background_music.stop() #changeee kiya h
        try:
            jumpscare_img = pygame.image.load(r"Game1/Assets/jumpscare.png").convert()
            jumpscare_sound = pygame.mixer.Sound(r"Game1/Sounds/jumpscare.wav")
        except Exception as e:
            print("Error loading jumpscare assets:", e)
            return

        jumpscare_img = pygame.transform.scale(jumpscare_img, (self.WIDTH, self.HEIGHT))
        jumpscare_sound.play()

        fade_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        fade_surface.fill((0, 0, 0))
        for alpha in range(255, -1, -10):
            self.screen.blit(jumpscare_img, (0, 0))
            fade_surface.set_alpha(alpha)
            self.screen.blit(fade_surface, (0, 0))
            pygame.display.update()
            pygame.time.delay(20)

        pygame.time.delay(1000)
        for alpha in range(0, 256, 10):
            self.screen.blit(jumpscare_img, (0, 0))
            fade_surface.set_alpha(alpha)
            self.screen.blit(fade_surface, (0, 0))
            pygame.display.update()
            pygame.time.delay(20)
            
            
    def log_data(self, events_list):
        elapsed_time = time.time() - self.start_time
        direction = math.degrees(math.atan2(self.player_y - self.enemy_y, self.player_x - self.enemy_x))
        distance  = math.hypot(self.player_x - self.enemy_x, self.player_y - self.enemy_y)
        time_in_safe = self.safe_time
        input_frequency = self.current_input_frequency

        ev_str = ';'.join(events_list) if events_list else 'No events'

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                f"{elapsed_time:.2f}", self.player_x, self.player_y,
                self.enemy_x, self.enemy_y, f"{distance:.2f}",
                f"{time_in_safe:.2f}",
                10 if self.player_speed_boost else 7,
                f"{direction:.1f}", f"{input_frequency:.2f}",
                ev_str
            ])

    def reset_game(self):
        # re‑initialize everything (including safe_time)
        self.__init__(self.screen)

    def run(self):
        self.start_logging()
        self.background_music1.play(loops=-1)

        # track last timestamp for dt
        last_time = time.time()

        while True:
            events_list = []
            self.screen.blit(self.background_img, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_q:
                            pygame.quit(); sys.exit()
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused

            if not self.game_over and not self.paused:
                # --- Input Frequency ---
                keys = pygame.key.get_pressed()
                if any((keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_UP], keys[pygame.K_DOWN])):
                    self.input_count += 1
                now_input = time.time()
                if now_input - self.last_input_time >= self.input_window:
                    self.current_input_frequency = self.input_count / self.input_window
                    self.input_count = 0
                    self.last_input_time = now_input

                # --- Movement ---
                speed = 10 if self.player_speed_boost else 7
                if keys[pygame.K_LEFT]  and self.player_x > 0:
                    self.player_x -= speed
                if keys[pygame.K_RIGHT] and self.player_x < self.WIDTH - self.cube_size:
                    self.player_x += speed
                if keys[pygame.K_UP]    and self.player_y > 0:
                    self.player_y -= speed
                if keys[pygame.K_DOWN]  and self.player_y < self.HEIGHT - self.cube_size:
                    self.player_y += speed

                # Enemy chases
                self.enemy_x, self.enemy_y = self.move_towards_player(
                    self.player_x, self.player_y, self.enemy_x, self.enemy_y, self.enemy_speed)

                # Move traps
                if not self.trap_disabled:
                    self.move_traps()

                # Spawn power‑ups
                self.spawn_power_ups()

                # --- Safe‑time tracking ---
                now = time.time()
                dt = now - last_time
                last_time = now
                distance = math.hypot(self.player_x - self.enemy_x, self.player_y - self.enemy_y)
                if distance > self.safe_distance:
                    self.safe_time += dt

                # --- Collision & Event Collection ---
                player_rect = pygame.Rect(self.player_x, self.player_y, self.cube_size, self.cube_size)
                enemy_rect  = pygame.Rect(self.enemy_x, self.enemy_y, self.cube_size, self.cube_size)

                # Enemy collision
                if player_rect.colliderect(enemy_rect):
                    events_list.append('Player collided with enemy')
                    if not self.player_shield:
                        self.game_over = True
                        self.game_over_sound.play()
                        self.log_data(events_list)
                        return True

                # Trap collision
                for trap in self.traps:
                    trap_rect = pygame.Rect(trap[0], trap[1], self.cube_size, self.cube_size)
                    if player_rect.colliderect(trap_rect):
                        events_list.append('Player hit a trap')
                        if not self.player_shield and not self.trap_disabled:
                            self.game_over = True
                            self.trap_collision_sound.play()
                            self.log_data(events_list)
                            return True

                # Power‑up pickup
                for pu in self.power_ups[:]:
                    pu_rect = pygame.Rect(pu[0], pu[1], 30, 30)
                    if player_rect.colliderect(pu_rect):
                        events_list.append(f"Picked {pu[3]}")
                        self.power_ups.remove(pu)
                        self.apply_power_up(pu[3])

                # --- Draw & Update ---
                self.update_power_ups()
                self.draw_player(self.player_x, self.player_y)
                self.draw_enemy(self.enemy_x, self.enemy_y)
                self.draw_traps()
                self.draw_power_ups()
                self.draw_power_up_timer()

                # Finally log this frame’s data
                self.log_data(events_list)

            pygame.display.flip()
            self.clock.tick(60)

    # Drawing helpers
    def draw_player(self, x, y):
        self.screen.blit(self.player_img, (x, y))
    def draw_enemy(self, x, y):
        self.screen.blit(self.enemy_img, (x, y))
    def draw_traps(self):
        for t in self.traps:
            self.screen.blit(self.trap_img, (t[0], t[1]))
    def draw_power_ups(self):
        for pu in self.power_ups:
            self.screen.blit(self.power_up_imgs[pu[3]], (pu[0], pu[1]))
    def draw_power_up_timer(self):
        if self.power_up_timer > 0:
            font = pygame.font.SysFont('Arial', 24)
            text = font.render(f"Power-up Time: {self.power_up_timer // 60}", True, self.CYAN)
            self.screen.blit(text, (self.WIDTH - 200, 10))

    # Game mechanics
    def move_towards_player(self, px, py, ex, ey, s):
        dx, dy = px - ex, py - ey
        dist = math.hypot(dx, dy)
        if dist:
            dx, dy = dx/dist, dy/dist
        return ex + dx*s, ey + dy*s

    def move_traps(self):
        for tr in self.traps:
            tr[0] += tr[2]; tr[1] += tr[3]
            if tr[0] <= 0 or tr[0] >= self.WIDTH - self.cube_size: tr[2] = -tr[2]
            if tr[1] <= 0 or tr[1] >= self.HEIGHT - self.cube_size: tr[3] = -tr[3]

    def spawn_power_ups(self):
        if random.random() < 0.01:
            x, y = random.randint(0, self.WIDTH-40), random.randint(0, self.HEIGHT-40)
            pt = random.choice(self.power_up_types)
            self.power_ups.append([x, y, 0, pt])

    def apply_power_up(self, pu):
        setattr(self, {
            'speed_boost': 'player_speed_boost',
            'shield': 'player_shield',
            'trap_disabler': 'trap_disabled',
            'enemy_freeze': 'enemy_frozen'
        }[pu], True)
        self.power_up_timer = self.power_up_duration
        self.power_up_sound.play()

    def update_power_ups(self):
        if self.power_up_timer > 0:
            self.power_up_timer -= 1
        else:
            self.player_speed_boost = self.player_shield = False
            self.trap_disabled = self.enemy_frozen = False

    def check_collision(self, r1, r2):
        return r1.colliderect(r2)  # collision helper
    def run(self):
        self.show_intro_image() 
        self.background_music1.play(loops=-1)  # Play background music #changeee kiya h
        while True:
            self.screen.blit(self.background_img, (0, 0))  # Draw background

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_r:  # Restart the game
                            self.reset_game()
                        if event.key == pygame.K_q:  # Quit the game
                            pygame.quit()
                            sys.exit()
                    if event.key == pygame.K_p:  # Pause the game
                        self.paused = not self.paused

            if not self.game_over and not self.paused:
                # Player movement
                keys = pygame.key.get_pressed()
                player_speed = 10 if self.player_speed_boost else 7
                if keys[pygame.K_LEFT] and self.player_x > 0:
                    self.player_x -= player_speed
                if keys[pygame.K_RIGHT] and self.player_x < self.WIDTH - self.cube_size:
                    self.player_x += player_speed
                if keys[pygame.K_UP] and self.player_y > 0:
                    self.player_y -= player_speed
                if keys[pygame.K_DOWN] and self.player_y < self.HEIGHT - self.cube_size:
                    self.player_y += player_speed

                # Enemy movement (bot chases player)
                self.enemy_x, self.enemy_y = self.move_towards_player(self.player_x, self.player_y, self.enemy_x, self.enemy_y, self.enemy_speed)

                # Move traps
                if not self.trap_disabled:
                    self.move_traps()

                # Spawn power-ups
                self.spawn_power_ups()

                # Check for collision with power-ups
                player_rect = pygame.Rect(self.player_x, self.player_y, self.cube_size, self.cube_size)
                for power_up in self.power_ups[:]:
                    power_up_rect = pygame.Rect(power_up[0], power_up[1], 30, 30)
                    if self.check_collision(player_rect, power_up_rect):
                        self.power_ups.remove(power_up)
                        self.apply_power_up(power_up[3])

                # Update power-up states
                self.update_power_ups()

                # Draw everything
                self.draw_player(self.player_x, self.player_y)
                self.draw_enemy(self.enemy_x, self.enemy_y)
                self.draw_traps()
                self.draw_power_ups()
                self.draw_power_up_timer()

                # Check for collision between player and enemy
                if not self.player_shield:
                    enemy_rect = pygame.Rect(self.enemy_x, self.enemy_y, self.cube_size, self.cube_size)
                    if self.check_collision(player_rect, enemy_rect):
                        self.game_over = True
                        self.background_music1.stop() #changeee kiya h
                        #self.game_over_sound.play()   #changeee kiya h
                        return True  # Signal game over

                # Check collision with traps
                if not self.player_shield and not self.trap_disabled:
                    for trap in self.traps:
                        trap_rect = pygame.Rect(trap[0], trap[1], self.cube_size, self.cube_size)
                        if self.check_collision(player_rect, trap_rect):
                            self.background_music1.stop() #changeee kiya h
                            self.game_over = True
                            self.trap_collision_sound.play()
                            return True  # Signal game over

                # Increase score over time
                self.score += 1

            elif self.paused:
                # Draw pause menu
                font = pygame.font.SysFont(None, 55)
                text = font.render("Paused", True, self.WHITE)
                self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))
            else:
                # Game over screen
                font = pygame.font.SysFont(None, 55)
                text = font.render(f"Game Over! Score: {self.score}", True, self.WHITE)
                self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))
                text2 = font.render("Press R to Restart or Q to Quit", True, self.WHITE)
                self.screen.blit(text2, (self.WIDTH // 2 - text2.get_width() // 2, self.HEIGHT // 2 + 50))

            # Update the display
            pygame.display.update()
            self.clock.tick(30)

# --- Entry Point ---
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Realm of Fear")
    game = RealmOfFear(screen)
    game.show_intro_image()
    game.run()
