
import pygame
import os
import subprocess
import sys

# Import your mini-games
from Game1.fear import RealmOfFear
from Game2.main import BrawlerGame
from Game3.mysterydoor import RealmOfPortals
from Game4.smile import ExpressionPlatformer
from Game1.analysis import run_analysis
from Game3.analysis import run_behavior_analysis
from graph import VoiceLogAnalyzer
from final_report import display_report_from_json

# Initialize Pygame
pygame.init()

# Print current working directory (to help with asset path issues)
print("Current working directory:", os.getcwd())

# Screen settings
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EmoQuest")

# Define states
INTRO = "intro"
GAME1 = "game1"
GAME2 = "game2"
GAME3 = "game3"
GAME4 = "game4"
END = "end"

class GameManager:
    def __init__(self):
        self.state = INTRO
        self.games = {}
        self.screen = screen
        self.WIDTH, self.HEIGHT = screen.get_width(), screen.get_height()
        
        self.game_order = [GAME1,GAME4,GAME2,GAME3]
        self.current_game_index = 0
        self.intro_images = {
            GAME1: "Game1/Assets/bg1fear.jpeg",
            GAME2: "Game2/Assets/images/background/ragebg.jpeg",
            GAME3: "Game3/Assets/surprisebg.jpeg",
            GAME4: "Game4/Assets/graphics/happybg.jpeg"
        }
        self.bg_music = pygame.mixer.Sound('part8_good_intro.ogg') #changeee kiya h
        self.bg_music.set_volume(0.5) #changeee kiya h

    def initialize_game(self, game_name):
        if game_name not in self.games:
            if game_name == GAME1:
                self.games[GAME1] = RealmOfFear(self.screen)
            elif game_name == GAME2:
                self.games[GAME2] = BrawlerGame(self.screen)
            elif game_name == GAME3:
                self.games[GAME3] = RealmOfPortals(self.screen)
            elif game_name == GAME4:
                self.games[GAME4] = ExpressionPlatformer(self.screen)
            print(f"Initialized {game_name}")

    def show_intro_image(self, image_path, fade_time=4000):
        try:
            
            intro_image = pygame.image.load(image_path).convert_alpha()
            intro_scaled = pygame.transform.scale(intro_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            print(f"Error loading intro image '{image_path}': {e}")
            self.screen.fill((0, 0, 0))
            pygame.display.flip()
            self.wait_for_space()
            return False

        # Fade in effect
        for alpha in range(0, 256, 5):
            self.screen.fill((0, 0, 0))
            intro_scaled.set_alpha(alpha)
            self.screen.blit(intro_scaled, (0, 0))
            pygame.display.flip()
            pygame.time.delay(max(fade_time // 51, 20))  # Smooth fade

        self.screen.blit(intro_scaled, (0, 0))
        pygame.display.flip()
        print(f"Intro image '{image_path}' displayed. Waiting for SPACE...")
        self.wait_for_space()
        return True

    def wait_for_space(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.bg_music.stop() #changeee kiya h
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.bg_music.stop() #changeee kiya h
                    waiting = False

    def run_intro(self):
        self.bg_music.play(loops=-1) #changeee kiya h
        print("Showing main intro screen...")
        self.show_intro_image("mainbg.jpeg")
        self.state = self.game_order[0]
        return True

    def run(self):
        if not self.run_intro():
            self.bg_music.stop() #changeee kiya h
            return

        running = True
        while running:
            print(f"Current State: {self.state}")

            if self.state in self.game_order:
                self.initialize_game(self.state)
                if self.state in self.intro_images:
                    self.show_intro_image(self.intro_images[self.state])
                if self.state == GAME1:
                    try:
                        jumpscare_img = pygame.image.load(r"Game1/Assets/jumpscare.jpeg").convert()
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


                game_over = self.games[self.state].run()

                if game_over:
                    self.current_game_index += 1
                    if self.current_game_index < len(self.game_order):
                        self.state = self.game_order[self.current_game_index]
                    else:
                        self.state = END
                        

            if self.state == END:

                       
                try:
                    # Load and scale the end image to fit the screen
                    end_image = pygame.image.load('endbg.jpeg').convert()
                    end_image = pygame.transform.scale(end_image, (self.WIDTH, self.HEIGHT))
                    
                    # Create a surface for fade effect
                    fade_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
                    fade_surface.fill((0, 0, 0))
                    
                    # Fade in effect
                    for alpha in range(255, -1, -5):  # From opaque to transparent
                        self.screen.blit(end_image, (0, 0))
                        fade_surface.set_alpha(alpha)
                        self.screen.blit(fade_surface, (0, 0))
                        pygame.display.flip()
                        pygame.time.delay(30)
                    
                    # Display the image for 3 seconds
                    start_time = pygame.time.get_ticks()
                    while pygame.time.get_ticks() - start_time < 3000:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                        self.screen.blit(end_image, (0, 0))
                        pygame.display.flip()
                        pygame.time.delay(30)
                    
                    print("All games completed. Running analysis...")
                    run_analysis()
                    run_behavior_analysis()
                    analyzer = VoiceLogAnalyzer('voice_log.json')
                    analyzer.plot()

                    try:
                        script_path = os.path.join('Game4', 'testdata.py')
                        subprocess.run(['python', script_path], check=True)
                        print("testdata.py script finished.")
                    except Exception as e:
                        print(f"Error running testdata.py: {e}")

                    running = False

                except Exception as e:
                    print(f"Error loading end image: {e}")
                    running = False

        pygame.quit()
        display_report_from_json()
               

# Run the game manager
if __name__ == "__main__":
    manager = GameManager()
    manager.run()

    

