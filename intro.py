import pygame
import os
import subprocess
# Import your mini-games
from Game1.fear import RealmOfFear
from Game2.main import BrawlerGame
from Game3.mysterydoor import RealmOfPortals
from Game4.smile import ExpressionPlatformer
from Game1.analysis import run_analysis
from Game3.analysis import run_behavior_analysis
from graph import VoiceLogAnalyzer
# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mini Game Collection")

# Define states
INTRO = "intro"
GAME1 = "game1"
GAME2 = "game2"
GAME3 = "game3"
GAME4 = "game4"
END = "end"



class GameManager:
    def __init__(self):
        self.state = INTRO  # Start with the intro screen
        self.games = {}  # Don't initialize games here
        
        self.game_order = [GAME4, GAME1, GAME2, GAME3]
        self.current_game_index = 0

    def initialize_game(self, game_name):
        """Create game instance only when needed."""
        if game_name not in self.games:
            if game_name == GAME1:
                self.games[GAME1] = RealmOfFear(screen)
            elif game_name == GAME2:
                self.games[GAME2] = BrawlerGame(screen)
            elif game_name==GAME3:
                self.games[GAME3]=RealmOfPortals(screen)
            elif game_name==GAME4:
                self.games[GAME4]=ExpressionPlatformer(screen)
            # Add other games here
            print(f"Initialized {game_name}")  # Debugging
    
    def run_intro(self):
        """Display intro screen and wait for user input to start."""
        running = True
        while running:
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 50)
            text = font.render("Welcome! Press SPACE to start.", True, (255, 255, 255))
            screen.blit(text, (200, 250))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    print("Starting game sequence")  # Debugging
                    running = False
                    self.state = self.game_order[0]  # Move to the first game
        return True
    
    def run(self):
        """Main game loop"""
        if not self.run_intro():
            return
        
        running = True
        while running:
            print("Current State:", self.state)  # Debugging

            if self.state in self.game_order:
                self.initialize_game(self.state)  # Create game instance only when needed
                game_over = self.games[self.state].run()
                
                if game_over:  # Move to the next game
                    self.current_game_index += 1
                    if self.current_game_index < len(self.game_order):
                        self.state = self.game_order[self.current_game_index]
                    else:
                        self.state = END  # End screen after last game
            
            if self.state == END:
                
                print("All games completed. Running analysis.")
                run_analysis()
                run_behavior_analysis()
                analyzer = VoiceLogAnalyzer('voice_log.json')
                analyzer.plot()
                try:
                    script_path = os.path.join('Game4', 'testdata.py')
                    self.emotion_process = subprocess.Popen(['python', script_path])
                    self.emotion_process.wait()  # Wait for the script to finish
                    print("testdata.py script finished.")
                except Exception as e:
                    print(f"Error running testdata.py: {e}")
                running = False
                 # Exit after finishing all games
        
        pygame.quit()

# Run the game manager
if __name__ == "__main__":
    manager = GameManager()
    manager.run()