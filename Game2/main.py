import pygame
from pygame import mixer
from Game2.fighter import Fighter

class BrawlerGame:
    def __init__(self, screen):
        self.screen = screen
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = screen.get_width(), screen.get_height()
        self.clock = pygame.time.Clock()
        self.FPS = 20

        # Define colours
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)

        # Define game variables
        self.intro_count = 3
        self.last_count_update = pygame.time.get_ticks()
        self.score = [0, 0]  # Player scores [P1, P2]
        self.round_over = False
        self.ROUND_OVER_COOLDOWN = 2000

        # Define fighter variables
        self.WARRIOR_SIZE = 162
        self.WARRIOR_SCALE = 4
        self.WARRIOR_OFFSET = [72, 56]
        self.WARRIOR_DATA = [self.WARRIOR_SIZE, self.WARRIOR_SCALE, self.WARRIOR_OFFSET]
        self.WIZARD_SIZE = 250
        self.WIZARD_SCALE = 3
        self.WIZARD_OFFSET = [112, 107]
        self.WIZARD_DATA = [self.WIZARD_SIZE, self.WIZARD_SCALE, self.WIZARD_OFFSET]

        # Load music and sounds
        mixer.init()
        pygame.mixer.music.load(r"Game2/Assets/audio/music.mp3")
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1, 0.0, 5000)
        self.sword_fx = pygame.mixer.Sound(r"Game2/Assets/audio/sword.wav")
        self.sword_fx.set_volume(0.5)
        self.magic_fx = pygame.mixer.Sound(r"Game2/Assets/audio/magic.wav")
        self.magic_fx.set_volume(0.75)

        # Load background image
        self.bg_image = pygame.image.load(r"Game2/Assets/images/background/background.jpg").convert_alpha()

        # Load spritesheets
        self.warrior_sheet = pygame.image.load(r"Game2/Assets/images/warrior/Sprites/warrior.png").convert_alpha()
        self.wizard_sheet = pygame.image.load(r"Game2/Assets/images/wizard/Sprites/wizard.png").convert_alpha()

        # Load victory image
        self.victory_img = pygame.image.load(r"Game2/Assets/images/icons/victory.png").convert_alpha()
        self.lost_img=pygame.image.load(r"Game2/Assets/images/icons/lost.png").convert_alpha()
        # Define number of steps in each animation
        self.WARRIOR_ANIMATION_STEPS = [10, 8, 1, 7, 7, 3, 7]
        self.WIZARD_ANIMATION_STEPS = [8, 8, 1, 8, 8, 3, 7]

        # Define font
        self.count_font = pygame.font.Font(r"Game2/Assets/fonts/turok.ttf", 80)
        self.score_font = pygame.font.Font(r"Game2/Assets/fonts/turok.ttf", 30)

        # Create two instances of fighters
        self.fighter_1 = Fighter(1, 200, 310, False, self.WARRIOR_DATA, self.warrior_sheet, self.WARRIOR_ANIMATION_STEPS, self.sword_fx)
        self.fighter_2 = Fighter(2, 700, 310, True, self.WIZARD_DATA, self.wizard_sheet, self.WIZARD_ANIMATION_STEPS, self.magic_fx, is_bot=True)

    def draw_text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))

    def draw_bg(self):
        scaled_bg = pygame.transform.scale(self.bg_image, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(scaled_bg, (0, 0))

    def draw_health_bar(self, health, x, y):
        ratio = health / 100
        pygame.draw.rect(self.screen, self.WHITE, (x - 2, y - 2, 404, 34))
        pygame.draw.rect(self.screen, self.RED, (x, y, 400, 30))
        pygame.draw.rect(self.screen, self.YELLOW, (x, y, 400 * ratio, 30))

    def run(self):
        run = True
        while run:
            self.clock.tick(self.FPS)

            # Draw background
            self.draw_bg()

            # Show player stats
            self.draw_health_bar(self.fighter_1.health, 20, 20)
            self.draw_health_bar(self.fighter_2.health, 580, 20)
            self.draw_text("P1: " + str(self.score[0]), self.score_font, self.RED, 20, 60)
            self.draw_text("P2: " + str(self.score[1]), self.score_font, self.RED, 580, 60)

            # Update countdown
            if self.intro_count <= 0:
                # Move fighters
                self.fighter_1.move(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.screen, self.fighter_2, self.round_over)
                self.fighter_2.move(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.screen, self.fighter_1, self.round_over)
            else:
                # Display count timer
                self.draw_text(str(self.intro_count), self.count_font, self.RED, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 3)
                # Update count timer
                if (pygame.time.get_ticks() - self.last_count_update) >= 1000:
                    self.intro_count -= 1
                    self.last_count_update = pygame.time.get_ticks()

            # Update fighters
            self.fighter_1.update()
            self.fighter_2.update()

            # Draw fighters
            self.fighter_1.draw(self.screen)
            self.fighter_2.draw(self.screen)

            # Check for player defeat
            if not self.round_over:
                if not self.fighter_1.alive or not self.fighter_2.alive:
                    self.round_over = True
                    if not self.fighter_2.alive:
                      self.screen.blit(self.victory_img, (500, 400))
                    else:
                      self.screen.blit(self.lost_img, (500, 400))
                    pygame.display.update()
                    pygame.time.delay(3000)  # Show victory screen for 3 seconds
                    run = False
                    return True
            # Event handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Update display
            pygame.display.update()

        # Exit pygame
        pygame.quit()

# Initialize Pygame
