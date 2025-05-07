import pygame
from sys import exit
from random import randint, choice
import cv2
import threading
import time
from queue import Queue
from datetime import datetime
class ExpressionPlatformer:
    def __init__(self, screen):
        if not pygame.get_init():
            pygame.init()

        if not pygame.display.get_surface():
            pygame.display.set_mode((800, 600))  # Or whatever your game window size is

        self.screen = screen
        self.WIDTH, self.HEIGHT = 1000, 800  # New screen size
        self.clock = pygame.time.Clock()

        # Detection system variables
        self.detection_queue = Queue()
        self.smile_detected = False
        self.camera_enabled = False
        self.camera_frame = None  # Initialize camera frame

        # Game variables
        self.game_over = True
        self.start_time = 0
        self.score = 0
        self.score_list = []
        self.high_score = 0

        # Groups
        self.player = pygame.sprite.GroupSingle()
        self.player.add(self.Player(self))  # Pass the parent class instance to Player

        self.obstacle_group = pygame.sprite.Group()

        self.testFont = pygame.font.Font('Game4/Assets/font/Pixeltype.ttf', 50)
        self.bg_music = pygame.mixer.Sound('Game4/Assets/audio/music.wav')
        self.bg_music.set_volume(0.1)
        self.bg_music.play(loops=-1)

        # Scene
        self.skySurface = pygame.image.load('Game4/Assets/graphics/sky.png').convert()
        self.skySurface = pygame.transform.scale(self.skySurface, (self.WIDTH, self.HEIGHT))  # Scale sky to fit screen
        self.gndSurface = pygame.image.load('Game4/Assets/graphics/ground.png').convert()
        self.gndSurface = pygame.transform.scale(self.gndSurface, (self.WIDTH, 100))  # Scale ground to fit screen width

        # Introscreen
        self.player_stand = pygame.image.load('Game4/Assets/graphics/Player/player_stand.png').convert_alpha()
        self.player_stand = pygame.transform.rotozoom(self.player_stand, 0, 2)
        self.player_stand_rect = self.player_stand.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 100))  # Center player

        self.game_name = self.testFont.render("EXPRESSION PLATFORMER", False, (111, 196, 169))
        self.game_name_rect = self.game_name.get_rect(center=(self.WIDTH // 2, 60))  # Center game name

        self.start_game = self.testFont.render("Press Space To Start", False, (111, 196, 169))
        self.start_game_rect = self.start_game.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 100))  # Position start text

        self.instruction_text = self.testFont.render("Smile to Move!", False, (111, 196, 169))
        self.instruction_rect = self.instruction_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 150))  # Position instruction text

        # Timers
        self.obstacle_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.obstacle_timer, 1500)

    class Player(pygame.sprite.Sprite):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent  # Store the parent class instance
            player_walk_1 = pygame.image.load('Game4/Assets/graphics/Player/player_walk_1.png').convert_alpha()
            player_walk_2 = pygame.image.load('Game4/Assets/graphics/Player/player_walk_2.png').convert_alpha()
            self.player_walk = [player_walk_1, player_walk_2]
            self.player_jump = pygame.image.load('Game4/Assets/graphics/Player/jump.png').convert_alpha()
            self.player_index = 0
            self.image = self.player_walk[self.player_index]
            self.rect = self.image.get_rect(midbottom=(100, self.parent.HEIGHT - 100))  # Position player
            self.gravity = 0

            self.jump_sound = pygame.mixer.Sound('Game4/Assets/audio/jump.mp3')
            self.jump_sound.set_volume(0.2)

        def player_input(self):
            # Only allow movement when smile is detected
            if self.parent.smile_detected:  # Access smile_detected from the parent class
                key = pygame.key.get_pressed()
                if key[pygame.K_SPACE] and self.rect.bottom >= self.parent.HEIGHT - 100:
                    self.gravity = -20
                    self.jump_sound.play()

        def apply_gravity(self):
            self.gravity += 1
            self.rect.bottom += self.gravity
            if self.rect.bottom >= self.parent.HEIGHT - 100:
                self.rect.bottom = self.parent.HEIGHT - 100

        def animation_state(self):
            if self.rect.bottom < self.parent.HEIGHT - 100:
                self.image = self.player_jump
            else:
                self.player_index += 0.1
                if self.player_index >= len(self.player_walk):
                    self.player_index = 0
                self.image = self.player_walk[int(self.player_index)]

        def update(self):
            self.player_input()
            self.apply_gravity()
            self.animation_state()

    class Obstacles(pygame.sprite.Sprite):
        def __init__(self, parent, type):  # Accept parent instance
            super().__init__()
            self.parent = parent  # Store the parent instance

            if type == 'fly':
                fly_frame_1 = pygame.image.load('Game4/Assets/graphics/Fly/Fly1.png').convert_alpha()
                fly_frame_2 = pygame.image.load('Game4/Assets/graphics/Fly/Fly2.png').convert_alpha()
                self.frames = [fly_frame_1, fly_frame_2]
                self.y_pos = self.parent.HEIGHT - 300  # Use parent's HEIGHT
            else:
                snail_frame_1 = pygame.image.load('Game4/Assets/graphics/snail/snail1.png').convert_alpha()
                snail_frame_2 = pygame.image.load('Game4/Assets/graphics/snail/snail2.png').convert_alpha()
                self.frames = [snail_frame_1, snail_frame_2]
                self.y_pos = self.parent.HEIGHT - 100  # Use parent's HEIGHT

            self.animation_index = 0
            self.image = self.frames[self.animation_index]
            self.rect = self.image.get_rect(midbottom=(randint(self.parent.WIDTH, self.parent.WIDTH + 200), self.y_pos))

        def animation_state(self):
            self.animation_index += 0.1
            if self.animation_index >= len(self.frames):
                self.animation_index = 0
            self.image = self.frames[int(self.animation_index)]

        def destroy(self):
            if self.rect.x < -100:
                self.kill()

        def update(self):
            self.animation_state()
            self.rect.left -= 6
            self.destroy()

    def disp_score(self):
        current_time = int(pygame.time.get_ticks() / 1000) - self.start_time
        score_surface = self.testFont.render(f'{current_time}', False, (64, 64, 64))
        score_rect = score_surface.get_rect(center=(self.WIDTH // 2, 50))  # Center score
        self.screen.blit(score_surface, score_rect)
        return current_time

    def collisions(self):
        if pygame.sprite.spritecollide(self.player.sprite, self.obstacle_group, False):
            self.obstacle_group.empty()
            return True
        else:
            return False

    def draw_detection_status(self):
        # Draw expression detection status
        smile_text = self.testFont.render(f"Smile: {'Yes' if self.smile_detected else 'No'}", False,
                                          (0, 200, 0) if self.smile_detected else (200, 0, 0))
        self.screen.blit(smile_text, (20, 90))

        # Draw combined status
        if self.smile_detected:
            ready_text = self.testFont.render("READY TO MOVE!", False, (0, 200, 0))
        else:
            ready_text = self.testFont.render("CANNOT MOVE", False, (200, 0, 0))
        self.screen.blit(ready_text, (self.WIDTH - 400, 110))  # Adjust position

    def setup_smile_detection(self):
        try:
            # Load face and smile cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            # Initialize camera
            

            

            # Change codec if necessary (Try 'MJPG' or 'H264' if 'XVID' doesn't work)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec for compatibility
            out = cv2.VideoWriter('game_recording.avi', fourcc, 20.0, (640, 480))  # Change format to .avi for compatibility

            def smile_detection_thread():
                try:
                    # Initialize camera
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Error: Could not open camera.")
                        return

                    # Set recording resolution to match Pygame display (200x150)
                    rec_width, rec_height = 200, 150
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Works well for small resolutions
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"face_recording.avi"
                    out = cv2.VideoWriter(filename, fourcc, 20.0, (rec_width, rec_height))

                    if not out.isOpened():
                        print("Error: Could not initialize video writer.")
                        return

                    print(f"Recording started: {filename}")

                    while self.camera_enabled:
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame.")
                            break

                        # Convert to grayscale for detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        is_smiling = False

                        # Draw detection rectangles on the frame
                        processed_frame = frame.copy()
                        for (x, y, w, h) in faces:
                            roi_gray = gray[y:y + h, x:x + w]
                            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                            
                            if len(smiles) > 0:
                                is_smiling = True
                                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                for (sx, sy, sw, sh) in smiles:
                                    cv2.rectangle(processed_frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)

                        self.detection_queue.put(("smile", is_smiling))

                        # Resize to match in-game display (200x150)
                        resized_frame = cv2.resize(processed_frame, (rec_width, rec_height))
                        
                        # Convert to RGB for Pygame display
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        pygame_frame = pygame.surfarray.make_surface(rgb_frame)
                        self.camera_frame = pygame_frame

                        # Write the resized frame to the video file
                        out.write(resized_frame)

                        time.sleep(0.05)  # Prevent CPU overload

                    print("Recording stopped.")
                    out.release()
                    cap.release()

                except Exception as e:
                    print(f"Recording error: {e}")
                    if 'out' in locals() and out.isOpened():
                        out.release()
                    if 'cap' in locals() and cap.isOpened():
                        cap.release()                          # Ensure video is saved
                    cv2.destroyAllWindows()

            # Start smile detection in a separate thread
            self.camera_enabled = True
            thread = threading.Thread(target=smile_detection_thread)
            thread.daemon = True
            thread.start()

            return True

        except Exception as e:
            print(f"Error setting up smile detection: {e}")
            self.camera_enabled = False
            return False
    
    def run(self):
        # Setup detection systems
        print("Setting up smile detection...")
        self.camera_enabled = self.setup_smile_detection()

        while True:
            # Process detection results
            while not self.detection_queue.empty():
                detection_type, is_detected = self.detection_queue.get()
                if detection_type == "smile":
                    self.smile_detected = is_detected

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Clean up detection threads
                    self.camera_enabled = False
                    # Close any opencv windows
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        # Clean up detection threads
                        self.camera_enabled = False
                        # Close any opencv windows
                        cv2.destroyAllWindows()
                        pygame.quit()
                        exit()

                if self.game_over:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        self.game_over = False
                        self.start_time = int(pygame.time.get_ticks() / 1000)

                if not self.game_over:
                    if event.type == self.obstacle_timer:
                        self.obstacle_group.add(self.Obstacles(self, choice(['fly', 'snail', 'snail', 'snail'])))  # Pass self as parent

            if not self.game_over:
                self.screen.blit(self.skySurface, (0, 0))
                self.screen.blit(self.gndSurface, (0, self.HEIGHT - 100))  # Position ground

                # Score calculation
                self.score = self.disp_score()
                self.score_list.append(self.score)
                self.high_score = max(self.score_list)

                # Display detection status
                self.draw_detection_status()

                # Display game elements
                self.player.draw(self.screen)
                self.player.update()

                self.obstacle_group.draw(self.screen)
                self.obstacle_group.update()

                # Display the camera frame in the bottom-right corner
                if self.camera_frame is not None:
                    self.screen.blit(self.camera_frame, (self.WIDTH - 210, self.HEIGHT - 160))  # Adjust position as needed

                # Check for game over
                self.game_over = self.collisions()
                if self.game_over:
                    self.camera_enabled = False 
                    return True  # Signal game over

            else:  # Game over screen
                self.screen.fill((94, 129, 162))
                self.screen.blit(self.game_name, self.game_name_rect)
                self.screen.blit(self.player_stand, self.player_stand_rect)
                self.screen.blit(self.instruction_text, self.instruction_rect)

                score_message = self.testFont.render(f"Your Score : {self.score}", False, (111, 196, 169))
                
                score_message_rect = score_message.get_rect(center=(400, 340))
                high_score_text = self.testFont.render(f"HI: {self.high_score}", False, (111, 196, 169))
                high_score_text_rect = high_score_text.get_rect(midleft=(650, 60))

                if self.score == 0:
                    self.screen.blit(self.start_game, self.start_game_rect)
                else:
                    score_msg = self.testFont.render(f"Score: {self.score}", False, (111, 196, 169))
                    self.screen.blit(score_msg, (self.WIDTH // 2 - 100, self.HEIGHT // 2 + 100))
                    high_score = self.testFont.render(f"High: {self.high_score}", False, (111, 196, 169))
                    self.screen.blit(high_score, (self.WIDTH // 2 + 100, self.HEIGHT // 2 + 100))
                # Still display detection status on game over screen
                self.draw_detection_status()

            pygame.display.update()
            self.clock.tick(60)
            score_message = self.testFont.render(f"Your Score: {self.score}", False, (111, 196, 169))
            score_message_rect = score_message.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 100))
            self.screen.blit(score_message, score_message_rect)
