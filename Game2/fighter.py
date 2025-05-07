import pygame
import speech_recognition as sr
import random
import pyaudio
import numpy as np
import json
import time
import os


from vosk import Model, KaldiRecognizer
from fuzzywuzzy import process
class Fighter():
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps,sound, is_bot=False):
        self.player = player
        self.size = data[0]
        self.image_scale = data[1]
        self.offset = data[2]
        self.flip = flip
        self.animation_list = self.load_images(sprite_sheet, animation_steps)
        self.action = 0  # 0:idle, 1:run, 2:jump, 3:attack1, 4:attack2, 5:hit, 6:death
        self.frame_index = 0
        self.image = self.animation_list[self.action][self.frame_index]
        self.update_time = pygame.time.get_ticks()
        self.rect = pygame.Rect((x, y, 80, 180))
        self.vel_y = 0
        self.running = False
        self.jump = False
        self.attacking = False
        self.attack_type = 0
        self.attack_cooldown = 0
        self.attack_sound = sound
        self.hit = False
        self.health = 100
        self.alive = True
        self.is_bot = is_bot  # Set bot flag
        self.bot_attack_multiplier = 3
        self.bot_action_timer = 0
        self.voice_log = []  # Store all entries here
        self.log_file = "voice_log.json"
        
        # Voice recognition setup for player 1
        if self.player == 1:
            self.model = Model("vosk-model-small-en-us-0.15")  # Load offline model
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.mic = pyaudio.PyAudio()
            self.stream = self.mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                        input=True, frames_per_buffer=4096)
            self.stream.start_stream()
            self.voice_command = None    # Stores the last recognized command
    
    
       
    def log_voice_event(self, word, intensity, confidence):
        entry = {
            "timestamp": time.time(),
            "word": word,
            "intensity_dB": intensity,
            "confidence": confidence
        }
        self.voice_log.append(entry)

        # Write to JSON file (overwrite or append based on your preference)
        with open(self.log_file, "w") as f:
            json.dump(self.voice_log, f, indent=2)
            

    def load_images(self, sprite_sheet, animation_steps):
        animation_list = []
        for y, animation in enumerate(animation_steps):
            temp_img_list = []
            for x in range(animation):
                temp_img = sprite_sheet.subsurface(x * self.size, y * self.size, self.size, self.size)
                temp_img_list.append(pygame.transform.scale(temp_img, (self.size * self.image_scale, self.size * self.image_scale)))
            animation_list.append(temp_img_list)
        return animation_list

    def get_intensity(self, audio_block):
        """Compute the intensity (in dB) from the audio data."""
        # Convert bytes to int16 numpy array
        audio_np = np.frombuffer(audio_block, dtype=np.int16)
        
        if len(audio_np) == 0:
            return -float('inf')  # Handle silence or zero input

        rms = np.sqrt(np.mean(np.square(audio_np)))  # Root Mean Square
        db = 20 * np.log10(rms + 1e-6)  # Convert to decibels (avoid log(0))
        return db

   
    def listen_for_commands(self):
        """Listens for voice commands using Vosk and updates movement accordingly."""
        data = self.stream.read(4096, exception_on_overflow=False)
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())  # Get JSON output
            print("Full Recognition Result:", result)  # Debugging: Print the full result
            command = result.get("text", "").lower()
            print("Recognized Text:", command)  # Debugging: Print the recognized text

            intensity = self.get_intensity(data)  # Get intensity of the audio block
            print(f"Audio Intensity: {intensity:.2f} dB")  # Print intensity for the detected audio block
            
            # Define valid commands
            valid_commands = ["left", "right", "jump", "attack"]

            # Step 1: Check for exact matches first
            for valid_command in valid_commands:
                if valid_command in command:
                    print(f"Recognized Command (Exact Match): {valid_command}| Intensity: {intensity:.2f} dB")
                    self.voice_command = valid_command
                    return

            # Step 2: If no exact match, use fuzzy matching
            best_match, match_confidence = process.extractOne(command, valid_commands)
            print(f"Best Match: {best_match}, Match Confidence: {match_confidence}")  # Debugging

            # Set a confidence threshold for fuzzy matching (e.g., 70%)
            if match_confidence >= 40:
                print(f"Recognized Command (Fuzzy Match): {best_match} | Intensity: {intensity:.2f} dB (Confidence: {match_confidence})")
                self.voice_command = best_match
                self.log_voice_event(best_match, intensity, match_confidence) 
                return

            # Step 3: Fallback to keyword spotting
            for word in command.split():
                if word in valid_commands:
                    print(f"Recognized Command (Keyword Spotting): {word}")
                    self.voice_command = word
                    self.log_voice_event(word, intensity, 100) 
                    return
                
            # If no valid command is detected
            print("No valid command detected.")
            self.voice_command = None  # Reset if no valid command is found  # Reset if no valid command is found
            

    def move(self, screen_width, screen_height, surface, target, round_over):
        SPEED = 10
        BOT_SPEED = 4  # Reduced bot speed
        GRAVITY = 2
        dx = 0
        dy = 0
        self.running = False
        self.attack_type = 0

        if self.is_bot:
            # Bot logic (move toward player slowly)
            self.bot_ai(target)
            

        elif self.player == 1:  # Voice-controlled player
            self.listen_for_commands()
            if self.voice_command == "left":
                dx = -SPEED
                self.running = True
            elif self.voice_command == "right":
                dx = SPEED
                self.running = True
            elif self.voice_command == "jump" and not self.jump:
                self.vel_y = -30
                self.jump = True
            elif self.voice_command == "attack":
                self.attack(target)
                self.attack_type = 1

        # Apply gravity
        self.vel_y += GRAVITY
        dy += self.vel_y

        # Ensure player stays on screen
        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > screen_width:
            dx = screen_width - self.rect.right
        if self.rect.bottom + dy > screen_height - 110:
            self.vel_y = 0
            self.jump = False
            dy = screen_height - 110 - self.rect.bottom

        # Ensure players face each other
        # Ensure players face each other properly
        if target.rect.centerx > self.rect.centerx and not self.attacking:
            self.flip = False
        elif target.rect.centerx < self.rect.centerx and not self.attacking:
            self.flip = True


        # Apply attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Update player position
        self.rect.x += dx
        self.rect.y += dy
    def bot_ai(self, target):
        """ Basic AI logic for bot movement and attacking """
        ATTACK_DISTANCE = 100 
        if self.attacking == False and self.alive and target.alive:
            # Move towards the player
            if abs(target.rect.centerx - self.rect.centerx) > ATTACK_DISTANCE:
                if target.rect.centerx > self.rect.centerx:
                    self.rect.x += 5  # Move right
                    self.running = True
                elif target.rect.centerx < self.rect.centerx:
                    self.rect.x -= 5  # Move left
                    self.running = True

            # Randomly jump (20% chance every frame)
            if random.randint(0, 100) < 2 and not self.jump:
                self.vel_y = -30
                self.jump = True

            if abs(target.rect.centerx - self.rect.centerx) < 120 and self.attack_cooldown == 0:
                self.attack(target)
                self.attack_type = random.choice([1, 2])  # Randomly choose attack type

    def update(self):
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self.update_action(6)  # Death
        elif self.hit:
            self.update_action(5)  # Hit
        elif self.attacking:
            if self.attack_type == 1:
                self.update_action(3)  # Attack 1
            elif self.attack_type == 2:
                self.update_action(4)  # Attack 2
        elif self.jump:
            self.update_action(2)  # Jump
        elif self.running:
            self.update_action(1)  # Run
        else:
            self.update_action(0)  # Idle

        animation_cooldown = 50
        self.image = self.animation_list[self.action][self.frame_index]
        if pygame.time.get_ticks() - self.update_time > animation_cooldown:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
        if self.frame_index >= len(self.animation_list[self.action]):
            if not self.alive:
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:
                self.frame_index = 0
                if self.action in [3, 4]:  # Attack actions
                    self.attacking = False
                    self.attack_cooldown = 20
                if self.action == 5:  # Hit action
                    self.hit = False
                    self.attacking = False
                    self.attack_cooldown = 20
    
    def attack(self, target):
        if self.attack_cooldown == 0:
            self.attacking = True
            self.attack_sound.play()
            attacking_rect = pygame.Rect(self.rect.centerx - (2 * self.rect.width * self.flip), self.rect.y, 2 * self.rect.width, self.rect.height)
            if attacking_rect.colliderect(target.rect):
                damage = 5  # Base damage
                if self.is_bot:  # If the attacker is a bot, apply the multiplier
                    damage *= self.bot_attack_multiplier
                target.health -= damage
                target.hit = True

    def update_action(self, new_action):
        if new_action != self.action:
            self.action = new_action
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self, surface):
        img = pygame.transform.flip(self.image, self.flip, False)
        surface.blit(img, (self.rect.x - (self.offset[0] * self.image_scale), self.rect.y - (self.offset[1] * self.image_scale)))