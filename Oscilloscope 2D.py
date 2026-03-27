#!/usr/bin/env python3
"""
Oscilloscope Music Viewer — 2D XY Edition
Supports: WAV only

Dependencies:
    pip3 install pygame numpy soundfile imageio imageio-ffmpeg

Controls:
    SPACE       play / pause
    LEFT/RIGHT  skip 5 seconds
    R           start render to MP4
    ESC         quit
"""

import sys
import time
import threading
from pathlib import Path

import pygame
import numpy as np
import soundfile as sf

# ─── imageio (optional — only needed for render) ─────────────────────────────
try:
    import imageio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False


# ─── Constants ───────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1280, 720
FPS            = 60
SUPPORTED = {".wav", ".mp3"}

C_BG           = (6,   8,  12)
C_GRID         = (15,  30,  20)
C_WAVE_CORE    = (0,  255, 130)
C_WAVE_GLOW    = (0,  180,  80)
C_SCANLINE     = (0,   15,   8)
C_TEXT         = (180, 255, 200)
C_TEXT_DIM     = (80,  140,  90)
C_BAR_BG       = (20,  35,  25)
C_BAR_FG       = (0,  210, 110)
C_BTN          = (18,  38,  24)
C_BTN_HOV      = (30,  65,  40)
C_BTN_BDR      = (0,  130,  60)
C_SCRUBBER     = (220, 255, 230)
C_RENDER_BTN   = (30,  20,  50)
C_RENDER_HOV   = (55,  35,  90)
C_RENDER_BDR   = (120,  60, 200)
C_RENDER_ACT   = (80,  40, 140)

OSC_X          = 60
OSC_Y          = 80
OSC_W          = WIN_W - 120
OSC_H          = 480
BAR_Y          = WIN_H - 110
BAR_X          = 80
BAR_W          = WIN_W - 160
BAR_H          = 10
BTN_W, BTN_H   = 90, 34
RND_W, RND_H   = 110, 34


# ─── Helpers ─────────────────────────────────────────────────────────────────
def find_audio(directory):
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in SUPPORTED and p.is_file():
            return p
    return None


def load_samples(path):
    ext = path.suffix.lower()

    if ext == ".wav":
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)

    elif ext == ".mp3":
        import subprocess

        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-f", "f32le",        # raw float32 output
            "-acodec", "pcm_f32le",
             "-ac", "2",           # force stereo
            "-ar", "44100",       # sample rate
            "-"
    ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout.read()
        proc.wait()

        data = np.frombuffer(raw, dtype=np.float32)
        data = data.reshape(-1, 2)
        sr = 44100

    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Ensure at least stereo
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)

    channels = data.shape[1]
    print(f"Loaded {channels}-channel audio at {sr} Hz")
    return data, sr, channels


def fmt_time(sec):
    sec = max(0.0, sec)
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def draw_rounded_rect(surf, color, rect, radius=6, border=0, border_color=None):
    if color:
        pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def make_scanline_overlay(w, h):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 3):
        pygame.draw.line(surf, (*C_SCANLINE, 80), (0, y), (w, y))
    return surf


# ─── Core waveform drawing (used by both live view and offline render) ────────
def render_frame(surf, samples, sr, pos_secs, scanlines,
                 fnt_title, fnt_ui, fnt_small,
                 title, duration, has_blanking, fmt_label,
                 show_ui=True):
    """Draw one complete frame onto surf. Returns nothing."""

    surf.fill(C_BG)

    # ── Oscilloscope panel ────────────────────────────────────────────────────
    panel = pygame.Rect(OSC_X, OSC_Y, OSC_W, OSC_H)
    draw_rounded_rect(surf, (8, 14, 10), panel, 4, 1, (25, 65, 40))

    # Grid
    for i in range(1, 12):
        x = OSC_X + (OSC_W * i) // 12
        pygame.draw.line(surf, C_GRID, (x, OSC_Y), (x, OSC_Y + OSC_H))
    for j in range(1, 6):
        y = OSC_Y + (OSC_H * j) // 6
        pygame.draw.line(surf, C_GRID, (OSC_X, y), (OSC_X + OSC_W, y))
    cy_grid = OSC_Y + OSC_H // 2
    cx_grid = OSC_X + OSC_W // 2
    pygame.draw.line(surf, (25, 55, 35), (OSC_X, cy_grid), (OSC_X + OSC_W, cy_grid))
    pygame.draw.line(surf, (25, 55, 35), (cx_grid, OSC_Y), (cx_grid, OSC_Y + OSC_H))

    # Waveform
    n_disp   = int(0.02 * sr)
    center_i = int(pos_secs * sr)
    start_i  = max(0, center_i - n_disp // 2)
    chunk    = samples[start_i:start_i + n_disp]

    if len(chunk) >= 2:
        cx = OSC_X + OSC_W // 2
        cy = OSC_Y + OSC_H // 2
        scale    = OSC_H * 0.45
        glow_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)

        for i in range(len(chunk) - 1):
            s1 = chunk[i]
            s2 = chunk[i + 1]
            p1 = (int(cx + s1[0] * scale), int(cy - s1[1] * scale))
            p2 = (int(cx + s2[0] * scale), int(cy - s2[1] * scale))
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if dist > 40:
                intensity = 0.01
            else:
                intensity = max(0.05, 1.0 - (dist / 40.0))

            alpha = int(255 * intensity)
            thick = 1 if dist > 15 else 2

            if alpha > 5:
                pygame.draw.line(glow_surf, (*C_WAVE_GLOW, int(80 * intensity)), p1, p2, thick + 3)
                pygame.draw.line(surf, (*C_WAVE_CORE, alpha), p1, p2, thick)

        surf.blit(glow_surf, (0, 0))

    # ── UI elements ───────────────────────────────────────────────────────────
    if show_ui:
        # Progress bar
        bar = pygame.Rect(BAR_X, BAR_Y, BAR_W, BAR_H)
        draw_rounded_rect(surf, C_BAR_BG, bar, radius=5)
        if duration > 0:
            frac   = max(0.0, min(1.0, pos_secs / duration))
            fill_w = int(frac * BAR_W)
            if fill_w > 0:
                draw_rounded_rect(surf, C_BAR_FG,
                                  pygame.Rect(BAR_X, BAR_Y, fill_w, BAR_H), radius=5)
            dot_x = max(BAR_X, min(BAR_X + BAR_W, BAR_X + int(frac * BAR_W)))
            pygame.draw.circle(surf, C_SCRUBBER, (dot_x, BAR_Y + BAR_H // 2), 7)
            pygame.draw.circle(surf, C_WAVE_CORE, (dot_x, BAR_Y + BAR_H // 2), 5)
        draw_rounded_rect(surf, None, bar, radius=5, border=1, border_color=(30, 70, 45))

        tl = fnt_small.render(fmt_time(pos_secs), True, C_TEXT_DIM)
        tr = fnt_small.render(fmt_time(duration),  True, C_TEXT_DIM)
        surf.blit(tl, (BAR_X, BAR_Y + 16))
        surf.blit(tr, (BAR_X + BAR_W - tr.get_width(), BAR_Y + 16))

        # Title
        ts = fnt_title.render(title, True, C_TEXT)
        surf.blit(ts, ts.get_rect(centerx=WIN_W // 2, top=22))

        badge_s = fnt_small.render(fmt_label, True, C_BG)
        badge_r = badge_s.get_rect()
        badge_r.topleft = (WIN_W // 2 + ts.get_width() // 2 + 10, 26)
        pygame.draw.rect(surf, C_WAVE_CORE, badge_r.inflate(10, 6), border_radius=3)
        surf.blit(badge_s, badge_r)

        # Status
        if has_blanking:
            status = fnt_small.render("BLANKING ACTIVE (3-channel)", True, (0, 255, 120))
        else:
            status = fnt_small.render("Stereo only - no blanking", True, C_TEXT_DIM)
        surf.blit(status, (OSC_X, OSC_Y + OSC_H + 15))

        hint = fnt_small.render("SPACE = play/pause   ← → = skip 5s   R = render MP4", True, C_TEXT_DIM)
        surf.blit(hint, (12, WIN_H - 18))

    # Scanlines always on
    surf.blit(scanlines, (0, 0))


# ─── Offline render ───────────────────────────────────────────────────────────
def offline_render(samples, sr, duration, has_blanking, title,
                   audio_path, out_path, scanlines,
                   fnt_title, fnt_ui, fnt_small,
                   progress_cb=None):
    """
    Render every frame offline and write MP4 with audio.
    progress_cb(fraction) called periodically.
    """
    total_frames = int(duration * FPS) + 1
    surf = pygame.Surface((WIN_W, WIN_H))

    writer = imageio.get_writer(
        str(out_path),
        fps=FPS,
        codec="libx264",
        quality=8,
        audio_path=str(audio_path),   # imageio-ffmpeg muxes audio automatically
        audio_codec="aac",
        macro_block_size=None,
        ffmpeg_log_level="quiet",
    )

    for frame_idx in range(total_frames):
        pos = frame_idx / FPS
        render_frame(surf, samples, sr, pos, scanlines,
             fnt_title, fnt_ui, fnt_small,
             title, duration, has_blanking,
             audio_path.suffix.upper().replace(".", ""),
             show_ui=True)

        # pygame surface → numpy RGB array
        raw = pygame.surfarray.array3d(surf)          # shape (W, H, 3)
        raw = np.transpose(raw, (1, 0, 2))            # → (H, W, 3)
        writer.append_data(raw)

        if progress_cb and frame_idx % 30 == 0:
            progress_cb(frame_idx / total_frames)

    writer.close()
    if progress_cb:
        progress_cb(1.0)

class ErrorScreen:
    def __init__(self, screen, clock, message):
        self.screen = screen
        self.clock = clock
        self.message = message
        mono = pygame.font.match_font("couriernew,liberationmono,dejavusansmono,monospace") or ""
        self.fnt_error = pygame.font.Font(mono, 30)
        self.scanlines = make_scanline_overlay(WIN_W, WIN_H)

    def run(self):
        # A simple static screen that waits for ESC or Close
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            self.screen.fill(C_BG)
            
            # Error message in C_TEXT (dimmer green) or a custom red if you prefer
            txt = self.fnt_error.render(self.message, True, (200, 80, 80)) # Subtle red-alert
            rect = txt.get_rect(center=(WIN_W // 2, WIN_H // 2))
            
            # Border box for the error
            pygame.draw.rect(self.screen, (60, 20, 20), rect.inflate(40, 40), border_radius=5)
            pygame.draw.rect(self.screen, (200, 60, 60), rect.inflate(40, 40), width=1, border_radius=5)
            
            self.screen.blit(txt, rect)
            self.screen.blit(self.scanlines, (0, 0))
            
            pygame.display.flip()
            self.clock.tick(FPS)

class WarningScreen:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.duration = 5.0
        self.start_time = time.time()
        
        mono = pygame.font.match_font("couriernew,liberationmono,dejavusansmono,monospace") or ""
        self.fnt_big = pygame.font.Font(mono, 40)
        self.fnt_warn = pygame.font.Font(mono, 80)
        
        self.scanlines = make_scanline_overlay(WIN_W, WIN_H)

    def draw_warning_symbol(self, y_center, alpha):
        points = [
            (WIN_W // 2, y_center - 45),      # Top
            (WIN_W // 2 - 55, y_center + 45), # Bottom Left
            (WIN_W // 2 + 55, y_center + 45)  # Bottom Right
        ]
        
        # Temporary surface to handle the Alpha fade
        temp_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        
        # Using C_WAVE_CORE for that "glow" look
        pygame.draw.polygon(temp_surf, (*C_WAVE_CORE, alpha), points, 3)
        
        excl = self.fnt_warn.render("!", True, C_WAVE_CORE)
        excl.set_alpha(alpha)
        excl_rect = excl.get_rect(center=(WIN_W // 2, y_center + 5))
        
        temp_surf.blit(excl, excl_rect)
        self.screen.blit(temp_surf, (0, 0))

    def run(self):
        while True:
            elapsed = time.time() - self.start_time
            if elapsed > self.duration:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    return # Skip if user interacts

            # Calculate alpha (0 to 255) based on a sine wave over 5 seconds
            alpha = int(255 * np.sin((elapsed / self.duration) * np.pi))
            
            # Use your background color
            self.screen.fill(C_BG)

            # Draw "!" symbols
            self.draw_warning_symbol(WIN_H // 2 - 160, alpha)
            self.draw_warning_symbol(WIN_H // 2 + 160, alpha)

            # Draw text
            text = self.fnt_big.render("EPILEPSY WARNING", True, C_TEXT)
            text.set_alpha(alpha)
            text_rect = text.get_rect(center=(WIN_W // 2, WIN_H // 2))
            
            # Add a slight "glow" effect by blitting twice
            self.screen.blit(text, text_rect)
            
            # Apply the scanline overlay to match the main viewer
            self.screen.blit(self.scanlines, (0, 0))

            pygame.display.flip()
            self.clock.tick(FPS)
            
# ─── Main viewer ─────────────────────────────────────────────────────────────
class OscilloscopeViewer:
    # 1. Ensure these are defined at the very top of the class
    IDLE       = "idle"
    RENDERING  = "rendering"
    DONE       = "done"
    ERROR      = "error"

    def __init__(self, audio_path):
        # ... (other init code like pygame.init())
        pygame.init()
        # 2. Reference them correctly here
        self.render_state    = OscilloscopeViewer.IDLE 
        self.render_progress = 0.0
        self.render_out_path = None
        self.render_thread   = None
        self.render_error    = ""
        
        # ... (rest of your init)
      
        # Increase buffer for MacOS 10.14 compatibility to prevent crackling
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()
        self.path   = audio_path
        self.title  = audio_path.stem
        self.format_label = audio_path.suffix.upper().replace(".", "")

        pygame.display.set_caption("Oscilloscope -- " + self.title)

        print("Loading: " + audio_path.name + " ...")
        self.samples, self.sr, self.channels = load_samples(audio_path)
        self.duration  = len(self.samples) / self.sr

        # --- MODIFIED START STATE ---
        self.paused    = True          # Start paused
        self.pause_pos = 0.0
        self.play_t0   = 0.0           # This will be set when we actually play
        self.finished  = False

        pygame.mixer.music.load(str(audio_path))
        self.play_t0 = time.perf_counter()

        mono = pygame.font.match_font("couriernew,liberationmono,dejavusansmono,monospace") or ""
        self.fnt_title = pygame.font.Font(mono, 22)
        self.fnt_ui    = pygame.font.Font(mono, 15)
        self.fnt_small = pygame.font.Font(mono, 13)

        self.scanlines    = make_scanline_overlay(WIN_W, WIN_H)
        self.dragging     = False
        self.btn_rect     = pygame.Rect(0, 0, BTN_W, BTN_H)
        self.bar_rect     = pygame.Rect(BAR_X, BAR_Y, BAR_W, BAR_H)
        self.rnd_btn_rect = pygame.Rect(0, 0, RND_W, RND_H)
        self.has_blanking = (self.channels >= 3)

        # Render state
        self.render_state    = self.IDLE
        self.render_progress = 0.0
        self.render_out_path = None
        self.render_thread   = None
        self.render_error    = ""

    # ── Playback ──────────────────────────────────────────────────────────────
    def elapsed(self):
        if self.paused or self.finished:
            return self.pause_pos
        return self.pause_pos + (time.perf_counter() - self.play_t0)

    def seek(self, seconds):
        seconds = max(0.0, min(seconds, self.duration))
        self.pause_pos = seconds
        self.finished  = False
        pygame.mixer.music.play(start=seconds)
        self.play_t0 = time.perf_counter()
        if self.paused:
            pygame.mixer.music.pause()

    def toggle_pause(self):
        if self.finished:
            self.seek(0.0)
            self.paused = False
            return
            
        if self.paused:
            # Check if music is actually playing/loaded in the mixer
            if not pygame.mixer.music.get_busy():
                # Start for the first time from the current pause_pos
                pygame.mixer.music.play(start=self.pause_pos)
            else:
                pygame.mixer.music.unpause()
                
            self.play_t0 = time.perf_counter()
            self.paused  = False
        else:
            self.pause_pos = self.elapsed()
            pygame.mixer.music.pause()
            self.paused = True

    # ── Render trigger ────────────────────────────────────────────────────────
    def start_render(self):
        if not IMAGEIO_OK:
            self.render_state = self.ERROR
            self.render_error = "imageio not installed — pip3 install imageio imageio-ffmpeg"
            return
        if self.render_state == self.RENDERING:
            return

        out_path = self.path.parent / (self.title + ".mp4")
        self.render_out_path  = out_path
        self.render_state     = self.RENDERING
        self.render_progress  = 0.0

        def _run():
            try:
                offline_render(
                    self.samples, self.sr, self.duration,
                    self.has_blanking, self.title,
                    self.path, out_path,
                    self.scanlines,
                    self.fnt_title, self.fnt_ui, self.fnt_small,
                    progress_cb=lambda f: setattr(self, "render_progress", f),
                )
                self.render_state = self.DONE
            except Exception as e:
                self.render_error = str(e)
                self.render_state = self.ERROR
                print("Render error:", e)

        self.render_thread = threading.Thread(target=_run, daemon=True)
        self.render_thread.start()

    # ── Draw UI ───────────────────────────────────────────────────────────────
    def draw_pause_button(self):
        bx  = WIN_W // 2 - BTN_W // 2
        by  = WIN_H - 55
        btn = pygame.Rect(bx, by, BTN_W, BTN_H)
        bg  = C_BTN_HOV if btn.collidepoint(pygame.mouse.get_pos()) else C_BTN
        draw_rounded_rect(self.screen, bg, btn, radius=6, border=1, border_color=C_BTN_BDR)
        label = "PLAY" if (self.paused or self.finished) else "PAUSE"
        ls = self.fnt_ui.render(label, True, C_TEXT)
        self.screen.blit(ls, ls.get_rect(center=btn.center))
        return btn

    def draw_render_button(self):
        bx  = WIN_W - RND_W - 20
        by  = WIN_H - 55
        btn = pygame.Rect(bx, by, RND_W, RND_H)

        if self.render_state == self.RENDERING:
            bg  = C_RENDER_ACT
            bdr = C_RENDER_BDR
            pct = int(self.render_progress * 100)
            label = f"  {pct}%..."
            # Progress fill inside button
            fill_w = int(self.render_progress * RND_W)
            pygame.draw.rect(self.screen, (50, 30, 90), btn, border_radius=6)
            if fill_w > 0:
                fill_r = pygame.Rect(bx, by, fill_w, RND_H)
                pygame.draw.rect(self.screen, (100, 60, 180), fill_r, border_radius=6)
        elif self.render_state == self.DONE:
            bg  = (10, 40, 20)
            bdr = (0, 200, 80)
            label = "SAVED!"
        elif self.render_state == self.ERROR:
            bg  = (40, 10, 10)
            bdr = (200, 60, 60)
            label = "ERROR"
        else:
            hovered = btn.collidepoint(pygame.mouse.get_pos())
            bg  = C_RENDER_HOV if hovered else C_RENDER_BTN
            bdr = C_RENDER_BDR
            label = "RENDER MP4"

        if self.render_state != self.RENDERING:
            draw_rounded_rect(self.screen, bg, btn, radius=6, border=1, border_color=bdr)

        ls = self.fnt_ui.render(label, True, C_TEXT)
        self.screen.blit(ls, ls.get_rect(center=btn.center))

        # Show output path under button when done
        if self.render_state == self.DONE and self.render_out_path:
            path_s = self.fnt_small.render(
                "Saved: " + self.render_out_path.name, True, (0, 200, 80))
            self.screen.blit(path_s, (bx, by + RND_H + 4))

        if self.render_state == self.ERROR:
            err_s = self.fnt_small.render(self.render_error[:60], True, (220, 80, 80))
            self.screen.blit(err_s, (20, WIN_H - 18))

        return btn

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        while True:
            pos = self.elapsed()

            if pos >= self.duration > 0 and not self.paused:
                self.pause_pos = self.duration
                self.finished  = True
                self.paused    = True

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.mixer.music.stop()
                    pygame.quit()
                    sys.exit()

                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_SPACE, pygame.K_p):
                        self.toggle_pause()
                    elif ev.key == pygame.K_ESCAPE:
                        pygame.mixer.music.stop()
                        pygame.quit()
                        sys.exit()
                    elif ev.key == pygame.K_LEFT:
                        self.seek(max(0, pos - 5))
                    elif ev.key == pygame.K_RIGHT:
                        self.seek(min(self.duration, pos + 5))
                    elif ev.key == pygame.K_r:
                        self.start_render()

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.bar_rect.inflate(0, 24).collidepoint(ev.pos):
                        self.dragging = True
                        frac = (ev.pos[0] - BAR_X) / BAR_W
                        self.seek(max(0.0, min(1.0, frac)) * self.duration)
                    elif self.btn_rect.collidepoint(ev.pos):
                        self.toggle_pause()
                    elif self.rnd_btn_rect.collidepoint(ev.pos):
                        self.start_render()

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = False

                elif ev.type == pygame.MOUSEMOTION and self.dragging:
                    frac = (ev.pos[0] - BAR_X) / BAR_W
                    self.seek(max(0.0, min(1.0, frac)) * self.duration)

            # Draw frame using shared render_frame function
            render_frame(self.screen, self.samples, self.sr, pos,
             self.scanlines, self.fnt_title, self.fnt_ui, self.fnt_small,
             self.title, self.duration, self.has_blanking,
             self.format_label,
             show_ui=True)

            # Draw interactive buttons on top
            self.btn_rect     = self.draw_pause_button()
            self.rnd_btn_rect = self.draw_render_button()

            # FPS
            fps_s = self.fnt_small.render(
                f"{int(self.clock.get_fps())} fps", True, C_TEXT_DIM)
            self.screen.blit(fps_s, (WIN_W - fps_s.get_width() - 12, 10))

            pygame.display.flip()
            self.clock.tick(FPS)


# ─── Entry point ─────────────────────────────────────────────────────────────
import shutil  # Add this to your imports at the top!

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    audio = find_audio(script_dir)

    # Initialize Pygame basics
    pygame.init()
    main_screen = pygame.display.set_mode((WIN_W, WIN_H))
    main_clock = pygame.time.Clock()

    # --- ERROR CHECK 1: File Existence ---
    if audio is None:
        err = ErrorScreen(main_screen, main_clock, "ERROR: No audio file (.wav/.mp3) found.")
        err.run()
        pygame.quit()
        sys.exit()

    # --- ERROR CHECK 2: FFmpeg for MP3s ---
    if audio.suffix.lower() == ".mp3" and not shutil.which("ffmpeg"):
        err = ErrorScreen(main_screen, main_clock, "ERROR: FFmpeg not found. Required for MP3 playback.")
        err.run()
        pygame.quit()
        sys.exit()

    # If all checks pass, proceed
    warn = WarningScreen(main_screen, main_clock)
    warn.run()

    # Start the viewer
    OscilloscopeViewer(audio).run()
