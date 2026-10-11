# Samples

Everything here was made with this repository on one NVIDIA DGX Spark on 2026-10-10. Nothing was retouched. Video and audio were
re-encoded once to keep the repo small (H.264 CRF 22–24, AAC 160k; MP3 192k), and all file metadata was stripped.

## Video (MiniMax H3 · Fast HD 1344x768 · TensorFold engine · picture and sound in one pass)

### `video/lion-roar-fasthd-10s.mp4`
- **Idea given to the director:** "A majestic male lion walks through tall golden savanna grass at sunrise, stops on a rock and lets out a mighty roar that echoes across the plains"
- **The director wrote** (Qwen3.8-27B, 8 s):
  - *Picture:* A massive male lion with a thick, dark mane stands amidst waist-high, golden savanna grass. The low-angle sunrise casts long, warm amber shadows, illuminating dust motes in the air. A shallow depth of field blurs the distant horizon into a soft, hazy orange glow, keeping the focus sharp on the lion's intense, amber eyes and textured fur.
  - *Action:* The lion strides slowly forward through the swaying grass, his heavy paws sinking slightly into the earth. He stops abruptly atop a flat, sun-baked rock, turning his head to face the camera directly. His chest expands as he opens his jaws wide, holding the pose as the camera pushes in slightly to emphasize his scale and power.
  - *Sound:* A low, rumbling bass note builds in the background music. The ambient sound includes the dry rustle of grass and distant birdsong. As the lion roars, a deep, thunderous vocalization erupts, echoing sharply across the open plains, followed by a sudden, crisp silence and a single, echoing bird call.
- **Settings:** Fast HD, 10 s, seed 2026, director review on. **Render:** 7 min 34 s including the review. **Director's review:** 8/10.

### `video/paper-boat-film-fasthd-33s.mp4`: a film of three segments joined seamlessly
- **Idea:** "A small paper boat sails down a rain-filled city gutter, tumbles over a tiny waterfall into a storm drain channel, and glides out into a sunlit park pond where ducks swim past"
- **The director wrote:**
  - *Picture:* A crumpled white paper boat drifts through a rain-slicked city gutter. Dark asphalt reflects neon signs. The lens is a macro 50mm, low angle. Lighting is moody, cool blue with warm amber streetlights. Raindrops create ripples. The boat is the sole focus, sharp against the blurred urban background.
  - *Action:* The camera tracks laterally with the boat as it moves smoothly. Suddenly, it hits a slight dip, tumbles forward over a small concrete lip, and plunges into a darker, deeper channel. The motion is fluid, capturing the splash and the boat stabilizing in the dark water.
  - *Sound:* Heavy rain patter, rhythmic dripping, and the soft splash of water. A gentle, melancholic piano melody plays. As the boat tumbles, a whooshing sound accompanies the drop. No dialogue.
  - *Beats, one per segment:* Drifting through rain-slicked gutter / Tumbling over tiny waterfall into drain / Gliding into sunlit park pond
- **Settings:** Fast HD, 30 s requested, which becomes 3 segments (15 + 15 + 5 s) with 22-frame guides carrying picture and sound across each seam. Seed 2026, review with re-take. **Render:** 39 min.
- **Review:** segments 1 and 2 scored 8/10. Segment 3's first take scored 8/10; its automatic re-take scored only 3/10, so the studio kept the first take. As a result the film stays in the rain and does not reach the sunlit pond of the third beat. The director judges against the whole prompt, so it is strict about a beat that changes the light.

## Images (1344x768, 2 per prompt, the better one kept)

| file | model | prompt |
|---|---|---|
| `images/marquee-qwen-image-2.1-turbo.jpg` | Qwen-Image 2.1 Turbo (8 steps) | A vintage cinema marquee at night glowing with warm light bulbs, the sign reads "LION AI STUDIOS" and below it "NOW SHOWING", rain-slick street reflecting the lights |
| `images/lighthouse-z-image-turbo.jpg` | Z-Image-Turbo | A lone lighthouse on a rocky coast under a dramatic stormy sunset, huge waves crashing on the rocks, spray catching the last orange light, photoreal, 35mm |
| `images/fisherman-krea2-realism.jpg` | Krea 2 Turbo + realism LoRA | Portrait of an elderly fisherman mending a net on a wooden dock at golden hour, weathered hands, knitted cap, shallow depth of field, natural light |
| `images/reading-nook-flux2-klein.jpg` | FLUX.2 klein 4B | A cozy reading nook by a rainy window, a ginger cat asleep on a stack of old books, a cup of tea steaming, soft morning light |

The people in these images are generated and fictional.

## Music

| file | engine | input |
|---|---|---|
| `music/run-with-the-lions-ace-step.mp3` | **Fast** · ACE-Step 1.5 Turbo, 45 s, 112 BPM, G major | Style: "uplifting indie pop, bright electric guitars, warm synth pads, punchy drums, female vocal, anthemic chorus, modern polished production". Original lyrics below; Whisper hears them sung almost word for word. |
| `music/trailer-score-minimax-music3.mp3` | **Best** · MiniMax Music 3, 30 s asked, 72 BPM, D minor | A structured brief: cinematic orchestral trailer score, opening quiet and building to a heroic peak with low strings, timpani, French horns, full strings, brass and choir pads; no vocals. |
| `music/lions-dawn-midi-instruments.mp3` | **MIDI** · `lions-dawn.mid` | Instruments chosen per part: melody → Violin, chords → String Ensemble 1, bass → Acoustic Bass, drums → Orchestra Kit. Rendered in about 2 s. |
| `music/lions-dawn-midi-ai-restyle.mp3` | **MIDI + AI restyle** (ACE-Step, strength 0.55) | The same render, restyled as "cinematic orchestral, soaring violin lead, lush strings, warm double bass, orchestral percussion, epic and hopeful". About 11 s. |

`music/lions-dawn.mid` is an original 16-bar piece in A minor at 96 BPM (melody, chords, bass, drums), written for this
repository and released under CC0. Use it to try the MIDI engine.

Lyrics of *Run with the Lions* (original, written for this sample):

```
[Verse]
Streetlights hum a quiet tune
Painting gold on every room
We were dreaming wide awake
Every step a chance to take

[Chorus]
Run with the lions, light up the night
Hearts like thunder, burning bright
We don't wait for the morning sun
We are the ones, we are the ones
```
