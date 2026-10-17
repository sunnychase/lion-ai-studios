# Samples

Everything here was made with this repository on one NVIDIA DGX Spark (2026-10-10 and 2026-10-16). No image was retouched. Video and audio were
re-encoded once to keep the repo small (H.264 CRF 22–24, or 1280x720 at 1.5–1.8 Mbit/s for the 10-16 videos; AAC 160k; MP3 192k), and all
file metadata was stripped. The 10-16 reel, explainer and café scene were assembled from studio output as described below.

## Showcase, 2026-10-16

### `video/showcase-reel-57s.mp4`
A one-minute tour of what the studio makes: ten stills, three Fast HD clips with their own sound, two shots of the café scene
below (with their voices), the watch edit as a before/after wipe, an excerpt of the matrix explainer, and an end card. The
footage, stills, voices and score all came out of the studio; the cuts, slow camera moves, captions and the mix were put
together afterwards with a short Python + ffmpeg script. Score: ACE-Step (Fast), 75 s asked, 110 BPM, C major, "cinematic
uplifting trailer score, orchestral strings and modern electronic percussion, rising tension into a soaring finale, instrumental".

### `video/matrix-math-explainer-62s.mp4`: "Matrix math, at the speed of light"
- **Footage:** ten Fast HD shots (8 s each), written as picture / action / sound: a person at a laptop at night, a point of light
  exploding into a lattice, a bending grid, crossing walls of light, a GPU die, a liquid-cooled server board, a data-centre aisle,
  a neural network, three orbs trading arcs of light, and a pull-back to a galaxy-sized web. Director reviews: 6–8/10.
- **Math:** the matrices, products, the neural-layer example and the attention weights are computed with numpy and drawn over the
  footage by a script; the video model never draws a number. The counters are stated as approximations on screen
  (a 70-billion-parameter model does about one multiply-add per parameter per word).
- **Voice:** a synthetic voice made with Fish Audio Drama 3 (no real person's voice); ten lines, each checked with Whisper and
  word-perfect on the first take.
- **Score:** ACE-Step (Fast), 80 s, 128 BPM, D minor: "epic cinematic electronic trailer score, driving 128 bpm pulse, punchy kick drum, deep sub bass, fast arpeggiated synths, rising risers and big impacts, hybrid orchestral hits, futuristic and exciting, instrumental, builds to a huge climax". The cuts sit on its beat grid.

### `video/cafe-conversation-cast-3-voices-24s.mp4`: three people, one café
- **Cast mode** (MiniMax H3 reference-to-video): one shot per speaker from a Z-Image-Turbo portrait
  (`images/cast-portraits-z-image-turbo.jpg`) and that character's synthetic voice (Fish Audio Drama 3, made from text alone).
  Same scene in all three shots, written so each shot shows only its speaker (the friends stay behind the camera) and looks
  filmed rather than generated: *Documentary-style handheld medium close-up, shot on a 35mm cinema camera: the person sits alone in the frame at a small worn wooden table in a busy seaside cafe, a large window of plain clear glass behind them (no writing, signs or lettering anywhere) with a grey-blue sea and a distant white lighthouse. Their two friends sit across the table behind the camera and are never seen: nobody else in the foreground, no shoulders, heads or hands at the edges of the frame; only a few soft, out-of-focus strangers far in the background. Soft overcast daylight from the window, real skin texture with pores and fine lines, natural unretouched look, a slight natural sway of a hand-held camera, fine film grain, muted true-to-life colours, like a frame from an independent film.*
- **Lines** (Whisper hears each one word for word):
  - Maya: "Okay, I found it. A lighthouse on the cliffs, and you can actually stay the night in the keeper's cottage."
  - Daniel: "A lighthouse? Maya, the last time you booked something romantic, we ended up in a barn."
  - Lena: "Oh, come on, Daniel. Sunrise over the sea, no phones, just us. I'm already packing."
- **Review:** director 8, 8 (second take) and 7/10, about 25 min per shot. Every shot was also checked for lip sync with SyncNet
  (offset 0 or -1 frame, confidence 7.9-9.1), word for word with Whisper, and by eye for extra people and garbled text (one take of
  Lena was redone because the window behind her carried made-up lettering). A light grade (fine grain, gentle contrast curve) was
  added afterwards with the audio copied untouched; the three shots were joined and the café score (`music/cafe-jazz-ace-step-40s.mp3`,
  ACE-Step, 92 BPM, D major) mixed under the dialogue with ffmpeg. The finished cut measures 0, -1 and -1 frames with SyncNet.
- Maya, Daniel and Lena are fictional; no real person's face or voice is used.

### `video/ink-fluid-flow-fasthd-8s.mp4`
Fast HD, 8 s, seed 5141983, picture and sound in one pass, review 8/10, 7 min. Prompt: *Macro view of clear water against a pure black background, lit by bright studio light. A plume of deep blue ink and a plume of molten-gold ink enter from opposite sides. The two inks meet in the centre and roll up into curling vortices and Kelvin-Helmholtz waves, fine filaments stretching and folding into swirling eddies; slow, graceful motion; the camera holds still, very slowly pushing in. Sound: A soft deep underwater hum with gentle bubbling and a calm ambient synth pad.*

### Images (1344x768 unless noted)

| file | model | prompt |
|---|---|---|
| `images/f1-car-flux3-klein.jpg` | FLUX 3 Image Klein | Hyperrealistic motorsport photograph of a modern Formula One race car at full speed through a fast corner at dusk, low panning shot with the background streaked by motion blur while the car stays tack-sharp, sparks flying from the floor, glowing brake discs, a deep blue and gold livery with a stylised golden lion emblem, generic plain sponsor panels, heat haze over the tarmac, floodlights and grandstands blurred behind, telephoto lens, professional sports photography. |
| `images/t-rex-flux3-klein.jpg` | FLUX 3 Image Klein | Hyperrealistic wildlife photograph of a Tyrannosaurus rex striding through a misty prehistoric fern forest at dawn, shafts of golden light through towering conifers, scaly textured skin with scars and mud, breath steaming in the cool air, a flock of small pterosaurs lifting off behind it, shot like a BBC nature documentary, telephoto lens, shallow depth of field, photorealistic. |
| `images/brachiosaurus-krea2-realism.jpg` | Krea 2 Turbo + realism LoRA | Hyperrealistic photograph of a herd of long-necked Brachiosaurus wading across a wide shallow river at sunset, water splashing around their legs, reflections on the surface, volcanic mountains on the horizon with a thin plume of smoke, warm orange light, wildlife documentary style, photorealistic detail on the wrinkled skin. |
| `images/spiral-galaxy-flux3-klein.jpg` | FLUX 3 Image Klein | Breathtaking deep-space astrophotograph of a grand-design spiral galaxy seen at an angle, glowing golden core, sweeping blue spiral arms dotted with pink star-forming nebulae and dark dust lanes, a smaller companion galaxy nearby, thousands of background galaxies and sharp foreground stars with diffraction spikes, space telescope image quality, rich true colour, photorealistic. |
| `images/milky-way-krea2-realism.jpg` | Krea 2 Turbo + realism LoRA | Hyperrealistic night-sky photograph from a mountain lake: the Milky Way galactic core arching over snow-capped peaks, perfectly mirrored in the still water, the Andromeda galaxy visible as a soft smudge, faint green airglow on the horizon, a single tent glowing warm orange on the shore, long-exposure astrophotography, pin-sharp stars, photorealistic. |
| `images/snow-leopard-krea2-realism.jpg` | Krea 2 Turbo + realism LoRA | Hyperrealistic wildlife photograph of a snow leopard crouched on a snowy Himalayan ridge, thick spotted coat dusted with snowflakes, pale green eyes staring into the lens, falling snow, blue-grey mountains in the haze behind, 500mm lens, shallow depth of field, photorealistic. |
| `images/elephants-flux3-klein.jpg` | FLUX 3 Image Klein | Hyperrealistic wildlife photograph of an elephant herd with a calf crossing a dusty plain at sunset, red dust lit by low sun, silhouettes of more elephants in the haze, Mount Kilimanjaro faint on the horizon, telephoto compression, nature documentary style, photorealistic. |
| `images/ink-fluid-flux3-klein.jpg` | FLUX 3 Image Klein | High-speed macro photograph of fluid dynamics: two streams of ink, deep blue and molten gold, colliding in clear water and curling into turbulent vortices and Kelvin-Helmholtz waves, fine filaments and swirling eddies frozen in time, black background, studio strobe lighting, ultra-sharp detail, photorealistic scientific art photography. |
| `images/wind-tunnel-krea2-realism.jpg` | Krea 2 Turbo + realism LoRA | Hyperrealistic photograph of a wind-tunnel smoke visualisation over a sleek aerofoil: smooth laminar smoke streamlines flowing over the top surface, separating into a turbulent wake with a clear von Karman vortex street behind the trailing edge, dark lab background, laser-sheet lighting, scientific photography, photorealistic. |
| `images/lion-poster-qwen-image-2.1.jpg` | Qwen-Image 2.1 (1024x1344) | Elegant minimalist poster: a golden lion head in profile on a deep charcoal background, fine engraved line-art style. At the top in bold modern serif lettering the words LION AI STUDIOS, and below in small spaced capitals PROFESSIONAL MEDIA SUITE. Clean layout, high contrast, print quality. |
| `images/watch-edit-flux3-klein.jpg` | Juggernaut Z (left), then edited with FLUX 3 Image Klein (right) | Original: *Product photograph of a luxury automatic wristwatch with a deep green dial and a brushed steel case resting on a dark slate stone, a few drops of water beading on the crystal, dramatic side light, shallow depth of field, studio quality, photorealistic.* Edit: *Give the watch a clean, plain deep-green sunburst dial with only the steel hour markers, the hands and the date window, and add a small engraved golden lion emblem at the twelve o clock position. Keep everything else exactly the same: the steel case, the slate stone, the water drops, the lighting and the framing.* |
| `images/cast-portraits-z-image-turbo.jpg` | Z-Image-Turbo (1024x1280 each) | Three head-and-shoulders studio portraits of fictional people, used as the Cast references for the café scene. |

## 2026-10-10

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

FLUX 3 Image Klein (16 steps, prompt upsampler on, one image per prompt):

| file | prompt |
|---|---|
| `images/serengeti-lion-flux3-klein.jpg` (1344x768) | A lion resting on a granite rock on the Serengeti at golden hour |
| `images/deco-poster-flux3-klein.jpg` (1024x1024) | Art deco travel poster that reads LION AI STUDIOS with the line Made on a DGX Spark underneath, a gold lion over a night city skyline |
| `images/mug-edit-flux3-klein.jpg` | Left: Qwen-Image 2.1 Turbo, "Product photo of a matte black coffee mug on a marble counter, soft morning light". Right: that image edited with FLUX 3 Image Klein, "Make the mug glossy red ceramic with steam rising from it" |

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
