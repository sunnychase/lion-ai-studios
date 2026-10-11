# The director

The director is a vision-language model the studio talks to over an **OpenAI-compatible chat API**. It:

- **writes**: turns one sentence into a full shot prompt (picture, action, sound) with one beat per 15 s segment, and music briefs;
- **reviews**: watches every rendered segment and scores it against its prompt; with *Director review → re-take* a weak
  segment is rendered once more with the director's fix;
- **picks voices and describes shots** for *Recreate a Short* (it watches the Short's shots); in the *Faithful* style it picks out
  the performer once and checks every anchor frame (is the performer there, is the face visible, expression, makeup) before a swap;
- **arranges MIDI files**: reads the parts of an attached `.mid` and a description of the sound you want, and picks an instrument
  or drum kit for every part, a style line for the AI restyle, and whether the restyle is needed.

It must accept **video input** (`"type": "video_url"` content parts with a `data:video/mp4;base64,…` URL). Without a
director the studio still renders; only *Write it*, reviews, music briefs, MIDI arranging and *Recreate* need it.

## Option A — TensorFold on the same Spark (what the studio was developed with)

[MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold) serves
Qwen3.8-27B (4-bit MLX weights + the `z-lab/Qwen3.8-27B-DFlash2` drafter) with vision on one DGX Spark. Run it *small*
so it stays loaded next to ComfyUI (~19 GiB in all):

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold ~/director
cd ~/director && cat > .env <<'ENV'
PORT=8486
HOST=127.0.0.1
SERVED_NAME=studio-director
CONTAINER_NAME=studio-director
PARALLEL=1
CONTEXT=65536
KV_POOL_GB=6
ENV
./start.sh            # first run downloads the weights and builds the image (follow that repo's README)
```

Then in `studio.env`:

```bash
DIRECTOR_URL=http://127.0.0.1:8486
DIRECTOR_MODEL=studio-director
DIRECTOR_LABEL="Qwen3.8-27B director"
DIRECTOR_SERVICE=lion-director.service     # optional: systemd/lion-director.service starts it on first use
```

## Option B — any other OpenAI-compatible server

vLLM, SGLang or llama.cpp's server work too, as long as the model takes video. Point `DIRECTOR_URL` at the server root
(the studio calls `/v1/models` and `/v1/chat/completions`) and set `DIRECTOR_MODEL` to the served name. The official
weights are [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0, vision built in; ~56 GB in bf16, so
a 4/8-bit build is the practical choice next to ComfyUI).

Requests send `chat_template_kwargs: {"enable_thinking": false}`; servers that do not know the field ignore it.

## Optional writer

`WRITER_URL` / `WRITER_MODEL` name a second, text-only LLM you may already run. Whenever it answers, *writing* uses it and
the director only reviews.
