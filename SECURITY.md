# Security policy

## Reporting a vulnerability

Please report security problems **privately**, through **Report a vulnerability** in this repository's **Security** tab
(GitHub private vulnerability reporting). Do not open a public issue, and do not post exploit details, tokens or private
media anywhere public.

Include what is affected, how to reproduce it, what an attacker could do with it, and the commit you tested. Remove
personal paths, addresses and credentials from logs before you send them. You will get an acknowledgement, and the fix and
its disclosure will be coordinated with you.

There are no releases or version numbers: fixes go to `main` and are listed in [CHANGELOG.md](CHANGELOG.md).

## What the studio is built for

Lion AI Studios is a tool for **one trusted user on their own machine**, or at most a trusted local network. Please keep
these boundaries in mind when you assess a report:

- **No login.** The web app listens on `127.0.0.1` by default and has no accounts. Anyone who can reach its port can use
  every feature. To use it from another device, put it behind an authenticating reverse proxy or tunnel; never expose it
  bare to a network you do not trust.
- **`studio.env` is trusted configuration.** Settings such as `GPU_TENANT_STOP` or `LOCALROUTER_START` are shell commands
  the studio runs as your user, by design. Whoever can edit `studio.env` can already run commands as you.
- **Optional services stay local.** The director, ComfyUI and the optional LocalRouter are expected on `127.0.0.1` too, and
  none of them has authentication of its own.
- **Downloads.** Pasting a YouTube link makes the studio fetch that video with `yt-dlp`, and models come from the sources
  listed in [docs/MODELS.md](docs/MODELS.md).

Reports we especially want: reading or writing files outside the studio's folders through the web API, running commands
through request data, cross-site requests that start jobs or delete files, and anything that leaks your library to another
user of the machine.
