# Security policy

- Never commit Telegram tokens, API keys, model-hub tokens, private video, or `.env` files.
- Revoke the Telegram token present in the legacy Git history before running the bot.
- Network access requires a `HAD_API_KEY` of at least 24 characters. Without a key, only loopback
  clients are accepted.
- MiniCPM contains custom Python code. Download it once with `had-download-minicpm`, review the
  recorded SHA/Hashes, and deploy only the local snapshot.
- Treat checkpoints as untrusted input. Runtime loading uses `torch.load(..., weights_only=True)`.
- Rotate a secret immediately if it appears in an issue, log, screenshot, commit, or build artifact.

Report vulnerabilities privately to the repository owner. Do not include real credentials or
private video in a report.
