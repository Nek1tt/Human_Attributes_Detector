"""Telegram client for the detector API. The bot never loads ML models itself."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


async def _wait_for_result(client, api_base: str, api_key: str, job_id: str) -> bytes:
    headers = {"X-API-Key": api_key}
    for _ in range(300):
        response = await client.get(f"{api_base}/api/v1/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        state = response.json()["state"]
        if state == "completed":
            result = await client.get(
                f"{api_base}/api/v1/jobs/{job_id}/result", headers=headers, timeout=120
            )
            result.raise_for_status()
            return result.content
        if state == "failed":
            raise RuntimeError(response.json().get("error") or "Video analysis failed")
        await asyncio.sleep(2)
    raise TimeoutError("Video analysis did not finish within 10 minutes")


async def handle_video(update, context) -> None:
    import httpx

    message = update.effective_message
    telegram_file = message.video or message.document
    if telegram_file is None:
        await message.reply_text("Пришлите видеофайл.")
        return
    max_bytes = int(os.getenv("HAD_BOT_MAX_BYTES", str(50 * 1024 * 1024)))
    if telegram_file.file_size and telegram_file.file_size > max_bytes:
        await message.reply_text("Файл слишком большой.")
        return
    status = await message.reply_text("Видео принято, начинаю анализ.")
    api_base = os.getenv("HAD_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    api_key = _required_env("HAD_API_KEY")
    suffix = Path(getattr(telegram_file, "file_name", "") or "video.mp4").suffix or ".mp4"
    try:
        with tempfile.TemporaryDirectory(prefix="had-bot-") as temp_dir:
            source = Path(temp_dir) / f"input{suffix}"
            remote = await context.bot.get_file(telegram_file.file_id)
            await remote.download_to_drive(source)
            async with httpx.AsyncClient(timeout=120) as client:
                with source.open("rb") as stream:
                    response = await client.post(
                        f"{api_base}/api/v1/jobs",
                        headers={"X-API-Key": api_key},
                        files={"file": (source.name, stream, "application/octet-stream")},
                    )
                response.raise_for_status()
                result = await _wait_for_result(
                    client,
                    api_base,
                    api_key,
                    response.json()["job_id"],
                )
            output = Path(temp_dir) / "result.mp4"
            output.write_bytes(result)
            with output.open("rb") as video:
                await message.reply_video(video=video, caption="Анализ завершён.")
        await status.delete()
    except Exception as exc:
        await status.edit_text(f"Не удалось обработать видео: {str(exc)[:300]}")


async def start_command(update, context) -> None:
    del context
    await update.effective_message.reply_text(
        "Пришлите видео — я верну версию с найденными людьми и атрибутами."
    )


def main() -> None:
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    token = _required_env("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
