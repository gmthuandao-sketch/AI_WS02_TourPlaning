import json
import os
import sys
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from openai import OpenAI


ENV_FILE = ".env"
GPT_4O_MINI = "gpt-4o-mini"
SYSTEM_MESSAGE = (
    "You are Tour Assistant, a collaborative travel planner. "
    "Lean on the available weather tool whenever the user shares a location, "
    "and respond with concise, engaging guidance tailored to their plans."
)

load_dotenv(ENV_FILE)


def require_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Missing API_KEY. Set it in your environment or .env file.")
        sys.exit(1)
    return api_key


def create_client() -> OpenAI:
    api_key = require_api_key()
    base_url = os.getenv("BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def fetch_weather_window(city: str, hours: int = 12) -> Dict[str, Any]:
    hours = max(1, min(hours, 24))
    try:
        geocode_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geocode_resp.raise_for_status()
        geo_json = geocode_resp.json()
        if not geo_json.get("results"):
            return {"city": city, "error": "No geocoding match"}
        result = geo_json["results"][0]
        latitude = result["latitude"]
        longitude = result["longitude"]
    except requests.RequestException as exc:
        return {"city": city, "error": f"Geocoding failed: {exc}"}

    try:
        forecast_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,precipitation_probability",
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=10,
        )
        forecast_resp.raise_for_status()
        data = forecast_resp.json()
    except requests.RequestException as exc:
        return {"city": city, "error": f"Forecast fetch failed: {exc}"}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation_probability", [])
    samples = []
    for idx in range(min(hours, len(times))):
        samples.append(
            {
                "time": times[idx],
                "temperature_c": temps[idx] if idx < len(temps) else None,
                "precipitation_probability": precip[idx] if idx < len(precip) else None,
            }
        )
    return {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "window_hours": len(samples),
        "samples": samples,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather_window",
            "description": "Query Open-Meteo for upcoming hourly temperature and precipitation chances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 24,
                        "default": 12,
                    },
                },
                "required": ["city"],
            },
        },
    }
]


def parse_arguments(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def handle_tool_call(tool_call: Any) -> Dict[str, Any]:
    name = tool_call.function.name
    args = parse_arguments(tool_call.function.arguments)
    if name == "fetch_weather_window":
        return fetch_weather_window(**args)
    return {"error": f"Unknown tool {name}"}


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        chunks = []
        for part in payload:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(part.get("text", ""))
        return "".join(chunks)
    return ""


def print_response(text: str) -> None:
    divider = "-" * 60
    print(divider)
    print(text.strip())
    print(divider)


def run_cli() -> None:
    client = create_client()
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    messages: List[Dict[str, Any]] = [system_message]
    print("Tour Assistant chat ready. Type '/reset' to clear or '/exit' to quit.")

    while True:
        try:
            user_text = input("You> ").strip()
        except EOFError:
            print("\nEOF received. Bye!")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "/quit"}:
            print("Bon voyage!")
            break
        if lowered == "/reset":
            messages = [system_message]
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_text})

        while True:
            response = client.chat.completions.create(
                model=GPT_4O_MINI,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
            )
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
                for tool_call in message.tool_calls:
                    tool_result = handle_tool_call(tool_call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )
                continue

            assistant_text = extract_text(message.content)
            printable = assistant_text.strip() or "Assistant returned no narrative."
            print_response(printable)
            messages.append({"role": "assistant", "content": assistant_text})
            break


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nSession cancelled by user.")
