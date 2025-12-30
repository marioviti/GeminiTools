from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, Dict, List
import json
import logging
import base64
import io

from PIL import Image

from google import genai
from google.genai import types
from google.oauth2 import service_account


logger = logging.getLogger(__name__)


from typing import Any, Dict, Optional


def extract_inference_metadata(resp: Any) -> Dict[str, Any]:
    """
    Best-effort metadata extractor for google.genai responses.
    Returns a stable dict even if some fields are missing.
    """
    md: Dict[str, Any] = {}

    # ---- Usage metadata (tokens, etc.)
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        md["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "candidate_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            # some variants expose these:
            "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
        }

    # ---- Model / request routing
    md["model"] = getattr(resp, "model_version", None) or getattr(resp, "model", None)

    # ---- Finish / safety / per-candidate details
    cands = getattr(resp, "candidates", None) or []
    md["candidates"] = []
    for c in cands:
        cand = {
            "finish_reason": getattr(c, "finish_reason", None),
            "avg_logprobs": getattr(c, "avg_logprobs", None),
        }

        # safety ratings if present
        safety = getattr(c, "safety_ratings", None)
        if safety:
            cand["safety_ratings"] = [
                {
                    "category": getattr(r, "category", None),
                    "probability": getattr(r, "probability", None),
                    "severity": getattr(r, "severity", None),
                    "blocked": getattr(r, "blocked", None),
                }
                for r in safety
            ]

        md["candidates"].append(cand)

    return md


# ----------------------------
# Small helpers
# ----------------------------
def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_pil(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGBA")


def maybe_json_load(s: Optional[str]) -> Optional[Any]:
    if not s:
        return None
    s = s.strip()
    # if model returns fenced json, strip fences
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s else s
        s = s.replace("json", "", 1).strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_parts(response: Any) -> List[Any]:
    """
    Compatible with google.genai response objects:
    response.candidates[*].content.parts[*]
    """
    try:
        parts: List[Any] = []
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for p in getattr(content, "parts", []) or []:
                parts.append(p)
        return parts
    except Exception:
        return []


def extract_text(response: Any) -> Optional[str]:
    # Prefer SDK convenience if available
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Fallback to parts
    for p in extract_parts(response):
        t = getattr(p, "text", None)
        if isinstance(t, str) and t.strip():
            return t
    return None


def extract_images(
    response: Any,
    *,
    output: Literal["pil", "bytes", "b64"] = "pil",
    force_format: Literal["png", "jpeg", None] = None,
    jpeg_quality: int = 95,
) -> List[Union[Image.Image, bytes, str]]:
    """
    Extract images from a Gemini response.

    output:
      - "pil"   -> return PIL.Image (decoded)
      - "bytes" -> return raw image bytes (as provided or re-encoded)
      - "b64"   -> return base64-encoded string

    force_format:
      - None    -> keep original mime
      - "png"   -> re-encode as PNG (lossless)
      - "jpeg"  -> re-encode as JPEG (lossy, quality via jpeg_quality)
    """
    out = []

    for p in extract_parts(response):
        inline = getattr(p, "inline_data", None)
        if inline is None:
            continue

        mime = getattr(inline, "mime_type", None)
        data = getattr(inline, "data", None)

        if mime not in ("image/png", "image/jpeg"):
            continue
        if not isinstance(data, (bytes, bytearray)):
            continue

        try:
            raw_bytes = bytes(data)

            # 1️⃣ Keep raw bytes exactly as received
            if force_format is None and output in ("bytes", "b64"):
                if output == "bytes":
                    out.append(raw_bytes)
                else:
                    out.append(base64.b64encode(raw_bytes).decode("ascii"))
                continue

            # 2️⃣ Decode to PIL
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

            # 3️⃣ Return PIL directly
            if output == "pil" and force_format is None:
                out.append(img)
                continue

            # 4️⃣ Re-encode (compression control)
            buf = io.BytesIO()
            if force_format == "jpeg":
                img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
                new_bytes = buf.getvalue()
            else:  # png (lossless)
                img.save(buf, format="PNG")
                new_bytes = buf.getvalue()

            if output == "bytes":
                out.append(new_bytes)
            elif output == "b64":
                out.append(base64.b64encode(new_bytes).decode("ascii"))
            else:
                out.append(Image.open(io.BytesIO(new_bytes)))

        except Exception:
            logger.exception("Failed extracting image")

    return out


# ----------------------------
# Config
# ----------------------------
@dataclass
class GeminiRuntime:
    model_id: str = "gemini-2.5-flash"
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 1024


class GeminiLLM:
    """
    Clean adapter for google.genai.

    Usage:
      llm = GeminiLLM(api_key="...", runtime=GeminiRuntime(model_id="gemini-2.5-flash"))
      out = llm.generate("hello")  -> out["text"]
      out = llm.generate(contents=[...], response_schema=types.Schema(...)) -> out["json"]
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        creds_path: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        runtime: Optional[GeminiRuntime] = None,
    ):
        self.runtime = runtime or GeminiRuntime()

        if vertexai:
            if not (project and location and creds_path):
                raise ValueError("Vertex mode requires project, location, creds_path.")
            creds = service_account.Credentials.from_service_account_file(
                creds_path, scopes=scopes
            )
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=creds,
            )
        else:
            if not api_key:
                raise ValueError("Non-Vertex mode requires api_key.")
            self.client = genai.Client(api_key=api_key)

    # ----------------------------
    # Content builders (optional)
    # ----------------------------
    @staticmethod
    def text(s: str) -> types.Part:
        return types.Part.from_text(text=s)

    @staticmethod
    def image(img: Union[Image.Image, bytes]) -> types.Part:
        if isinstance(img, Image.Image):
            data = pil_to_png_bytes(img)
            mime = "image/png"
        else:
            data = img
            mime = "image/png"  # assume png bytes
        return types.Part.from_bytes(data=data, mime_type=mime)

    # ----------------------------
    # Main call
    # ----------------------------
    def generate(
        self,
        contents: Union[str, Sequence[Any]],
        *,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_schema: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        response_modalities: Optional[List[str]] = None,
        config: Optional[types.GenerateContentConfig] = None,
        image_output: Literal["pil", "bytes", "b64"] = "pil",
        image_force_format: Literal["png", "jpeg", None] = None,
        image_jpeg_quality: int = 95,
    ) -> Dict[str, Any]:
        mid = model_id or self.runtime.model_id

        cfg = config or types.GenerateContentConfig()

        cfg.temperature = self.runtime.temperature if temperature is None else temperature
        cfg.top_p = self.runtime.top_p if top_p is None else top_p
        cfg.max_output_tokens = self.runtime.max_output_tokens if max_output_tokens is None else max_output_tokens

        # Modalities
        if response_modalities is not None:
            cfg.response_modalities = response_modalities
        else:
            # sensible default based on model name
            cfg.response_modalities = ["IMAGE"] if "image" in mid.lower() else ["TEXT"]

        # Structured output
        if response_schema is not None:
            cfg.response_schema = response_schema
            cfg.response_mime_type = response_mime_type or "application/json"
        elif response_mime_type is not None:
            cfg.response_mime_type = response_mime_type

        resp = self.client.models.generate_content(
            model=mid,
            contents=contents,
            config=cfg,
        )

        txt = extract_text(resp)
        js = maybe_json_load(txt) if (cfg.response_mime_type == "application/json") else None
        imgs = extract_images(resp, output=image_output, 
                    force_format=image_force_format,
                    jpeg_quality=image_jpeg_quality) if ("IMAGE" in (cfg.response_modalities or [])) else []
        meta = extract_inference_metadata(resp)

        return {
            "text": txt,
            "json": js,
            "images": imgs,   # list[PIL.Image.Image]
            "raw": resp,      # full SDK response
            "meta": meta,
        }
