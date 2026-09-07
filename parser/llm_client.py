"""
Shared LLM helper for the parser/normalization pipeline.
Mirrors graph/entity_extractor.py's Ollama call pattern exactly (same
model, same deterministic decoding options, same proxy bypass, same
brace-slicing JSON extraction) so the Document Type Detector's LLM
fallback and the Template Normalizer's gap-filling pass behave
consistently with the rest of the project instead of introducing a
second calling convention.
"""
import json
import requests
from config import OLLAMA_URL
MODEL = "llama3.1:latest"
def call_ollama(prompt: str, timeout: int = 120, num_ctx: int = 8192, model: str = None, base_url: str = None) -> dict:
    """
    Call Ollama's /api/generate with the project's standard deterministic
    settings. Returns {"raw": str, "error": Optional[str]} — never raises;
    errors come back in "error" the same way entity_extractor.py does it,
    so callers can decide whether to fall back further.
    num_ctx defaults to 8192 (well above Ollama's own 2048-token default,
    which applies regardless of what the underlying model actually
    supports). Callers processing something long — e.g. a whole-document
    canonicalization pass rather than a single chunk — should pass a
    larger value explicitly; don't rely on the default here for that.
    model defaults to this module's MODEL (llama3.1:latest, matching
    the rest of the pipeline) when not given. Pass a different model
    name for a step that needs different characteristics — e.g.
    parser/canonicalizer.py uses a larger/coder model for whole-document
    reformatting, while per-chunk work stays on the cheaper default.
    base_url defaults to config.OLLAMA_URL when not given. Pass a
    different value when the model you want isn't served by the
    project's default Ollama instance at all — e.g. the coder models
    canonicalizer.py uses live on a separate server
    (config.OLLAMA_CODER_URL), not just a different model name on the
    same host. Passing model= alone against the wrong host is what
    produces a 404, not a connection error — the server is reachable,
    it just doesn't have that model.
    """
    target_url = base_url or OLLAMA_URL
    try:
        response = requests.post(
            f"{target_url}/api/generate",
            json={
                "model":  model or MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed":        42,
                    "top_p":       1,
                    "top_k":       1,
                    "num_ctx":     num_ctx
                }
            },
            proxies={"http": None, "https": None},
            timeout=timeout
        )
        response.raise_for_status()
        return {"raw": response.json()["response"], "error": None}
    except requests.exceptions.ConnectionError as e:
        return {"raw": "", "error": f"Cannot reach Ollama server at {target_url}: {str(e)}"}
    except requests.exceptions.Timeout:
        return {"raw": "", "error": f"Ollama request to {target_url} timed out ({timeout}s)"}
    except requests.exceptions.HTTPError as e:
        return {"raw": "", "error": f"Ollama request to {target_url} failed: {str(e)} — check the model is actually pulled on that server"}
    except Exception as e:
        return {"raw": "", "error": f"Ollama request to {target_url} failed: {str(e)}"}
def extract_json_block(raw_output: str):
    """
    Find the first { or [ and parse exactly one JSON value starting
    there, via json.JSONDecoder().raw_decode() — NOT rfind() to the
    last matching bracket, which breaks (json.JSONDecodeError: Extra
    data) the moment the LLM appends any trailing text after the JSON,
    same failure mode fixed in graph/entity_extractor.py.
    Returns the parsed object. Raises ValueError if no bracket is
    found, or json.JSONDecodeError if what follows isn't valid JSON —
    callers should catch both.
    """
    obj_start = raw_output.find("{")
    arr_start = raw_output.find("[")
    starts = [s for s in (obj_start, arr_start) if s != -1]
    if not starts:
        raise ValueError("LLM did not return any JSON block")
    # Prefer whichever bracket type appears first in the output, since a
    # normalizer prompt asking for a JSON array could still get a stray
    # "{" inside prose commentary before it.
    start = min(starts)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(raw_output, start)
    return parsed

def call_ollama_vision(prompt: str, image_path: str, timeout: int = 180,
                        model: str = None, base_url: str = None) -> dict:
    """
    Vision counterpart to call_ollama() above -- same return shape
    ({"raw": str, "error": Optional[str]}), same proxy-bypass, same
    deterministic decoding options. Only difference: sends the image
    as base64 via Ollama's `images` field on /api/generate (which
    supports multimodal models the same way /api/chat does).

    model/base_url default to VISION_MODEL/VISION_MODEL_URL from
    config.py rather than the text MODEL/OLLAMA_URL -- the vision
    model very likely lives on a different host/name than the text
    extraction model (same reasoning EXTRACTION_MODEL_URL already
    exists for the coder model).
    """
    import base64
    from config import VISION_MODEL, VISION_MODEL_URL

    target_url = base_url or VISION_MODEL_URL
    target_model = model or VISION_MODEL

    try:
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
    except OSError as e:
        return {"raw": "", "error": f"Could not read image file {image_path}: {e}"}

    try:
        response = requests.post(
            f"{target_url}/api/generate",
            json={
                "model": target_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "options": {"temperature": 0, "seed": 42, "top_p": 1, "top_k": 1},
            },
            proxies={"http": None, "https": None},
            timeout=timeout,
        )
        response.raise_for_status()
        return {"raw": response.json()["response"], "error": None}
    except requests.exceptions.ConnectionError as e:
        return {"raw": "", "error": f"Cannot reach vision Ollama server at {target_url}: {str(e)}"}
    except requests.exceptions.Timeout:
        return {"raw": "", "error": f"Vision request to {target_url} timed out ({timeout}s)"}
    except requests.exceptions.HTTPError as e:
        return {"raw": "", "error": f"Vision request to {target_url} failed: {str(e)} — check {target_model} is actually pulled on that server"}
    except Exception as e:
        return {"raw": "", "error": f"Vision request to {target_url} failed: {str(e)}"}
