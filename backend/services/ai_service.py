import os
import time
import torch
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load .env
load_dotenv()

# -------------------------
# ENV VARIABLES
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
IBM_MODEL = os.getenv("IBM_MODEL", "ibm-granite/granite-4.0-h-350m")

# -------------------------
# 1) IBM GRANITE (Local) for CHAT
# -------------------------
device = "cpu"
print(f"⚙️ Loading IBM Granite model: {IBM_MODEL} ...")

granite_model = None
granite_tokenizer = None

try:
    granite_tokenizer = AutoTokenizer.from_pretrained(IBM_MODEL, trust_remote_code=True)
    granite_model = AutoModelForCausalLM.from_pretrained(
        IBM_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

    granite_model.eval()
    print("✅ IBM Granite model loaded!")

except Exception as e:
    print(f"❌ Granite load failed: {e}")

# -------------------------
# 2) GROQ CLIENT (LLaMA) for Branding
# -------------------------
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
    print("⚠️ Warning: GROQ_API_KEY not set in .env")

async def generate_with_groq(prompt: str, max_tokens: int = 250) -> str:
    """Generate using Groq LLaMA 3."""
    if not groq_client:
        return "❌ Error: GROQ_API_KEY not set in .env"

    message = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    return message.choices[0].message.content.strip()

# -------------------------
# 3) SDXL LOGO GENERATION (HuggingFace)
# -------------------------
async def generate_logo_image(logo_prompt: str):
    """Generate logo image using Stable Diffusion XL."""
    try:
        if not HF_API_KEY:
            return {"success": False, "error": "HF_API_KEY not set in .env"}

        print("🎨 Generating logo with Stable Diffusion XL...")

        client = InferenceClient(api_key=HF_API_KEY)

        enhanced_prompt = (
            f"Professional brand logo for: {logo_prompt}. "
            f"Modern minimalist vector style, clean, centered, high quality."
        )

        image = client.text_to_image(
            enhanced_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )

        # Save inside frontend/static/generated_logos/
        os.makedirs("../frontend/static/generated_logos", exist_ok=True)
        timestamp = int(time.time())
        filename = f"logo_{timestamp}.png"
        save_path = os.path.join("../frontend/static/generated_logos", filename)

        image.save(save_path)

        print(f"✅ Logo saved at: {save_path}")

        return {
            "success": True,
            "image_url": f"/static/generated_logos/{filename}",
            "error": None
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
