from services.ai_service import generate_with_groq

# -------------------------------
# 2.10 Logo Prompt Generator
# -------------------------------
async def generate_logo_prompt(brand_name: str, industry: str, keywords: str):
    prompt = f"""
You are BizForge. You are a professional logo designer.

TASK:
Create a high-quality logo prompt for Stable Diffusion XL.

INPUT:
Brand Name: {brand_name}
Industry: {industry}
Keywords: {keywords}

OUTPUT RULES:
- Must be one single prompt (not multiple options)
- Must describe style, colors, shapes, emotion, and logo type
- Must be suitable for SDXL logo generation
- Must be clean, detailed, and professional

Return only the final prompt text.
"""
    return await generate_with_groq(prompt, max_tokens=300)
