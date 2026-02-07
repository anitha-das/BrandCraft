from services.ai_service import generate_with_groq


# -------------------------------
# 2.5 Brand Name Generator
# -------------------------------
async def generate_brand_names(industry: str, keywords: str, tone: str, language: str = "en"):
    prompt = f"""
You are BizForge. You are a world-class branding expert.

TASK:
Generate 15 unique, memorable, brand-ready business names.

RULES:
- Names must match the industry
- Use the keywords naturally (do not repeat the same keyword in every name)
- Match the tone: {tone}
- Language: {language}
- Avoid names that are too long
- Avoid generic names like "Tech Solutions", "Business Hub"
- Avoid trademark-like famous names
- Provide a short one-line meaning/explanation for each name

INPUT:
Industry: {industry}
Keywords: {keywords}
Tone: {tone}
Language: {language}

OUTPUT FORMAT (STRICT):
1. Name - explanation
2. Name - explanation
...
15. Name - explanation
"""
    return await generate_with_groq(prompt, max_tokens=500)


# -------------------------------
# 2.6 Marketing Content Generator
# -------------------------------
async def generate_marketing_content(
    brand_name: str,
    content_type: str,
    tone: str,
    description: str,
    language: str = "en"
):
    prompt = f"""
You are BizForge. You are a professional marketing copywriter.

TASK:
Generate high-quality marketing content for this brand.

Brand Name: {brand_name}
Brand Description: {description}

Tone: {tone}
Language: {language}
Content Type: {content_type}

CONTENT TYPE RULES:
- product_description: 120-180 words, persuasive, clean, benefits + features
- caption: 2 short catchy captions + 10 hashtags
- ad_copy: headline + 2 short ad variations + CTA
- email: subject line + email body (short)
- tagline: 10 short taglines

IMPORTANT RULES:
- Keep it on-brand and consistent
- Avoid repeating the brand name too many times
- Avoid generic boring lines
- Output must be well formatted

Return only the final content.
"""
    return await generate_with_groq(prompt, max_tokens=500)

# -------------------------------
# 2.7 Sentiment Analysis
# -------------------------------
async def analyze_sentiment(text: str, brand_tone: str):
    prompt = f"""
You are BizForge. Analyze this customer review sentiment.

Review:
{text}

TASK:
1) Sentiment must be one of: Positive / Neutral / Negative
2) Give a confidence score from 0 to 100
3) Explain in 1-2 lines WHY you chose that sentiment
4) Check if the review tone matches the brand tone "{brand_tone}"
5) Rewrite the review into a {brand_tone} tone (polite and professional)

Return STRICTLY in this format:

Sentiment: ...
Confidence: ...
Reason: ...
Tone Match: Yes/No
Rewrite: ...
"""
    return await generate_with_groq(prompt, max_tokens=400)



# -------------------------------
# 2.8 Chatbot
# -------------------------------
async def chat_with_ai(message: str):
    prompt = f"""
You are BizForge Branding Chatbot.
Answer clearly, professionally, and helpfully.

User message: {message}
"""
    return await generate_with_groq(prompt, max_tokens=350)


# -------------------------------
# 2.9 Color Palette
# -------------------------------
async def get_color_palette(tone: str, industry: str):
    prompt = f"""
Suggest a branding design system.

Industry: {industry}
Tone: {tone}

Give:
1) 5 color hex codes
2) 2 font suggestions
3) UI style suggestions

Return in a neat format.
"""
    return await generate_with_groq(prompt, max_tokens=350)
