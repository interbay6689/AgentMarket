import os
import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from Backtest_projects.fetch_news_rss import link_news

client = OpenAI(api_key="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")

def extract_text_from_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        print(paragraphs.text)
        return "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])[:3000]
    except Exception as e:
        return f"ERROR extracting content: {e}"

def analyze_links_with_gpt(link: str):

    article_text = extract_text_from_url(link)
    if article_text.startswith("ERROR"):
        return article_text

    prompt = f"""
הטקסט הבא הוא כתבת חדשות כלכלית. נתח את התוכן והצג את הקטגוריות הכלכליות המרכזיות שהכתבה עוסקת בהן.

עבור כל קטגוריה אפשרית (כגון: אינפלציה, ריבית, שוק ההון, פוליטיקה, מט"ח, אנרגיה, גיאופוליטי, אבטלה, נדל"ן, סנטימנט כלכלי וכו'):
- קטגוריה
- השפעה: מספר בין -3 ל+3
- אחוז רלוונטיות (0–100%)
- משפט תמיכה מהכתבה

כתבה:
\"\"\"
{article_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "אתה מומחה כלכלה, מזהה השפעות כתבות על חוזים עתידיים."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR calling GPT: {e}"

if __name__ == "__main__":
    urls = link_news()

    for i, url in enumerate(urls, 1):
        print(f"\n🔗 ({i}/{len(urls)}) מנתח כתבה:\n{url}")
        result = analyze_links_with_gpt(url)
        print(result)
