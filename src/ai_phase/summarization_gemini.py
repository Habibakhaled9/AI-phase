import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

text = """
Artificial intelligence is increasingly used in education to provide personalized learning experiences. AI-powered systems can analyze a student’s performance, identify areas of difficulty, and recommend exercises that match the student’s individual needs. These systems can also give immediate feedback, allowing learners to correct mistakes without waiting for a teacher. However, AI should support teachers rather than replace them because educators provide emotional support, creativity, and an understanding of each student’s personal circumstances. The use of AI also creates concerns about privacy, since educational applications may collect sensitive information about students and their learning behavior. Schools must therefore protect student data, evaluate AI tools carefully, and train teachers to use them responsibly. When implemented with appropriate safeguards and human supervision, artificial intelligence can make education more accessible, efficient, and responsive while allowing teachers to focus on activities that require human judgment and interaction.
"""

prompt = f"""
Summarize the following text in exactly 2 sentences.
Preserve the most important facts and do not add new information.

Text:
{text}
"""

load_dotenv()

client = genai.Client(
    api_key=os.getenv("gemini_api_key")
)

config = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=300,
)



response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
    config=config,
)



print(response.text)
