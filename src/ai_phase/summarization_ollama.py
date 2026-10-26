from ollama import chat

text = """
Artificial intelligence is increasingly used in education to provide personalized learning experiences. AI-powered systems can analyze a student’s performance, identify areas of difficulty, and recommend exercises that match the student’s individual needs. These systems can also give immediate feedback, allowing learners to correct mistakes without waiting for a teacher. However, AI should support teachers rather than replace them because educators provide emotional support, creativity, and an understanding of each student’s personal circumstances. The use of AI also creates concerns about privacy, since educational applications may collect sensitive information about students and their learning behavior. Schools must therefore protect student data, evaluate AI tools carefully, and train teachers to use them responsibly. When implemented with appropriate safeguards and human supervision, artificial intelligence can make education more accessible, efficient, and responsive while allowing teachers to focus on activities that require human judgment and interaction.
"""

#I tried to be more strict at writing the prompt(in the last sentence) as before adding it, the output length was more than two sentences 
prompt = f"""
summarize the text in exactly 2 sentences
make sure the output contains exactly two periods


Text:
{text}
"""


response = chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    options={
        "temperature": 0.2,
        "num_predict": 300,
        "seed": 42,
    },
)



print(response.message.content)
