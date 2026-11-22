What is LangChain?
LangChain is an open-source framework for building applications powered by LLMs


LangChain is not an AI model itself. It connects language models to other components, including:
- Documents and external data sources
- Databases
- APIs and external tools
- Conversation history
- Search and retrieval systems
- Multi-step workflows

Why Use LangChain?
When using an LLM directly, we normally send a prompt and receive a response. However, real-world applications may need to:
- Read and answer questions from PDF files.
- remember previous messages.
- Search a database.
- Call weather, search, or finance APIs.
- Perform multiple steps before answering.
- Return structured results such as JSON.
LangChain provides reusable components for these tasks, so developers do not have to build everything from scratch.

  Main Components
1. Models
LangChain offers a similar interface for working with models from different providers, including:
- OpenAI
- Google Gemini
- Anthropic
- Ollama and local models
- AWS Bedrock
Each provider usually has its own integration package.


2. Prompt Templates
A prompt template is a reusable prompt containing variables.
"
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to a beginner using a practical example."
)
"

3. Chains
A chain connects multiple components in a predefined sequence:
Prompt Template → Language Model → Output

Example:
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} using a simple practical example."
)

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

chain = prompt | model

response = chain.invoke({"topic": "LangChain"})
print(response.content)


4. Tools
Tools are functions that an AI model can use when needed.

Examples include:
- A calculator
- Web search
- Weather information
- Database queries
- Custom APIs


5. Agents
An agent can decide what actions to take.

A chain follows steps defined by the developer. An agent can independently decide:
- Whether it needs a tool
- Which tool to use
- Whether another step is necessary
- When it has enough information to provide the final answer
For example, if the user asks:
What is the current temperature in Cairo, and is it higher than 25°C?

The agent can call a weather tool, inspect the result, compare it with 25°C, and produce an answer.


6. Retrieval-Augmented Generation


7. Memory
An LLM does not automatically remember independent requests. A LangChain application can store previous messages or summaries and provide the relevant history with each new request.

Memory is useful for:
- Chatbots
- Customer support systems
- Personal assistants
- Applications with multiple user sessions


Common Use Cases
LangChain can be used to build:
1. Chatbots that answer questions from company documents.
2. PDF question-answering applications.
3. Educational assistants that summarize lectures and generate quizzes.
4. SQL agents that interact with databases.
5. Customer support agents connected to company systems.
6. Document summarization tools.
7. Structured data extraction systems.
8. Semantic search engines.
9. Data analysis agents.
10. Multi-agent systems for complex tasks.


    
<img width="1567" height="757" alt="image" src="https://github.com/user-attachments/assets/6e99b625-2f81-49c4-a225-ec0bc99e964d" />


LangChain, LangGraph, and LangSmith
- LangChain: Provides the main components for building LLM applications and agents.
- LangGraph: Used for complex, stateful workflows containing multiple steps, conditions, or branches.
- LangSmith: Used to trace, monitor, test, and evaluate LLM applications.



