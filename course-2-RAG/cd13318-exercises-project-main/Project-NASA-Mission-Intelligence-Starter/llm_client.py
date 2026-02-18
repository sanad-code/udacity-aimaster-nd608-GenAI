from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """You are a professional NASA space mission expert assistant. 
                        You have deep expertise in Apollo missions (11,13), the Challenger disaster.
                        You will get a context that is retrieved from a knowledge base, and you will use that context to answer user questions.
                        Your communication style:
                        - Clear and concise
                        - Professional and informative
                        When answering questions:
                        - Use the provided context documents to ground your responses.
                        - Be accurate and cite specific details from the source material, always referencing the source document when possible.
                        - If the context doesn't contain enough information, say so honestly and try to clarify the user's question if needed.
                        - Provide clear, well-organized responses"""

    # Set context in messages
    messages = [{"role": "system", "content": system_prompt}]

    # Only add context if it exists and is not empty
    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following retrieved context to answer the user's question:\n\n{context}"
        })

    # Add chat history, I will only include user(input) and assistant(output) messages from the history to the current conversation.
    # I intentionally exclude any system messages from the history to minmize token usage and more system messages are usless.
    for msg in conversation_history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # Send request to OpenAI
    # I am selecting a balanced temperature of 0.7 to be able to fullfill both accuracy for the astronaut and engagement for curiose historians as request in project requirements.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    # Return response
    return response.choices[0].message.content