import ollama

def chatbot_reply(query, disease_info=None, crop=None, soil_data=None):

    if not query:
        return "Please ask a valid question."

    # 🔥 HARD SAFETY (prevents null)
    if not disease_info:
        return "No disease data available."

    context = f"""
Disease: {disease_info.get('name')}
Cause: {disease_info.get('cause')}

Treatment:
Medicine: {disease_info.get('treatment', {}).get('en', {}).get('medicine')}
Dosage: {disease_info.get('treatment', {}).get('en', {}).get('dosage')}
Frequency: {disease_info.get('treatment', {}).get('en', {}).get('frequency')}
"""

    system_prompt = """
You are a helpful agriculture assistant.

- Answer based on given data.
- Keep it short and clear.
"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context + "\nUser Question: " + query}
            ],
            options={
                "temperature": 0.3,
                "num_predict": 100
            }
        )

        # 🔥 SAFE RETURN (fixes null issue)
        if not response or 'message' not in response:
            return "AI response error"

        content = response['message'].get('content', None)

        if not content:
            return "No response generated"

        return content.strip()

    except Exception as e:
        print("Ollama Error:", e)
        return "⚠️ AI not available"