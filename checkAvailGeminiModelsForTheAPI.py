import google.generativeai as genai

# Configure API
genai.configure(api_key="AIzaSyDK1abnhrks0dRp0xMsHbyv6kiaUVy4538")

# List available models
available_models = genai.list_models()
for model in available_models:
    print(model.name)
