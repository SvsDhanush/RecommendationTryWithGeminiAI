import requests

# Replace with your Hugging Face API key
api_key = "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"

def get_embeddings(text):
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error details:", response.text)
        return f"Error: {response.status_code}"

# Test the function
text = "This is an example sentence."
embeddings = get_embeddings(text)
print(embeddings)
