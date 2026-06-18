from ollama import Client

client = Client(
    host="http://52.206.209.141:8002"
)

def generate_embedding(text):

    response = client.embeddings(
        model="nomic-embed-text:latest",
        prompt=text
    )

    return response["embedding"]