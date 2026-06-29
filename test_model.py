import requests

session = requests.Session()
session.trust_env = False

response = session.post(
    "http://52.206.209.141:8002/api/generate",
    json={
        "model": "llama3.1:latest",
        "prompt": "hello",
        "stream": False
    },
    timeout=60
)

print("STATUS:", response.status_code)
print("BODY:", response.text[:1000])