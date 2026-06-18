import requests

url = "http://52.206.209.141:8002/api/tags"

try:
    response = requests.get(url, timeout=10)

    print("Status:", response.status_code)
    print(response.text[:500])

except Exception as e:
    print("ERROR:", e)