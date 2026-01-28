import os
from dotenv import load_dotenv
load_dotenv(override=False)

from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

endpoint = os.environ["AZURE_AI_ENDPOINT"].rstrip("/")
if endpoint.endswith("/api/projects") or "/api/projects/" in endpoint:

    endpoint = endpoint.split("/api/projects/")[0] + "/models"
elif endpoint.endswith("/models"):
    pass
else:
    # if base endpoint
    endpoint = endpoint + "/models"

client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(os.environ["AZURE_AI_API_KEY"]),
    model=os.environ["EMBED_DEPLOYMENT"],
)

try:
    r = client.embed(input=["ping"])
    print("OK dims =", len(r.data[0].embedding))
except Exception as e:
    print("EXC:", type(e).__name__, str(e))
    # попытаться вытащить статус-код/тело
    resp = getattr(e, "response", None)
    if resp is not None:
        print("status:", getattr(resp, "status_code", None))
        try:
            print("body:", resp.text())
        except Exception:
            pass
