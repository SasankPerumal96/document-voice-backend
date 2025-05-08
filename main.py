from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
from openai import OpenAI
import requests
import io

# Utility: Split text into overlapping chunks
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"message": "Accessible AI Backend is live 🎉"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        doc.close()

        chunks = split_text(full_text)
        with open("pdf_chunks.txt", "w") as f:
            for c in chunks:
                f.write(c.replace("\n", " ") + "\n---CHUNK---\n")

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "message": "Text chunked and saved"
        }

    except Exception as e:
        return {"error": str(e)}

class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(payload: Question):
    question = payload.question
    try:
        with open("pdf_chunks.txt", "r") as f:
            raw_chunks = f.read().split("---CHUNK---")
            chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

        prompt = """You are a helpful assistant. Based on the following document snippet and user question, return a relevance score from 0 to 10.

Document:
{chunk}

Question: {question}

Respond only with the number."""

        scored = []
        for chunk in chunks[:10]:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Rate relevance 0–10."},
                    {"role": "user", "content": prompt.replace("{chunk}", chunk)}
                ]
            )
            score = response.choices[0].message.content.strip()
            try:
                score_val = int(score)
            except:
                score_val = 0
            scored.append((score_val, chunk))

        best_chunk = max(scored, key=lambda x: x[0])[1]

        final_prompt = f"""Use the following context to answer the question:

Context:
{best_chunk}

Question: {question}"""

        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )

        return {"answer": final_response.choices[0].message.content.strip()}

    except Exception as e:
        return {"error": str(e)}

@app.post("/tts/")
async def text_to_speech(question: str = File(...)):
    try:
        api_key = os.getenv("RIME_API_KEY")

        response = requests.post(
            "https://users.rime.ai/v1/rime-tts",
            headers={
                "Accept": "audio/mp3",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "text": question,
                "speaker": "celeste",
                "modelId": "arcana"
            }
        )

        if response.status_code != 200:
            return {"error": response.text}

        audio_stream = io.BytesIO(response.content)
        return StreamingResponse(audio_stream, media_type="audio/mpeg")

    except Exception as e:
        return {"error": str(e)}
