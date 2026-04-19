from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import pdfplumber
import docx
import io
import os

app = FastAPI(title="Resume Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")
    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")

@app.post("/optimize")
async def optimize_resume(
    file: UploadFile = File(...),
    job_title: str = Form(""),
    job_description: str = Form(""),
):
    file_bytes = await file.read()
    resume_text = extract_text(file_bytes, file.filename)

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    job_context = ""
    if job_title or job_description:
        job_context = f"\n\nTarget Role: {job_title}\nJob Description:\n{job_description}"

    prompt = f"""You are an expert resume coach and ATS optimization specialist.

Analyze the resume below and return a JSON object with exactly these keys:
- "score": integer 0-100 (current ATS/quality score)
- "optimized_resume": full rewritten resume as plain text, with improved bullet points using strong action verbs + quantified impact, better keyword density for ATS, cleaner structure
- "improvements": array of 5 specific improvement strings made
- "keywords_added": array of keywords added for ATS
- "tips": array of 3 quick tips for this candidate

Resume:
{resume_text}
{job_context}

Return ONLY valid JSON. No markdown, no explanation."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    result = json.loads(raw)
    return JSONResponse(content=result)

@app.get("/health")
def health():
    return {"status": "ok"}
