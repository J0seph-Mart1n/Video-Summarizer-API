import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

# Initialize FastAPI
app = FastAPI(title="PhiData YouTube Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Input/Output ---
class VideoRequest(BaseModel):
    url: str

class SummaryResponse(BaseModel):
    video_id: str
    summary: str

# --- Helper Function: Extract Video ID ---
def get_video_id(url: str) -> Optional[str]:
    """
    Extracts video ID from standard URL, Short link, or Share link.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',       # Standard & Shorts
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'     # Share URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- Core Logic ---
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_video(request: VideoRequest):
    
    # 1. Extract ID
    video_id = get_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # 2. Fetch Transcript
    try:
        # Returns a list of dicts: [{'text': 'hello', 'start': 0.0, ...}, ...]
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=PROXY_USERNAME,
                proxy_password=PROXY_PASSWORD,
            )
        )
        transcript_list = ytt_api.fetch(video_id)
        
        # Combine text (Fixing the dictionary access here)
        caption_text = " ".join([snippet['text'] for snippet in transcript_list])
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Transcript unavailable: {str(e)}")

    # 3. Define Agents (Instantiated per request to ensure fresh context)
    
    # A. The Transcript Analyst
    analyst = Agent(
        name="Transcript Analyst",
        role="Extracts logical segments and raw data from captions",
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Clean filler words from the transcript.",
            "Divide the content into logical chapters based on the flow of conversation.",
            "Extract all specific entities like tools, links, or names mentioned."
        ],
    )

    # B. The Insight Miner
    miner = Agent(
        name="Insight Miner",
        role="Identifies deep insights and the 'why' behind the video",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo()],
        instructions=[
            "Identify the top 3-5 unique insights or 'Gold Nuggets'.",
            "Determine the creator's tone and the target audience.",
            "Highlight the primary problem and solution discussed."
        ],
    )

    # C. The Lead Editor (The Orchestrator)
    editor = Agent(
        name="Lead Editor",
        team=[analyst, miner],
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Receive the analysis from the Analyst and Miner.",
            "Format the final output into professional Markdown.",
            "Start with a 'TL;DR' section.",
            "Follow with a 'Detailed Breakdown' using headers.",
            "End with a 'Key Takeaways' checklist.",
            "Ensure the summary is concise and removes any fluff."
        ],
        markdown=True,
    )

    # 4. Run the Agent Team
    try:
        # stream=False waits for the full generation
        response = editor.run(caption_text, stream=False)
        
        # Return the content content
        return SummaryResponse(
            video_id=video_id,
            summary=response.content
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Error: {str(e)}")

# --- Health Check ---
@app.get("/")
def home():
    return {"message": "YouTube Summarizer API is running"}

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/testing_api/")
async def create_item(item: Item):
    return item


# video_id = "JDYtbVxtBhw"

# ytt_api = YouTubeTranscriptApi()
# transcript = ytt_api.fetch(video_id)

# #Convert to plain text
# caption_text = " ".join([snippet.text for snippet in transcript])

#print(caption_text)


# Get the response in a variable
# run: RunResponse = editor.run(caption_text)
# print(run.content)


# Print the response in the terminal
#editor.print_response(caption_text)