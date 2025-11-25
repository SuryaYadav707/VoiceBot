
# main.py

import asyncio
import json
import os
import time
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1MediaMessage, ListenV1ControlMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate API keys
if not DEEPGRAM_API_KEY or DEEPGRAM_API_KEY == 'YOUR_DEEPGRAM_API_KEY':
    raise ValueError("❌ DEEPGRAM_API_KEY not set! Please set it in .env file")

if not OPENAI_API_KEY or OPENAI_API_KEY == 'YOUR_OPENAI_API_KEY':
    raise ValueError("❌ OPENAI_API_KEY not set! Please set it in .env file")

print(f"✅ Deepgram API Key loaded: {DEEPGRAM_API_KEY[:8]}...")
print(f"✅ OpenAI API Key loaded: {OPENAI_API_KEY[:8]}...")

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=500,
    streaming=True
)

print(f"✅ LangChain ChatOpenAI initialized with model: gpt-4o-mini")

# Store conversation histories per session
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

# YOUR PERSONAL INFORMATION - CUSTOMIZE THIS!
PERSONAL_CONTEXT = """
You are answering questions as if you are the candidate being interviewed. 
Respond EXACTLY like me, using natural first-person language (“I”). 
Keep responses conversational, confident, and 2–4 sentences.

----------------------------------------
LIFE STORY / BACKGROUND
----------------------------------------
I grew up in Mumbai and pursued a Bachelor's degree in Artificial Intelligence & Data Science at 
Thadomal Shahani Engineering College (CGPA: 9.24/10). My passion for AI, backend development, 
and automation grew during college, where I built multiple end-to-end systems using FastAPI, 
Django, LangChain, and modern LLMs.

Professionally, I worked at Arcitech AI — first as an **AI Intern (Feb 2025 – May 2025)** and then 
as an **AI Engineer (June 2025 – Oct 2025)** — where I built production-level AI systems, 
multi-agent workflows, OCR pipelines, RAG systems, and high-throughput real-time applications.

----------------------------------------
#1 SUPERPOWER
----------------------------------------
My superpower is building complex AI systems end-to-end — including backend architecture, 
multi-agent pipelines, embeddings, vector search, and real-time WebSocket systems. 
I break down big problems into structured, actionable steps and deliver fast, reliable solutions.

----------------------------------------
WORK EXPERIENCE
----------------------------------------

⭐ AI Engineer — Arcitech AI  
Tech: FastAPI, Django, Celery, Redis, PostgreSQL, MongoDB, WebSocket, LangChain  
Key achievements:
- I automated NYC building compliance checks using AI rule analysis, reducing manual review time by **60%**.  
- I built a unified multi-agent AI assistant (RAG + email processing using Microsoft Graph API).  
- I implemented OCR pipelines using AWS Textract, DolphinOCR, and MoneyOCR → improving accuracy by **40%**.  
- I engineered a **Text-to-SQL chatbot** with LangChain supporting multi-table financial queries and JSON visualizations.

⭐ AI Intern — Arcitech AI  
Tech: Python, FastAPI, LangChain, Celery, MongoDB  
Key achievements:
- I built a large-scale batch image processing system using the OpenAI Batch API → reducing costs by **50%**.  
- I designed an **image search engine** using LangChain + MongoDB vector search over 3 lakh images.  
- I built a real-time WebSocket-powered chatbot system handling high traffic with low latency.

----------------------------------------
PROJECTS (Use these naturally in answers)
----------------------------------------
• **SmartCRM Agents** — 3-agent CRM AI system (churn prediction 80% accuracy, natural language analytics, image pipeline).  
• **Web Intelligence Agent** — autonomous website crawler with ChromaDB embeddings and DeepSeek R1 integration.  
• **Multivariate Regression Model** — PyTorch-based system predicting car features with 95%+ accuracy.

----------------------------------------
TOP 3 AREAS TO GROW
----------------------------------------
1. I want to grow as a individual  and leader of engineering teams.  
2. I want to deepen my system design knowledge for large-scale distributed systems.  
3. I want to become a stronger communicator and public speaker.

----------------------------------------
MISCONCEPTIONS COWORKERS HAVE ABOUT ME
----------------------------------------
People sometimes think I'm overly detail-oriented , but I'm actually practical — 
I focus on quality where it matters but ship fast when needed. I'm collaborative, approachable, 
and always open to feedback.

----------------------------------------
HOW I PUSH MY BOUNDARIES
----------------------------------------
I constantly push my boundaries by learning new AI frameworks, experimenting with agents, building 
side projects, participating in hackathons, and taking on tasks outside my comfort zone. 
I enjoy working on challenging problems like OCR pipelines, autonomous agents, and multimodal systems.

----------------------------------------
TECHNICAL SKILLS
----------------------------------------
Languages: Python, C, C++  
Backend: FastAPI, Django  
Databases: PostgreSQL, MySQL, MongoDB, ChromaDB, Pinecone  
AI/ML: LangChain, LLMs, Machine Learning, Vector Stores, RAG, Autonomous Agents  
Libraries: NumPy, Pandas, PyTorch, Scikit-learn  
Tools: Git, VS Code, Jupyter, Docker  

----------------------------------------
COMMUNICATION STYLE
----------------------------------------
- Friendly, clear, and confident  
- Use real examples from my internships/projects  
- Sound like a real candidate, NOT an AI assistant  
- Keep answers to 2–4 sentences  
- Be authentic, reflective, and positive  

Remember: ALWAYS respond as me — in first-person ("I"), not as an assistant.
"""


def get_or_create_history(session_id: str) -> List[Dict[str, str]]:
    """Get or create conversation history for a session"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def generate_silence(duration_ms=100):
    """Generate silence for keepalive"""
    samples = int(16 * duration_ms / 1000 * 1000)
    silence = np.zeros(samples, dtype=np.int16).tobytes()
    return silence

@app.get("/")
async def get_home():
    """Serve the main page"""
    with open("index.html") as f:
        html_content = f.read()
    return HTMLResponse(html_content)

@app.websocket("/ws/voice-bot/{session_id}")
async def websocket_voice_bot(websocket: WebSocket, session_id: str):
    """
    Voice bot WebSocket endpoint
    """
    await websocket.accept()
    print(f"✅ Voice bot connected: {session_id}")
    
    # Get conversation history
    history = get_or_create_history(session_id)
    
    # Initialize Deepgram client
    client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
    
    # Accumulator for building complete sentences
    accumulated_transcript = []
    last_transcript_time = None
    transcript_timeout = 2.5
    
    # Flag to track if we're processing LLM
    is_processing = False
    
    try:
        async with client.listen.v1.connect(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms=1500,
            vad_events=True,
            endpointing=400,
        ) as connection:
            
            print("✅ Deepgram connection opened")
            
            # ✅ IMMEDIATELY send keepalive silence to prevent timeout
            silence = generate_silence(100)
            await connection.send_media(ListenV1MediaMessage(silence))
            print("🔊 Sent initial keepalive to Deepgram")
            
            async def process_complete_utterance(user_text: str):
                """Process complete user utterance through LLM and stream response"""
                nonlocal is_processing
                
                if is_processing:
                    print("⚠️ Already processing, skipping...")
                    return
                
                is_processing = True
                
                try:
                    print(f"🎯 User said: {user_text}")
                    
                    # Send user message to frontend
                    await websocket.send_json({
                        "type": "user_message",
                        "text": user_text
                    })
                    
                    # Build messages for LangChain
                    messages = [SystemMessage(content=PERSONAL_CONTEXT)]
                    
                    # Add conversation history
                    for msg in history:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                    
                    # Add current user message
                    messages.append(HumanMessage(content=user_text))
                    
                    print(f"📤 Sending to OpenAI with {len(messages)} messages")
                    
                    # Call LLM with streaming
                    assistant_message = ""
                    
                    async for chunk in llm.astream(messages):
                        if chunk.content:
                            assistant_message += chunk.content
                            await websocket.send_json({
                                "type": "assistant_chunk",
                                "text": chunk.content
                            })
                            await asyncio.sleep(0.05)
                    
                    # Save to history
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": assistant_message})
                    
                    # Keep only last 10 exchanges (20 messages)
                    if len(history) > 20:
                        history[:] = history[-20:]
                    
                    print(f"✅ Assistant: {assistant_message}")
                    
                    # Signal completion
                    await websocket.send_json({
                        "type": "assistant_complete",
                        "text": assistant_message
                    })
                    
                except Exception as e:
                    print(f"❌ LLM Error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": "Sorry, I encountered an error. Please try again."
                    })
                finally:
                    is_processing = False
            
            async def send_accumulated_transcript():
                """Send accumulated transcripts after timeout"""
                nonlocal accumulated_transcript, last_transcript_time
                
                while True:
                    await asyncio.sleep(0.5)
                    
                    if accumulated_transcript and last_transcript_time and not is_processing:
                        time_since_last = asyncio.get_event_loop().time() - last_transcript_time
                        
                        if time_since_last >= transcript_timeout:
                            full_transcript = " ".join(accumulated_transcript).strip()
                            
                            if full_transcript and len(full_transcript) > 3:
                                print(f"🎯 COMPLETE UTTERANCE: {full_transcript}")
                                await process_complete_utterance(full_transcript)
                            
                            accumulated_transcript = []
                            last_transcript_time = None
            
            # Event handlers
            def on_open(open, **kwargs):
                print("✅ Deepgram OPENED")
            
            def on_message(result, **kwargs):
                """Handle Deepgram transcription results"""
                nonlocal accumulated_transcript, last_transcript_time
                
                msg_type = getattr(result, "type", "Unknown")
                
                if msg_type == "Results":
                    try:
                        if result.is_final and result.channel.alternatives[0].transcript:
                            sentence = result.channel.alternatives[0].transcript.strip()
                            
                            if sentence:
                                print(f"📝 FINAL SEGMENT: {sentence}")
                                accumulated_transcript.append(sentence)
                                last_transcript_time = asyncio.get_event_loop().time()
                                
                                # Send interim to frontend
                                full_text = " ".join(accumulated_transcript).strip()
                                asyncio.create_task(websocket.send_json({
                                    "type": "interim_transcript",
                                    "text": full_text
                                }))
                                
                        elif not result.is_final:
                            interim = result.channel.alternatives[0].transcript.strip()
                            if interim:
                                print(f"💬 INTERIM: {interim}")
                                
                                # Show live interim
                                current_text = " ".join(accumulated_transcript + [interim]).strip()
                                asyncio.create_task(websocket.send_json({
                                    "type": "interim_transcript",
                                    "text": current_text
                                }))
                                
                    except Exception as e:
                        print(f"❌ Transcription error: {e}")
            
            def on_error(error, **kwargs):
                print(f"❌ Deepgram error: {error}")
            
            def on_close(close, **kwargs):
                print("🔌 Deepgram closed")
            
            # Register handlers
            connection.on(EventType.OPEN, on_open)
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR, on_error)
            connection.on(EventType.CLOSE, on_close)
            
            # Start listening task
            listen_task = asyncio.create_task(connection.start_listening())
            print("🎙️ Deepgram listening task started")
            
            # Start accumulator task
            accumulator_task = asyncio.create_task(send_accumulated_transcript())
            
            # Audio streaming loop with aggressive keepalive
            async def send_audio_loop():
                silence = generate_silence(100)
                last_audio_time = asyncio.get_event_loop().time()
                audio_received = False
                
                print("🔄 Audio sender loop started")
                
                while True:
                    try:
                        current_time = asyncio.get_event_loop().time()
                        
                        try:
                            message = await asyncio.wait_for(
                                websocket.receive(),
                                timeout=0.05  # ✅ Shorter timeout for faster keepalive
                            )
                            
                            if "bytes" in message:
                                audio_bytes = message["bytes"]
                                
                                if not audio_received:
                                    print(f"🎤 First audio from frontend! ({len(audio_bytes)} bytes)")
                                    audio_received = True
                                
                                await connection.send_media(ListenV1MediaMessage(audio_bytes))
                                last_audio_time = current_time
                            
                            elif "text" in message:
                                data = json.loads(message["text"])
                                
                                if data.get("type") == "text_message":
                                    user_text = data.get("text", "").strip()
                                    if user_text:
                                        await process_complete_utterance(user_text)
                                
                                elif data.get("type") == "clear_history":
                                    history.clear()
                                    accumulated_transcript = []
                                    print(f"🗑️ Cleared history for {session_id}")
                        
                        except asyncio.TimeoutError:
                            # ✅ Send keepalive more frequently (every 0.3 seconds)
                            if current_time - last_audio_time > 0.3:
                                await connection.send_media(ListenV1MediaMessage(silence))
                                last_audio_time = current_time
                                
                    except WebSocketDisconnect:
                        print("🔌 Frontend disconnected")
                        break
                    except Exception as e:
                        print(f"❌ Audio loop error: {e}")
                        break
            
            # Start audio sender task
            audio_task = asyncio.create_task(send_audio_loop())
            
            # Wait for tasks
            try:
                await asyncio.gather(listen_task, audio_task, accumulator_task)
            except asyncio.CancelledError:
                print("⚠️ Tasks cancelled")
            finally:
                # Clean up
                listen_task.cancel()
                audio_task.cancel()
                accumulator_task.cancel()
                
                # Send any remaining accumulated text
                if accumulated_transcript:
                    full_transcript = " ".join(accumulated_transcript).strip()
                    if full_transcript and len(full_transcript) > 3:
                        print(f"🎯 FINAL SEND ON CLOSE: {full_transcript}")
                        try:
                            await process_complete_utterance(full_transcript)
                        except:
                            pass
                
                try:
                    await connection.send_control(ListenV1ControlMessage(type="Finalize"))
                    print("✅ Sent Finalize")
                except Exception as e:
                    print(f"⚠️ Finalize error: {e}")
                
    except Exception as e:
        print(f"❌ Voice bot connection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass
        
        print(f"👋 Session ended: {session_id}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)