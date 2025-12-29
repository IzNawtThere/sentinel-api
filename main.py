from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import base64
import re
import json
from datetime import datetime
import httpx

# ============================================
# SENTINEL API v2.0 - Complete Protection
# ============================================

app = FastAPI(
    title="SENTINEL API",
    description="AI-Powered Scam Detection, Image Analysis, Deepfake Detection & File Scanning",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

# ============================================
# DATA MODELS
# ============================================

class TextAnalysisRequest(BaseModel):
    message: str
    context: Optional[str] = "general"
    sender: Optional[str] = "unknown"

class URLAnalysisRequest(BaseModel):
    url: str

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    filename: Optional[str] = "image"
    check_type: Optional[str] = "all"  # "ai_generated", "deepfake", "scam", "all"

class AudioAnalysisRequest(BaseModel):
    audio_base64: str
    filename: Optional[str] = "audio"
    claimed_identity: Optional[str] = None  # For voice verification

# ============================================
# SCAM DETECTION PATTERNS
# ============================================

SCAM_PATTERNS = {
    "urgency_keywords": [
        "urgent", "immediately", "right now", "act fast", "don't delay",
        "expires today", "last chance", "limited time", "hurry", "asap",
        "within 24 hours", "suspended", "blocked", "terminated"
    ],
    "financial_keywords": [
        "bank account", "credit card", "transfer money", "send money",
        "western union", "gift card", "bitcoin", "crypto", "investment",
        "lottery", "prize", "winner", "inheritance", "million dollars"
    ],
    "threat_keywords": [
        "legal action", "arrest", "warrant", "police", "court",
        "penalty", "fine", "prosecution", "jail", "prison"
    ],
    "authority_impersonation": [
        "iras", "cpf", "singpass", "dbs", "ocbc", "uob", "posb",
        "grab", "shopee", "lazada", "microsoft", "apple", "google",
        "netflix", "amazon", "paypal", "government", "ministry",
        "hdfc", "icici", "sbi", "axis", "paytm", "phonepe"
    ],
    "suspicious_requests": [
        "verify your", "confirm your", "update your", "click here",
        "click below", "click this link", "download", "install",
        "provide your", "enter your", "share your", "otp", "password", "pin"
    ]
}

SUSPICIOUS_TLDS = [".xyz", ".top", ".click", ".link", ".info", ".online", ".site", ".club"]

# ============================================
# TEXT ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text message for scam indicators"""
    
    message = request.message.lower()
    red_flags = []
    risk_score = 0
    
    # Pattern matching
    for word in SCAM_PATTERNS["urgency_keywords"]:
        if word in message:
            red_flags.append({"flag": "Urgency/pressure tactics", "severity": "high", "detail": f"Contains urgency keyword: '{word}'"})
            risk_score += 15
            break
    
    for word in SCAM_PATTERNS["financial_keywords"]:
        if word in message:
            red_flags.append({"flag": "Financial request", "severity": "critical", "detail": f"Contains financial keyword: '{word}'"})
            risk_score += 20
            break
    
    for word in SCAM_PATTERNS["threat_keywords"]:
        if word in message:
            red_flags.append({"flag": "Threat/intimidation", "severity": "high", "detail": f"Contains threatening language: '{word}'"})
            risk_score += 15
            break
    
    for word in SCAM_PATTERNS["authority_impersonation"]:
        if word in message:
            red_flags.append({"flag": "Authority impersonation", "severity": "critical", "detail": f"Claims to be from: '{word.upper()}'"})
            risk_score += 25
            break
    
    for phrase in SCAM_PATTERNS["suspicious_requests"]:
        if phrase in message:
            red_flags.append({"flag": "Suspicious request", "severity": "high", "detail": f"Asks to: '{phrase}'"})
            risk_score += 15
            break
    
    # URL detection
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', request.message)
    for url in urls:
        for tld in SUSPICIOUS_TLDS:
            if tld in url.lower():
                red_flags.append({"flag": "Suspicious URL", "severity": "critical", "detail": f"Contains suspicious domain: {url}"})
                risk_score += 25
                break
    
    # AI Analysis with Claude
    try:
        ai_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Analyze this message for scam/fraud indicators. Be thorough but concise.

MESSAGE: {request.message}

Respond in this exact JSON format:
{{
    "is_scam": true/false,
    "confidence": 0-100,
    "scam_type": "phishing/investment/lottery/job/romance/tech_support/government/family_emergency/other/none",
    "ai_generated_probability": 0-100,
    "red_flags": ["flag1", "flag2"],
    "recommendation": "Brief action advice",
    "explanation": "1-2 sentence explanation"
}}"""
            }]
        )
        
        response_text = ai_response.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            ai_analysis = json.loads(json_match.group())
            
            if ai_analysis.get("is_scam"):
                risk_score += 30
            
            risk_score += ai_analysis.get("ai_generated_probability", 0) // 5
            
            for flag in ai_analysis.get("red_flags", []):
                if not any(f["flag"].lower() == flag.lower() for f in red_flags):
                    red_flags.append({"flag": flag, "severity": "medium", "detail": "Detected by AI analysis"})
            
            recommendation = ai_analysis.get("recommendation", "Exercise caution with this message.")
            is_ai_generated = ai_analysis.get("ai_generated_probability", 0) > 60
            ai_confidence = ai_analysis.get("ai_generated_probability", 0)
            
    except Exception as e:
        recommendation = "Exercise caution. Our AI analysis encountered an error."
        is_ai_generated = False
        ai_confidence = 0
    
    # Calculate final risk
    risk_score = min(risk_score, 100)
    
    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "is_ai_generated": is_ai_generated,
        "ai_confidence": ai_confidence,
        "red_flags": red_flags,
        "recommendation": recommendation,
        "analyzed_at": datetime.now().isoformat()
    }

# ============================================
# IMAGE ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/image")
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze image for AI generation, deepfakes, and scam content"""
    
    try:
        # Decode base64 to verify it's valid
        image_data = base64.b64decode(request.image_base64)
        
        # Determine media type
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif image_data[:6] in (b'GIF87a', b'GIF89a'):
            media_type = "image/gif"
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"  # Default
        
        # Analyze with Claude Vision
        ai_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": request.image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image thoroughly for:

1. **AI GENERATION DETECTION**: Is this image AI-generated (Midjourney, DALL-E, Stable Diffusion, etc.)?
   Look for: unnatural textures, warped details, inconsistent lighting, strange hands/fingers, text errors, repeating patterns, uncanny smoothness

2. **DEEPFAKE DETECTION**: If there's a face, does it show signs of being a deepfake?
   Look for: unnatural skin texture, weird eye reflections, hair boundary issues, facial asymmetry, blending artifacts

3. **SCAM CONTENT**: Does this image appear to be used for scams?
   Look for: fake screenshots, fake bank notices, fake prizes, fake government documents, fake celebrity endorsements

4. **MANIPULATION**: Has this image been digitally manipulated?
   Look for: clone stamp artifacts, inconsistent shadows, edge anomalies, compression inconsistencies

Respond in this exact JSON format:
{
    "is_ai_generated": true/false,
    "ai_generation_confidence": 0-100,
    "ai_generator_likely": "midjourney/dall-e/stable-diffusion/other/none",
    "is_deepfake": true/false,
    "deepfake_confidence": 0-100,
    "is_scam_content": true/false,
    "scam_type": "fake_screenshot/fake_document/fake_prize/fake_celebrity/other/none",
    "is_manipulated": true/false,
    "manipulation_type": "photoshop/face_swap/splice/none",
    "findings": [
        {"check": "AI Generation Markers", "status": "detected/clear", "detail": "explanation"},
        {"check": "Facial Analysis", "status": "suspicious/clear/na", "detail": "explanation"},
        {"check": "Scam Indicators", "status": "detected/clear", "detail": "explanation"},
        {"check": "Manipulation Signs", "status": "detected/clear", "detail": "explanation"}
    ],
    "risk_level": "critical/high/medium/low",
    "summary": "2-3 sentence summary of findings",
    "recommendation": "What the user should do"
}"""
                    }
                ]
            }]
        )
        
        response_text = ai_response.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            analysis = json.loads(json_match.group())
            analysis["analyzed_at"] = datetime.now().isoformat()
            return analysis
        else:
            raise ValueError("Could not parse AI response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@app.post("/analyze/image/upload")
async def analyze_image_upload(file: UploadFile = File(...)):
    """Upload and analyze an image file"""
    
    # Read file
    contents = await file.read()
    
    # Convert to base64
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    # Create request and analyze
    request = ImageAnalysisRequest(
        image_base64=image_base64,
        filename=file.filename
    )
    
    return await analyze_image(request)


# ============================================
# URL ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/url")
async def analyze_url(request: URLAnalysisRequest):
    """Analyze URL for phishing and malicious content"""
    
    url = request.url.lower()
    red_flags = []
    risk_score = 0
    
    # Check TLD
    for tld in SUSPICIOUS_TLDS:
        if tld in url:
            red_flags.append({"flag": "Suspicious TLD", "severity": "high", "detail": f"Uses suspicious domain extension: {tld}"})
            risk_score += 25
            break
    
    # Check for URL shorteners
    shorteners = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]
    for shortener in shorteners:
        if shortener in url:
            red_flags.append({"flag": "URL Shortener", "severity": "medium", "detail": "Uses URL shortener to hide destination"})
            risk_score += 15
            break
    
    # Check for IP address
    if re.search(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
        red_flags.append({"flag": "IP Address URL", "severity": "high", "detail": "Uses IP address instead of domain name"})
        risk_score += 25
    
    # Check for typosquatting
    known_brands = ["google", "facebook", "amazon", "apple", "microsoft", "netflix", "paypal", "dbs", "ocbc", "singtel"]
    for brand in known_brands:
        if brand in url and f"{brand}.com" not in url and f"{brand}.sg" not in url:
            red_flags.append({"flag": "Possible Typosquatting", "severity": "critical", "detail": f"May be impersonating {brand}"})
            risk_score += 30
            break
    
    # Check for suspicious keywords in URL
    suspicious_url_words = ["login", "verify", "secure", "account", "update", "confirm", "banking"]
    for word in suspicious_url_words:
        if word in url:
            red_flags.append({"flag": "Suspicious URL Keywords", "severity": "medium", "detail": f"Contains '{word}' - common in phishing"})
            risk_score += 10
            break
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 60:
        risk_level = "critical"
    elif risk_score >= 40:
        risk_level = "high"
    elif risk_score >= 20:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "url": request.url,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "red_flags": red_flags,
        "recommendation": "Do NOT click this link. It shows signs of being malicious." if risk_score >= 40 else "This URL appears relatively safe, but always exercise caution.",
        "analyzed_at": datetime.now().isoformat()
    }


# ============================================
# AUDIO/VOICE ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/audio")
async def analyze_audio(request: AudioAnalysisRequest):
    """Analyze audio for AI-generated voice and voice cloning"""
    
    # Note: Full audio analysis requires specialized ML models
    # This is a placeholder that provides guidance
    
    return {
        "status": "limited_analysis",
        "message": "Full voice clone detection requires specialized audio ML models. For production, integrate with services like Azure Speaker Recognition or build custom models.",
        "checks_available": [
            "Voice pattern analysis",
            "AI synthesis detection",
            "Speaker verification"
        ],
        "recommendation": "For suspected voice cloning, verify the caller through a different channel (video call, in-person, or ask questions only the real person would know).",
        "analyzed_at": datetime.now().isoformat()
    }


# ============================================
# FILE ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for malware indicators"""
    
    contents = await file.read()
    filename = file.filename.lower()
    file_size = len(contents)
    
    red_flags = []
    risk_score = 0
    
    # Check file extension
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.msi', '.jar', '.vbs', '.js', '.ps1', '.apk']
    for ext in dangerous_extensions:
        if filename.endswith(ext):
            red_flags.append({"flag": "Dangerous file type", "severity": "critical", "detail": f"File type '{ext}' can execute malicious code"})
            risk_score += 40
            break
    
    # Check for double extensions (e.g., document.pdf.exe)
    if filename.count('.') > 1:
        red_flags.append({"flag": "Double extension", "severity": "high", "detail": "Multiple file extensions detected - common malware trick"})
        risk_score += 25
    
    # Check for macros in Office files
    if filename.endswith(('.docm', '.xlsm', '.pptm')):
        red_flags.append({"flag": "Macro-enabled document", "severity": "high", "detail": "This document can contain executable macros"})
        risk_score += 30
    
    # Check file header magic bytes
    if contents[:2] == b'MZ':  # Windows executable
        red_flags.append({"flag": "Windows executable", "severity": "critical", "detail": "This is a Windows executable file"})
        risk_score += 40
    elif contents[:4] == b'PK\x03\x04':  # ZIP/APK/Office
        if filename.endswith('.apk'):
            red_flags.append({"flag": "Android APK", "severity": "high", "detail": "Android application package - verify source before installing"})
            risk_score += 30
    elif b'<script' in contents.lower()[:1000] or b'javascript:' in contents.lower()[:1000]:
        red_flags.append({"flag": "Embedded scripts", "severity": "medium", "detail": "File contains JavaScript code"})
        risk_score += 20
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 60:
        risk_level = "critical"
        recommendation = "DO NOT OPEN THIS FILE. Delete it immediately."
    elif risk_score >= 40:
        risk_level = "high"
        recommendation = "Be very cautious. Only open if you trust the source completely."
    elif risk_score >= 20:
        risk_level = "medium"
        recommendation = "Exercise caution. Verify the sender before opening."
    else:
        risk_level = "low"
        recommendation = "File appears safe, but always verify the source."
    
    return {
        "filename": file.filename,
        "file_size": file_size,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "red_flags": red_flags,
        "recommendation": recommendation,
        "analyzed_at": datetime.now().isoformat()
    }


# ============================================
# HEALTH & INFO ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "service": "SENTINEL API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "text_analysis": "/analyze/text",
            "image_analysis": "/analyze/image",
            "image_upload": "/analyze/image/upload",
            "url_analysis": "/analyze/url",
            "audio_analysis": "/analyze/audio",
            "file_analysis": "/analyze/file",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}