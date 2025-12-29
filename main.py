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
import os

# ============================================
# SENTINEL API v2.1 - Enhanced AI Detection
# ============================================

app = FastAPI(
    title="SENTINEL API",
    description="AI-Powered Scam Detection with Real AI Image/Video Detection",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

# Hugging Face API for AI image detection
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Optional: for higher rate limits

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
    check_type: Optional[str] = "all"

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
# HUGGING FACE AI DETECTION
# ============================================

async def detect_ai_image_huggingface(image_bytes: bytes) -> dict:
    """Use Hugging Face models to detect AI-generated images"""
    
    results = {
        "ai_detection_models": [],
        "average_ai_probability": 0,
        "is_likely_ai": False
    }
    
    # List of AI detection models to try
    models = [
        "umm-maybe/AI-image-detector",
        "Organika/sdxl-detector", 
    ]
    
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    probabilities = []
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for model in models:
            try:
                response = await http_client.post(
                    f"{HF_API_URL}{model}",
                    headers=headers,
                    content=image_bytes
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse the result - format varies by model
                    ai_prob = 0
                    human_prob = 0
                    
                    if isinstance(result, list):
                        for item in result:
                            label = item.get("label", "").lower()
                            score = item.get("score", 0)
                            
                            if "artificial" in label or "ai" in label or "generated" in label or "fake" in label:
                                ai_prob = score
                            elif "human" in label or "real" in label or "natural" in label:
                                human_prob = score
                        
                        # If we only got one label, infer the other
                        if ai_prob > 0 and human_prob == 0:
                            human_prob = 1 - ai_prob
                        elif human_prob > 0 and ai_prob == 0:
                            ai_prob = 1 - human_prob
                    
                    ai_percentage = round(ai_prob * 100, 1)
                    probabilities.append(ai_percentage)
                    
                    results["ai_detection_models"].append({
                        "model": model.split("/")[-1],
                        "ai_probability": ai_percentage,
                        "human_probability": round(human_prob * 100, 1),
                        "status": "success"
                    })
                else:
                    results["ai_detection_models"].append({
                        "model": model.split("/")[-1],
                        "status": "error",
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                results["ai_detection_models"].append({
                    "model": model.split("/")[-1],
                    "status": "error", 
                    "error": str(e)
                })
    
    # Calculate average probability
    if probabilities:
        results["average_ai_probability"] = round(sum(probabilities) / len(probabilities), 1)
        results["is_likely_ai"] = results["average_ai_probability"] > 50
    
    return results


# ============================================
# CLAUDE VISION ANALYSIS
# ============================================

async def analyze_image_with_claude(image_base64: str, media_type: str) -> dict:
    """Use Claude for deepfake, scam, and manipulation detection"""
    
    try:
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
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image for:

1. **DEEPFAKE DETECTION** (if faces present):
   - Unnatural skin texture or blurring
   - Inconsistent lighting on face
   - Weird eye reflections or gaze
   - Hair boundary issues
   - Facial feature warping
   - Blending artifacts around face edges

2. **SCAM CONTENT DETECTION**:
   - Fake bank notices or screenshots
   - Fake prize/lottery announcements
   - Fake government documents
   - Fake celebrity endorsements
   - Fake payment confirmations
   - Phishing content

3. **MANIPULATION DETECTION**:
   - Clone stamp or healing artifacts
   - Inconsistent shadows or lighting
   - Edge anomalies from cutting/pasting
   - Compression inconsistencies
   - Text that looks added/edited

4. **AI GENERATION MARKERS** (secondary check):
   - Unnatural textures or patterns
   - Warped or impossible geometry
   - Strange hands, fingers, teeth
   - Text/writing errors
   - Repeating patterns
   - Uncanny smoothness

Respond in this exact JSON format:
{
    "has_face": true/false,
    "deepfake_analysis": {
        "is_deepfake": true/false,
        "confidence": 0-100,
        "indicators": ["indicator1", "indicator2"]
    },
    "scam_content": {
        "is_scam": true/false,
        "scam_type": "fake_screenshot/fake_document/fake_prize/phishing/none",
        "confidence": 0-100,
        "details": "explanation"
    },
    "manipulation": {
        "is_manipulated": true/false,
        "manipulation_type": "photoshop/splice/clone/text_edit/none",
        "confidence": 0-100,
        "indicators": ["indicator1", "indicator2"]
    },
    "ai_indicators": {
        "has_ai_artifacts": true/false,
        "artifacts_found": ["artifact1", "artifact2"],
        "notes": "explanation"
    },
    "overall_risk": "critical/high/medium/low",
    "summary": "2-3 sentence summary"
}"""
                    }
                ]
            }]
        )
        
        response_text = ai_response.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not parse response"}
            
    except Exception as e:
        return {"error": str(e)}


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
# COMPREHENSIVE IMAGE ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/image")
async def analyze_image(request: ImageAnalysisRequest):
    """Comprehensive image analysis using multiple AI models"""
    
    try:
        # Decode base64
        image_bytes = base64.b64decode(request.image_base64)
        
        # Determine media type
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            media_type = "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"
        
        # Run both analyses in parallel
        hf_result = await detect_ai_image_huggingface(image_bytes)
        claude_result = await analyze_image_with_claude(request.image_base64, media_type)
        
        # Combine results
        ai_probability = hf_result.get("average_ai_probability", 0)
        is_likely_ai = hf_result.get("is_likely_ai", False)
        
        # Check Claude's AI indicators too
        claude_ai_indicators = claude_result.get("ai_indicators", {})
        if claude_ai_indicators.get("has_ai_artifacts", False):
            # Boost AI probability if Claude also detected artifacts
            ai_probability = min(100, ai_probability + 15)
            is_likely_ai = ai_probability > 50
        
        # Determine overall risk
        risk_factors = []
        risk_score = 0
        
        # AI Generation Risk
        if is_likely_ai:
            risk_factors.append(f"AI-generated image detected ({ai_probability}% confidence)")
            risk_score += 30
        
        # Deepfake Risk
        deepfake_analysis = claude_result.get("deepfake_analysis", {})
        if deepfake_analysis.get("is_deepfake", False):
            risk_factors.append(f"Deepfake detected ({deepfake_analysis.get('confidence', 0)}% confidence)")
            risk_score += 40
        
        # Scam Content Risk
        scam_content = claude_result.get("scam_content", {})
        if scam_content.get("is_scam", False):
            risk_factors.append(f"Scam content: {scam_content.get('scam_type', 'unknown')}")
            risk_score += 35
        
        # Manipulation Risk
        manipulation = claude_result.get("manipulation", {})
        if manipulation.get("is_manipulated", False):
            risk_factors.append(f"Image manipulated: {manipulation.get('manipulation_type', 'unknown')}")
            risk_score += 25
        
        risk_score = min(risk_score, 100)
        
        if risk_score >= 60:
            overall_risk = "critical"
        elif risk_score >= 40:
            overall_risk = "high"
        elif risk_score >= 20:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        # Build recommendation
        if is_likely_ai and ai_probability > 70:
            recommendation = "This image is very likely AI-generated. Do not trust it as authentic."
        elif is_likely_ai:
            recommendation = "This image shows signs of AI generation. Verify its authenticity before trusting."
        elif deepfake_analysis.get("is_deepfake"):
            recommendation = "This image appears to be a deepfake. Do not trust it."
        elif scam_content.get("is_scam"):
            recommendation = "This image contains scam content. Do not trust or act on it."
        elif manipulation.get("is_manipulated"):
            recommendation = "This image has been digitally manipulated. Verify with original source."
        else:
            recommendation = "No significant threats detected, but always verify important images."
        
        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            
            "ai_detection": {
                "is_ai_generated": is_likely_ai,
                "ai_probability": ai_probability,
                "models_used": hf_result.get("ai_detection_models", []),
                "verdict": "AI-GENERATED" if ai_probability > 70 else "LIKELY AI" if ai_probability > 50 else "LIKELY REAL" if ai_probability < 30 else "UNCERTAIN"
            },
            
            "deepfake_analysis": deepfake_analysis,
            "scam_content": scam_content,
            "manipulation": manipulation,
            "ai_artifacts": claude_ai_indicators,
            
            "summary": claude_result.get("summary", "Analysis complete."),
            "recommendation": recommendation,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@app.post("/analyze/image/upload")
async def analyze_image_upload(file: UploadFile = File(...)):
    """Upload and analyze an image file"""
    
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    request = ImageAnalysisRequest(
        image_base64=image_base64,
        filename=file.filename
    )
    
    return await analyze_image(request)


# ============================================
# VIDEO ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze video for deepfakes and AI generation"""
    
    # For video, we'll extract key frames and analyze them
    contents = await file.read()
    
    # Check file type
    filename = file.filename.lower() if file.filename else ""
    
    if not any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv']):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, MOV, AVI, WebM, or MKV.")
    
    # For now, return guidance on video analysis
    # Full video analysis requires ffmpeg for frame extraction
    return {
        "status": "video_analysis_limited",
        "message": "Full video deepfake detection requires frame-by-frame analysis. For production, we recommend:",
        "recommendations": [
            "Extract key frames using ffmpeg",
            "Analyze each frame for deepfake indicators",
            "Check audio-visual sync for lip-sync deepfakes",
            "Look for temporal inconsistencies across frames"
        ],
        "services_for_video": [
            {"name": "Microsoft Video Authenticator", "url": "https://www.microsoft.com/en-us/ai/responsible-ai"},
            {"name": "Sensity AI", "url": "https://sensity.ai/"},
            {"name": "Deepware Scanner", "url": "https://deepware.ai/"}
        ],
        "file_received": {
            "filename": file.filename,
            "size_mb": round(len(contents) / (1024 * 1024), 2)
        },
        "analyzed_at": datetime.now().isoformat()
    }


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
    shorteners = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly"]
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
    known_brands = ["google", "facebook", "amazon", "apple", "microsoft", "netflix", "paypal", "dbs", "ocbc", "singtel", "grab", "gojek", "lazada", "shopee"]
    for brand in known_brands:
        # Check if brand name is in URL but not the official domain
        if brand in url:
            official_domains = [f"{brand}.com", f"{brand}.sg", f"{brand}.co", f"{brand}.in", f"www.{brand}"]
            if not any(domain in url for domain in official_domains):
                red_flags.append({"flag": "Possible Typosquatting", "severity": "critical", "detail": f"May be impersonating {brand}"})
                risk_score += 30
                break
    
    # Check for suspicious keywords in URL
    suspicious_url_words = ["login", "verify", "secure", "account", "update", "confirm", "banking", "password", "signin", "authenticate"]
    for word in suspicious_url_words:
        if word in url:
            red_flags.append({"flag": "Suspicious URL Keywords", "severity": "medium", "detail": f"Contains '{word}' - common in phishing"})
            risk_score += 10
            break
    
    # Check for excessive subdomains
    try:
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        subdomain_count = parsed.netloc.count('.')
        if subdomain_count > 3:
            red_flags.append({"flag": "Excessive Subdomains", "severity": "medium", "detail": f"Has {subdomain_count} subdomains - may be hiding real domain"})
            risk_score += 15
    except:
        pass
    
    # Check for HTTPS
    if url.startswith("http://"):
        red_flags.append({"flag": "No HTTPS", "severity": "medium", "detail": "Site doesn't use secure HTTPS connection"})
        risk_score += 10
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 60:
        risk_level = "critical"
        recommendation = "DO NOT click this link. It shows strong signs of being a phishing/scam URL."
    elif risk_score >= 40:
        risk_level = "high"
        recommendation = "Be very careful with this link. Verify the source before clicking."
    elif risk_score >= 20:
        risk_level = "medium"
        recommendation = "This URL has some suspicious elements. Proceed with caution."
    else:
        risk_level = "low"
        recommendation = "This URL appears relatively safe, but always verify important links."
    
    return {
        "url": request.url,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "red_flags": red_flags,
        "recommendation": recommendation,
        "analyzed_at": datetime.now().isoformat()
    }


# ============================================
# FILE ANALYSIS ENDPOINT
# ============================================

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for malware indicators"""
    
    contents = await file.read()
    filename = file.filename.lower() if file.filename else "unknown"
    file_size = len(contents)
    
    red_flags = []
    risk_score = 0
    
    # Check file extension
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.msi', '.jar', '.vbs', '.js', '.ps1', '.apk', '.dll', '.com']
    for ext in dangerous_extensions:
        if filename.endswith(ext):
            red_flags.append({"flag": "Dangerous file type", "severity": "critical", "detail": f"File type '{ext}' can execute malicious code"})
            risk_score += 40
            break
    
    # Check for double extensions
    if filename.count('.') > 1:
        parts = filename.rsplit('.', 2)
        if len(parts) >= 2:
            red_flags.append({"flag": "Double extension", "severity": "high", "detail": f"Multiple file extensions detected - common malware trick"})
            risk_score += 25
    
    # Check for macros in Office files
    macro_extensions = ['.docm', '.xlsm', '.pptm', '.dotm', '.xltm']
    for ext in macro_extensions:
        if filename.endswith(ext):
            red_flags.append({"flag": "Macro-enabled document", "severity": "high", "detail": "This document can contain executable macros"})
            risk_score += 30
            break
    
    # Check file header magic bytes
    if len(contents) >= 4:
        if contents[:2] == b'MZ':  # Windows executable
            if not filename.endswith(('.exe', '.dll')):
                red_flags.append({"flag": "Hidden executable", "severity": "critical", "detail": "File is actually a Windows executable despite different extension"})
                risk_score += 50
            else:
                red_flags.append({"flag": "Windows executable", "severity": "high", "detail": "Executable files can run malicious code"})
                risk_score += 35
        
        elif contents[:4] == b'PK\x03\x04':  # ZIP/APK/Office
            if filename.endswith('.apk'):
                red_flags.append({"flag": "Android APK", "severity": "high", "detail": "Android app - verify source before installing"})
                risk_score += 30
            elif not filename.endswith(('.zip', '.xlsx', '.docx', '.pptx', '.apk')):
                red_flags.append({"flag": "Hidden archive", "severity": "medium", "detail": "File is actually an archive/ZIP despite extension"})
                risk_score += 20
        
        elif contents[:4] == b'%PDF':  # PDF
            # Check for JavaScript in PDF
            if b'/JavaScript' in contents or b'/JS' in contents:
                red_flags.append({"flag": "PDF with JavaScript", "severity": "high", "detail": "PDF contains JavaScript which could be malicious"})
                risk_score += 25
        
        elif contents[:5] == b'<html' or contents[:14] == b'<!DOCTYPE html':
            if not filename.endswith(('.html', '.htm')):
                red_flags.append({"flag": "Hidden HTML", "severity": "medium", "detail": "File is actually HTML despite different extension"})
                risk_score += 15
    
    # Check for suspicious strings
    suspicious_strings = [b'cmd.exe', b'powershell', b'WScript', b'eval(', b'<script', b'fromCharCode']
    for sus_string in suspicious_strings:
        if sus_string in contents[:10000]:  # Check first 10KB
            red_flags.append({"flag": "Suspicious code", "severity": "high", "detail": f"Contains potentially malicious code pattern"})
            risk_score += 20
            break
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 60:
        risk_level = "critical"
        recommendation = "DO NOT OPEN THIS FILE. It shows strong signs of being malicious. Delete it immediately."
    elif risk_score >= 40:
        risk_level = "high"
        recommendation = "Be very cautious. Only open if you completely trust the source."
    elif risk_score >= 20:
        risk_level = "medium"
        recommendation = "Exercise caution. Verify the sender before opening."
    else:
        risk_level = "low"
        recommendation = "File appears safe, but always verify the source."
    
    return {
        "filename": file.filename,
        "file_size_bytes": file_size,
        "file_size_readable": f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB",
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
        "version": "2.1.0",
        "status": "operational",
        "features": [
            "Text/Message Scam Detection",
            "AI Image Detection (Hugging Face + Claude)",
            "Deepfake Detection",
            "Scam Content Detection",
            "URL Phishing Analysis",
            "File Malware Scanning",
            "Video Analysis (Limited)"
        ],
        "endpoints": {
            "text_analysis": "POST /analyze/text",
            "image_analysis": "POST /analyze/image",
            "image_upload": "POST /analyze/image/upload",
            "video_analysis": "POST /analyze/video",
            "url_analysis": "POST /analyze/url",
            "file_analysis": "POST /analyze/file",
            "documentation": "GET /docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.1.0", "timestamp": datetime.now().isoformat()}