from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from supabase import create_client, Client
from matching import MentorMenteeMatcher
import json

app = FastAPI(title="REM Matching System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# Pydantic models for request/response
class User(BaseModel):
    email: str
    first_name: str
    last_name: str
    year: str
    research_interests: str
    meeting_frequency: str
    major: str
    hobbies: str
    mentoring_needs: str
    is_mentor: bool

class MatchRequest(BaseModel):
    mentor_id: str
    mentee_id: str
    compatibility_score: float

# Initialize matcher
matcher = MentorMenteeMatcher()

@app.get("/")
async def root():
    return {"message": "REM Matching System API"}

@app.post("/users")
async def create_user(user: User):
    try:
        data = user.dict()
        result = supabase.table("users").insert(data).execute()
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users")
async def get_users(is_mentor: Optional[bool] = None):
    try:
        query = supabase.table("users").select("*")
        if is_mentor is not None:
            query = query.eq("is_mentor", is_mentor)
        result = query.execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/matches")
async def create_matches():
    try:
        # Get all mentors and mentees
        mentors = supabase.table("users").select("*").eq("is_mentor", True).execute()
        mentees = supabase.table("users").select("*").eq("is_mentor", False).execute()
        
        # Find optimal matches
        matches = matcher.find_optimal_matches(mentors.data, mentees.data)
        
        # Save matches to database
        for mentor, mentee, score in matches:
            match_data = {
                "mentor_id": mentor["id"],
                "mentee_id": mentee["id"],
                "compatibility_score": score
            }
            supabase.table("matches").insert(match_data).execute()
        
        return {"message": "Matches created successfully", "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/matches")
async def get_matches():
    try:
        result = supabase.table("matches").select("*").execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/matches/{match_id}")
async def delete_match(match_id: str):
    try:
        supabase.table("matches").delete().eq("id", match_id).execute()
        return {"message": "Match deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 