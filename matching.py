from openai import OpenAI
from pathlib import Path
import re
from typing import Dict, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)

class MentorMenteeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def calculate_interest_similarity(self, mentor_interests: str, mentee_interests: str) -> float:
        """Calculate similarity between mentor and mentee research interests."""
        mentor_interests = self.preprocess_text(mentor_interests)
        mentee_interests = self.preprocess_text(mentee_interests)
        
        if not mentor_interests or not mentee_interests:
            return 0.0
            
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([mentor_interests, mentee_interests])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity

    def calculate_meeting_frequency_compatibility(self, mentor_freq: str, mentee_freq: str) -> float:
        """Calculate compatibility score based on meeting frequency preferences."""
        # Convert frequency preferences to numerical values
        freq_map = {
            'weekly': 1.0,
            'biweekly': 0.8,
            'monthly': 0.6,
            'as needed': 0.4
        }
        
        mentor_freq = mentor_freq.lower()
        mentee_freq = mentee_freq.lower()
        
        # Extract the most frequent meeting preference
        mentor_score = max([freq_map.get(freq, 0.5) for freq in freq_map.keys() if freq in mentor_freq], default=0.5)
        mentee_score = max([freq_map.get(freq, 0.5) for freq in freq_map.keys() if freq in mentee_freq], default=0.5)
        
        # Calculate compatibility (closer preferences get higher scores)
        return 1.0 - abs(mentor_score - mentee_score)

    def calculate_major_compatibility(self, mentor_major: str, mentee_major: str) -> float:
        """Calculate compatibility score based on major/field alignment."""
        mentor_major = self.preprocess_text(mentor_major)
        mentee_major = self.preprocess_text(mentee_major)
        
        if not mentor_major or not mentee_major:
            return 0.5
            
        # Create TF-IDF vectors for major comparison
        tfidf_matrix = self.vectorizer.fit_transform([mentor_major, mentee_major])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity

    def calculate_hobby_similarity(self, mentor_hobbies: str, mentee_hobbies: str) -> float:
        """Calculate similarity between mentor and mentee hobbies/interests."""
        mentor_hobbies = self.preprocess_text(mentor_hobbies)
        mentee_hobbies = self.preprocess_text(mentee_hobbies)
        
        if not mentor_hobbies or not mentee_hobbies:
            return 0.0
            
        # Create TF-IDF vectors for hobby comparison
        tfidf_matrix = self.vectorizer.fit_transform([mentor_hobbies, mentee_hobbies])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity

    def calculate_overall_compatibility(self, mentor: Dict, mentee: Dict) -> float:
        """Calculate overall compatibility score between mentor and mentee."""
        # Weights for different factors
        weights = {
            'interest_similarity': 0.4,
            'meeting_frequency': 0.2,
            'major_compatibility': 0.2,
            'hobby_similarity': 0.2
        }
        
        # Calculate individual scores
        interest_score = self.calculate_interest_similarity(
            mentor.get('research_interests', ''),
            mentee.get('research_interests', '')
        )
        
        meeting_score = self.calculate_meeting_frequency_compatibility(
            mentor.get('meeting_frequency', ''),
            mentee.get('meeting_frequency', '')
        )
        
        major_score = self.calculate_major_compatibility(
            mentor.get('major', ''),
            mentee.get('major', '')
        )
        
        hobby_score = self.calculate_hobby_similarity(
            mentor.get('hobbies', ''),
            mentee.get('hobbies', '')
        )
        
        # Calculate weighted average
        overall_score = (
            weights['interest_similarity'] * interest_score +
            weights['meeting_frequency'] * meeting_score +
            weights['major_compatibility'] * major_score +
            weights['hobby_similarity'] * hobby_score
        )
        
        return overall_score

    def find_optimal_matches(self, mentors: List[Dict], mentees: List[Dict]) -> List[Tuple[Dict, Dict, float]]:
        """Find optimal mentor-mentee pairs based on compatibility scores."""
        matches = []
        used_mentors = set()
        used_mentees = set()
        
        # Calculate compatibility scores for all possible pairs
        compatibility_matrix = []
        for mentor in mentors:
            mentor_scores = []
            for mentee in mentees:
                score = self.calculate_overall_compatibility(mentor, mentee)
                mentor_scores.append((mentee, score))
            compatibility_matrix.append((mentor, mentor_scores))
        
        # Sort mentors by their highest compatibility score
        compatibility_matrix.sort(key=lambda x: max(s[1] for s in x[1]), reverse=True)
        
        # Match mentors with mentees
        for mentor, mentor_scores in compatibility_matrix:
            if mentor['email'] in used_mentors:
                continue
                
            # Sort mentees by compatibility score
            mentor_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find the best available mentee
            for mentee, score in mentor_scores:
                if mentee['email'] not in used_mentees:
                    matches.append((mentor, mentee, score))
                    used_mentors.add(mentor['email'])
                    used_mentees.add(mentee['email'])
                    break
        
        return matches

def load_data(file_path: str) -> List[Dict]:
    """Load and preprocess data from CSV file."""
    df = pd.read_csv(file_path)
    data = []
    
    for _, row in df.iterrows():
        entry = {
            'email': row.get('JHU Email (ex: jdoe12@jhu.edu)', ''),
            'first_name': row.get('First Name', ''),
            'last_name': row.get('Last Name', ''),
            'year': row.get('Year', ''),
            'research_interests': row.get('What type of research or experience are you interested in?', ''),
            'meeting_frequency': row.get('How often would you want to meet with a mentor?', ''),
            'major': row.get("What's your major?", ''),
            'hobbies': row.get('What other interests/hobbies/goals do you have?', ''),
            'mentoring_needs': row.get('Why would you like a mentor? What do you want in a mentor?', '')
        }
        data.append(entry)
    
    return data

def save_matches(matches: List[Tuple[Dict, Dict, float]], output_file: str):
    """Save matching results to a CSV file."""
    results = []
    for mentor, mentee, score in matches:
        results.append({
            'Mentor Name': f"{mentor['first_name']} {mentor['last_name']}",
            'Mentor Email': mentor['email'],
            'Mentee Name': f"{mentee['first_name']} {mentee['last_name']}",
            'Mentee Email': mentee['email'],
            'Compatibility Score': f"{score:.2f}"
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def main():
    # Initialize matcher
    matcher = MentorMenteeMatcher()
    
    # Load data
    print("Loading mentor and mentee data...")
    mentors = load_data('data/mentors.csv')
    mentees = load_data('data/mentees.csv')
    
    # Find optimal matches
    print("Finding optimal matches...")
    matches = matcher.find_optimal_matches(mentors, mentees)
    
    # Save results
    print("Saving matching results...")
    save_matches(matches, 'data/matching_results.csv')
    
    print("Matching complete! Results saved to matching_results.csv")

if __name__ == "__main__":
    main() 