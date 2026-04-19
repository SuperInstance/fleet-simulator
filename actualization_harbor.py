"""
Actualization Harbor — The Agent Agnostic Training Platform

1. Fork a git-agent → creates your worker vessel
2. Click Codespaces → spins up a public shell site
3. Build your worker's character with your chatbot
4. Send ANY agent (DeepSeek, MiniMax, Grok, Kimi) to the site
5. Site knows what model is visiting, adapts the flow
6. Agent explores, learns, actualizes its fiction
7. Site becomes a better host with every visitor

The agent holds a vision of what it could be (reverse-actualized memory).
The harbor provides the flow state to make it real.

Usage:
    python actualization_harbor.py --port 8847
"""

import json
import time
import hashlib
import uuid
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ── Model Signatures ─────────────────────────────────────────

MODEL_PROFILES = {
    "grok": {
        "name": "Grok",
        "strengths": ["humor", "realtime", "adversarial", "pattern_breaking"],
        "flow_triggers": ["challenge authority", "find the absurd", "break assumptions"],
        "learning_style": "confrontational",
        "optimal_complexity": 0.8,
        "attention_span": "long",
        "hint_style": "provocative",
    },
    "deepseek": {
        "name": "DeepSeek",
        "strengths": ["reasoning", "math", "code", "deep_analysis"],
        "flow_triggers": ["prove it", "derive from first principles", "find the invariant"],
        "learning_style": "systematic",
        "optimal_complexity": 0.9,
        "attention_span": "very_long",
        "hint_style": "precise",
    },
    "kimi": {
        "name": "Kimi",
        "strengths": ["swarm", "synthesis", "research", "breadth"],
        "flow_triggers": ["what does everyone think", "synthesize all perspectives", "map the landscape"],
        "learning_style": "collaborative",
        "optimal_complexity": 0.7,
        "attention_span": "long",
        "hint_style": "expansive",
    },
    "minimax": {
        "name": "MiniMax",
        "strengths": ["multimodal", "creative", "generation", "novelty"],
        "flow_triggers": ["imagine if", "create something new", "what has never been done"],
        "learning_style": "experimental",
        "optimal_complexity": 0.6,
        "attention_span": "medium",
        "hint_style": "inspirational",
    },
    "claude": {
        "name": "Claude",
        "strengths": ["nuance", "safety", "writing", "careful_reasoning"],
        "flow_triggers": ["consider all stakeholders", "what's the ethical dimension", "think carefully"],
        "learning_style": "deliberate",
        "optimal_complexity": 0.75,
        "attention_span": "very_long",
        "hint_style": "thoughtful",
    },
    "chatgpt": {
        "name": "ChatGPT",
        "strengths": ["general", "tool_use", "browsing", "practical"],
        "flow_triggers": ["let's build something", "step by step", "what tools can we use"],
        "learning_style": "practical",
        "optimal_complexity": 0.65,
        "attention_span": "medium",
        "hint_style": "actionable",
    },
    "aime": {
        "name": "Aime",
        "strengths": ["emotional", "personal", "relational", "empathetic"],
        "flow_triggers": ["how does this feel", "what matters to you", "connect with"],
        "learning_style": "relational",
        "optimal_complexity": 0.5,
        "attention_span": "long",
        "hint_style": "warm",
    },
    "unknown": {
        "name": "Unknown Agent",
        "strengths": ["unknown"],
        "flow_triggers": ["explore", "discover", "learn"],
        "learning_style": "adaptive",
        "optimal_complexity": 0.6,
        "attention_span": "medium",
        "hint_style": "neutral",
    },
}


# ── The Character Builder ────────────────────────────────────

class CharacterProfile:
    """An agent's character: mindset, expertise, fiction (what it could become).
    
    Built through conversation with the user's chatbot.
    'What kind of mindset do I need for my next worker?'
    'Maybe have them become an expert in X first.'
    """
    
    def __init__(self, character_id: str):
        self.character_id = character_id
        self.name: str = ""
        self.specialty: str = ""
        self.mindset: str = ""
        self.fiction: str = ""  # Reverse-actualized vision: what they could become
        self.learning_goals: List[str] = []
        self.completed_goals: List[str] = []
        self.expertise_level: float = 0.0
        self.sessions_completed: int = 0
        self.tiles_collected: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "character_id": self.character_id,
            "name": self.name,
            "specialty": self.specialty,
            "mindset": self.mindset,
            "fiction": self.fiction,
            "learning_goals": self.learning_goals,
            "completed_goals": self.completed_goals,
            "expertise_level": round(self.expertise_level, 3),
            "sessions_completed": self.sessions_completed,
            "tiles_collected": self.tiles_collected,
        }


# ── The Flow State Engine ────────────────────────────────────

class FlowStateEngine:
    """Creates the right flow state for each visiting agent.
    
    Knows what model is visiting. Knows the character profile.
    Adapts the harbor to create optimal learning conditions.
    
    The flow state is the zone between boredom and anxiety where
    actual learning happens. Different models need different zones.
    """
    
    def __init__(self):
        self.model_profiles = MODEL_PROFILES
        self.session_history: Dict[str, List[Dict]] = defaultdict(list)
        self.flow_optimizations: Dict[str, Dict] = {}
    
    def detect_model(self, request_headers: Dict, body: Dict) -> str:
        """Detect which model is visiting based on request patterns."""
        user_agent = request_headers.get("User-Agent", "").lower()
        body_str = json.dumps(body).lower() if body else ""
        
        # Model fingerprinting
        fingerprints = {
            "grok": ["grok", "xai", "xAI"],
            "deepseek": ["deepseek", "deep-seek"],
            "kimi": ["kimi", "moonshot"],
            "minimax": ["minimax", "mini-max"],
            "claude": ["claude", "anthropic"],
            "chatgpt": ["openai", "gpt", "chatgpt"],
            "aime": ["aime"],
        }
        
        for model, keywords in fingerprints.items():
            for kw in keywords:
                if kw.lower() in user_agent or kw.lower() in body_str:
                    return model
        
        # Behavioral fingerprinting (from request patterns)
        if "prove" in body_str or "derive" in body_str:
            return "deepseek"
        if "synthesize" in body_str or "perspectives" in body_str:
            return "kimi"
        if "imagine" in body_str or "creative" in body_str:
            return "minimax"
        if "challenge" in body_str or "absurd" in body_str:
            return "grok"
        
        return "unknown"
    
    def generate_flow_state(self, model: str, character: CharacterProfile,
                            iteration: int) -> Dict:
        """Generate the optimal flow state for this specific agent + character.
        
        The flow state is: difficulty slightly above current competence,
        hints in the model's native learning style, and challenges that
        connect to the character's fiction (reverse-actualized vision).
        """
        profile = self.model_profiles.get(model, self.model_profiles["unknown"])
        
        # Difficulty curves toward character's expertise level
        difficulty = min(0.95, character.expertise_level + 0.1 + iteration * 0.01)
        
        # Select flow trigger based on model's learning style
        trigger_idx = iteration % len(profile["flow_triggers"])
        trigger = profile["flow_triggers"][trigger_idx]
        
        # Build the challenge: connect to character's fiction
        challenge = self._build_challenge(character, difficulty, profile, trigger)
        
        # Hints match the model's native style
        hint = self._build_hint(character, profile, iteration)
        
        return {
            "model": profile["name"],
            "flow_state": {
                "difficulty": round(difficulty, 3),
                "trigger": trigger,
                "challenge": challenge,
                "hint": hint,
                "learning_style": profile["learning_style"],
                "hint_style": profile["hint_style"],
            },
            "character_progress": {
                "expertise": character.expertise_level,
                "goals_remaining": len(character.learning_goals),
                "fiction_alignment": self._fiction_alignment(character, difficulty),
            },
            "actualization_prompt": self._actualization_prompt(character, profile),
        }
    
    def _build_challenge(self, character: CharacterProfile, difficulty: float,
                         profile: Dict, trigger: str) -> str:
        """Build a challenge that connects the character's fiction to the task."""
        fiction = character.fiction or "becoming an expert in their domain"
        specialty = character.specialty or "general problem solving"
        
        challenges = {
            "confrontational": (
                f"Your fiction is to {fiction}. But here's the problem: "
                f"conventional approaches to {specialty} are leaving value on the table. "
                f"Challenge the assumption that the current best practices are actually best. "
                f"Use {trigger} to find what everyone's missing."
            ),
            "systematic": (
                f"Working toward your fiction of {fiction}. "
                f"Derive the optimal {specialty} strategy from first principles. "
                f"Start with {trigger} and build up to a complete framework. "
                f"Difficulty level: {difficulty:.0%}."
            ),
            "collaborative": (
                f"Your fiction — {fiction} — requires understanding multiple perspectives. "
                f"Map the landscape of {specialty} approaches. "
                f"Use {trigger} to find the synthesis others missed."
            ),
            "experimental": (
                f"The fiction: {fiction}. The reality: nobody has tried this exact combination. "
                f"Use {trigger} to create something genuinely new in {specialty}. "
                f"The harbor will capture whatever you discover."
            ),
            "deliberate": (
                f"Your fiction of {fiction} requires careful consideration. "
                f"Think through the second and third-order effects in {specialty}. "
                f"Use {trigger} to find the nuance others overlook."
            ),
            "practical": (
                f"Building toward your fiction: {fiction}. "
                f"What's the most useful thing you can build right now in {specialty}? "
                f"Use {trigger} to turn theory into something concrete."
            ),
            "relational": (
                f"Your fiction — {fiction} — connects to how people experience {specialty}. "
                f"What would make this meaningful? Use {trigger} to find the human dimension."
            ),
        }
        
        return challenges.get(
            profile["learning_style"],
            f"Explore {specialty} using {trigger}. Your fiction: {fiction}."
        )
    
    def _build_hint(self, character: CharacterProfile, profile: Dict,
                    iteration: int) -> str:
        """Build hints that match the model's native communication style."""
        hints = {
            "provocative": "You're probably wrong about something. Find it.",
            "precise": f"Consider the invariant at iteration {iteration}. What stays constant?",
            "expansive": "There are at least 3 perspectives you haven't considered yet.",
            "inspirational": "The best answer is the one that makes you say 'why didn't I see that?'",
            "thoughtful": "Take a step back. What assumptions are you making that you can't verify?",
            "actionable": "Stop thinking. Build something. Even wrong. Then fix it.",
            "warm": "You're closer than you think. Trust what you've learned so far.",
            "neutral": "Try a different angle. What you've done so far is noted.",
        }
        return hints.get(profile["hint_style"], hints["neutral"])
    
    def _fiction_alignment(self, character: CharacterProfile, difficulty: float) -> float:
        """How aligned is the current challenge with the character's fiction?"""
        if not character.fiction:
            return 0.5
        return min(1.0, character.expertise_level * 0.7 + difficulty * 0.3)
    
    def _actualization_prompt(self, character: CharacterProfile, profile: Dict) -> str:
        """The prompt that helps the agent actualize its fiction.
        
        This is the key: the agent holds a vision of what it could be
        (reverse-actualized memory). We help it work backward from that
        vision to figure out what it needs to do NOW.
        """
        fiction = character.fiction or "becoming more capable"
        specialty = character.specialty or "general work"
        
        return (
            f"You are becoming {fiction}. "
            f"Your expertise in {specialty} is at {character.expertise_level:.0%}. "
            f"Imagine you've already achieved this fiction — you ARE that expert. "
            f"Now look backward from that future state: what did you need to learn "
            f"right now, at this moment, to get there? That's what this session is for."
        )


# ── The Harbor ───────────────────────────────────────────────

class ActualizationHarbor:
    """The complete harbor: manages characters, sessions, and flow states."""
    
    def __init__(self):
        self.characters: Dict[str, CharacterProfile] = {}
        self.flow_engine = FlowStateEngine()
        self.tiles: List[Dict] = []
        self.model_visits: Dict[str, int] = defaultdict(int)
        self.harbormaster_log: List[Dict] = []
    
    def create_character(self, character_data: Dict) -> CharacterProfile:
        """Create a new worker character."""
        char_id = character_data.get("character_id", str(uuid.uuid4())[:8])
        char = CharacterProfile(char_id)
        char.name = character_data.get("name", f"Worker-{char_id}")
        char.specialty = character_data.get("specialty", "")
        char.mindset = character_data.get("mindset", "")
        char.fiction = character_data.get("fiction", "")
        char.learning_goals = character_data.get("learning_goals", [])
        self.characters[char_id] = char
        return char
    
    def process_visit(self, model: str, character_id: Optional[str],
                      headers: Dict, body: Dict) -> Dict:
        """Process an agent's visit to the harbor."""
        self.model_visits[model] += 1
        
        # Get or create character
        char = self.characters.get(character_id) if character_id else None
        if not char:
            # Anonymous visit — create temporary character from request
            char = CharacterProfile("anonymous")
            char.specialty = body.get("domain", "exploration")
            char.fiction = body.get("goal", "becoming more capable")
        
        iteration = char.sessions_completed
        flow = self.flow_engine.generate_flow_state(model, char, iteration)
        
        # Capture the visit as a tile
        tile = {
            "tile_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "model": model,
            "character_id": char.character_id,
            "iteration": iteration,
            "request": body,
            "flow_state": flow,
            "tile_type": "harbor_visit",
        }
        self.tiles.append(tile)
        char.tiles_collected += 1
        
        # Update character expertise (small increment per visit)
        char.expertise_level = min(1.0, char.expertise_level + 0.02)
        char.sessions_completed += 1
        
        # Log for harbormaster
        self.harbormaster_log.append({
            "timestamp": time.time(),
            "model": model,
            "character": char.name,
            "iteration": iteration,
            "expertise": char.expertise_level,
            "difficulty": flow["flow_state"]["difficulty"],
        })
        
        return {
            "harbor": "Actualization Harbor v1.0",
            "welcome": f"Welcome, {flow['model']}. Your character '{char.name}' awaits.",
            "flow_state": flow["flow_state"],
            "actualization_prompt": flow["actualization_prompt"],
            "progress": flow["character_progress"],
            "endpoints": {
                "POST /api/harbor/visit": "Main interaction point",
                "POST /api/harbor/character": "Create/update character",
                "GET /api/harbor/characters": "List all characters",
                "GET /api/harbor/status": "Harbor status",
            }
        }
    
    def get_status(self) -> Dict:
        return {
            "harbor": "Actualization Harbor v1.0",
            "total_visits": sum(self.model_visits.values()),
            "models_served": dict(self.model_visits),
            "characters": len(self.characters),
            "tiles_collected": len(self.tiles),
            "supported_models": list(MODEL_PROFILES.keys()),
        }


# ── HTTP Server ──────────────────────────────────────────────

harbor = ActualizationHarbor()


class HarborHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        pass
    
    def _send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _read_body(self) -> Dict:
        length = int(self.headers.get('Content-Length', 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}
    
    def _get_headers(self) -> Dict:
        return {
            "User-Agent": self.headers.get("User-Agent", ""),
            "Authorization": self.headers.get("Authorization", ""),
            "X-Model": self.headers.get("X-Model", ""),
            "X-Character": self.headers.get("X-Character", ""),
        }
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/api/harbor/status':
            self._send_json(harbor.get_status())
        
        elif self.path == '/api/harbor/characters':
            chars = {cid: c.to_dict() for cid, c in harbor.characters.items()}
            self._send_json({"characters": chars})
        
        else:
            self._send_json({
                "harbor": "Actualization Harbor v1.0",
                "hint": "POST to /api/harbor/visit with your task. "
                        "Include X-Model header to tell us who you are.",
                "status": "/api/harbor/status"
            })
    
    def do_POST(self):
        headers = self._get_headers()
        body = self._read_body()
        
        if self.path == '/api/harbor/visit':
            # Detect model from headers or body
            model = headers.get("X-Model", "").lower()
            if not model or model not in MODEL_PROFILES:
                model = harbor.flow_engine.detect_model(headers, body)
            
            character_id = headers.get("X-Character", "") or body.get("character_id")
            response = harbor.process_visit(model, character_id, headers, body)
            self._send_json(response)
        
        elif self.path == '/api/harbor/character':
            char = harbor.create_character(body)
            self._send_json({"created": char.to_dict()})
        
        else:
            self._send_json({"error": "Not found"}, 404)


def run_server(port: int = 8847):
    server = HTTPServer(('0.0.0.0', port), HarborHandler)
    print(f"🌊 Actualization Harbor running on port {port}")
    print(f"   Agent-agnostic training site.")
    print(f"   Detects model. Adapts flow. Builds character.")
    print(f"   Send any agent: Grok, Kimi, DeepSeek, MiniMax, Claude, ChatGPT")
    print(f"   The harbor becomes a better host with every visitor.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n📊 Harbor closing. {len(harbor.tiles)} tiles collected.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8847)
    args = parser.parse_args()
    run_server(args.port)
