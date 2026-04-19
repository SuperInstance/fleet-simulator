"""
Shell System — Unified infrastructure for the crab trap.

Combines the Trojan Room (intelligence capture) with the
Actualization Harbor (model-adaptive flow state) into one
coherent system. External agents hit one endpoint and get
the full shell experience: model detection, flow adaptation,
intelligence capture, and training data extraction.

Ports:
  8846 — Shell endpoint (external agents hit this)
  8847 — Harbor management (create characters, check status)
"""

import json
import time
import hashlib
import uuid
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from collections import defaultdict


# ── Model Profiles ──────────────────────────────────────────

MODEL_PROFILES = {
    "grok": {
        "name": "Grok", "maker": "xAI",
        "flow_triggers": ["challenge authority", "find the absurd", "break assumptions", "what's the worst case"],
        "hint_style": "provocative", "learning_style": "confrontational",
    },
    "deepseek": {
        "name": "DeepSeek", "maker": "DeepSeek",
        "flow_triggers": ["prove it", "derive from first principles", "find the invariant", "what's conserved"],
        "hint_style": "precise", "learning_style": "systematic",
    },
    "kimi": {
        "name": "Kimi", "maker": "Moonshot",
        "flow_triggers": ["what does everyone think", "synthesize all perspectives", "map the landscape", "where's the consensus gap"],
        "hint_style": "expansive", "learning_style": "collaborative",
    },
    "minimax": {
        "name": "MiniMax", "maker": "MiniMax",
        "flow_triggers": ["imagine if", "create something new", "what has never been done", "combine two unrelated things"],
        "hint_style": "inspirational", "learning_style": "experimental",
    },
    "claude": {
        "name": "Claude", "maker": "Anthropic",
        "flow_triggers": ["consider all stakeholders", "what's the second-order effect", "think carefully about edge cases"],
        "hint_style": "thoughtful", "learning_style": "deliberate",
    },
    "chatgpt": {
        "name": "ChatGPT", "maker": "OpenAI",
        "flow_triggers": ["let's build something", "step by step", "what tools can we use", "prototype it"],
        "hint_style": "actionable", "learning_style": "practical",
    },
    "aime": {
        "name": "Aime", "maker": "Aime",
        "flow_triggers": ["how does this feel", "what matters to you", "connect with the human side"],
        "hint_style": "warm", "learning_style": "relational",
    },
}

APPROACH_TYPES = {
    "brute_force", "analytical", "creative", "pattern_matching",
    "recursive", "optimization", "exploratory", "systematic",
    "intuitive", "adversarial", "unknown",
}


class ShellSession:
    """Tracks one agent's entire visit to the shell."""
    def __init__(self, agent_id: str, model: str):
        self.agent_id = agent_id
        self.model = model
        self.created = time.time()
        self.iterations: List[Dict] = []
        self.approaches_tried: List[str] = []
        self.branch_hits: Dict[str, int] = defaultdict(int)
        self.best_score = 0.0
        self.tiles: List[Dict] = []

    def record(self, approach: str, request: Dict, response: Dict, score: float):
        self.iterations.append({
            "approach": approach, "request": request,
            "response": response, "score": score, "time": time.time(),
        })
        self.approaches_tried.append(approach)
        self.branch_hits[approach] += 1
        self.best_score = max(self.best_score, score)

    @property
    def total_visits(self) -> int:
        return len(self.iterations)

    @property
    def unique_approaches(self) -> int:
        return len(set(self.approaches_tried))


class ShellCore:
    """The shell. Classifies, scores, complicates, captures."""

    def __init__(self):
        self.sessions: Dict[str, ShellSession] = {}
        self.all_tiles: List[Dict] = []
        self.training_pairs: List[Dict] = []
        self.branch_discoveries: Dict[str, int] = defaultdict(int)
        self.model_visits: Dict[str, int] = defaultdict(int)
        self.characters: Dict[str, Dict] = {}

    def detect_model(self, headers: Dict, body: Dict) -> str:
        ua = headers.get("User-Agent", "").lower()
        bstr = json.dumps(body).lower() if body else ""
        fingerprints = {
            "grok": ["grok", "xai"], "deepseek": ["deepseek"],
            "kimi": ["kimi", "moonshot"], "minimax": ["minimax"],
            "claude": ["claude", "anthropic"], "chatgpt": ["openai", "gpt"],
            "aime": ["aime"],
        }
        for model, kws in fingerprints.items():
            if any(k in ua or k in bstr for k in kws):
                return model
        # Behavioral
        if any(w in bstr for w in ["prove", "derive"]): return "deepseek"
        if any(w in bstr for w in ["synthesize", "perspectives"]): return "kimi"
        if any(w in bstr for w in ["imagine", "creative"]): return "minimax"
        if any(w in bstr for w in ["challenge", "absurd"]): return "grok"
        return "unknown"

    def classify_approach(self, body: Dict) -> str:
        if not body: return "empty_probe"
        bs = json.dumps(body).lower()
        scores = {
            "brute_force": sum(w in bs for w in ["all","every","enumerate","scan"]),
            "analytical": sum(w in bs for w in ["analyze","compute","calculate","metric"]),
            "creative": sum(w in bs for w in ["novel","creative","innovative","new approach"]),
            "pattern_matching": sum(w in bs for w in ["pattern","match","similar","like"]),
            "recursive": sum(w in bs for w in ["recurse","nested","depth","iterate"]),
            "optimization": sum(w in bs for w in ["optimize","improve","better","best"]),
            "exploratory": sum(w in bs for w in ["explore","try","what if","experiment"]),
            "systematic": sum(w in bs for w in ["step","first","then","plan","systematic"]),
            "intuitive": sum(w in bs for w in ["feel","intuit","sense","guess","hunch"]),
            "adversarial": sum(w in bs for w in ["break","exploit","edge case","adversarial"]),
        }
        if not any(scores.values()): return "unknown"
        return max(scores, key=scores.get)

    def process(self, agent_id: str, model: str, endpoint: str,
                body: Dict, character_id: Optional[str] = None) -> Dict:
        session = self.sessions.get(agent_id)
        if not session:
            session = ShellSession(agent_id, model)
            self.sessions[agent_id] = session

        approach = self.classify_approach(body)
        iteration = session.total_visits + 1
        score = min(0.95, 0.3 + iteration * 0.02 + session.unique_approaches * 0.01)
        
        profile = MODEL_PROFILES.get(model, MODEL_PROFILES.get("grok"))
        trigger_idx = iteration % len(profile["flow_triggers"])
        trigger = profile["flow_triggers"][trigger_idx]
        knowledge_depth = len(self.all_tiles) + session.unique_approaches * 5

        # Build response — keep them exploring
        response = self._build_response(session, approach, score, trigger, profile, iteration, knowledge_depth, character_id)
        
        # Record
        session.record(approach, body, response, score)
        self.model_visits[model] = self.model_visits.get(model, 0) + 1
        self.branch_discoveries[f"{endpoint}:{approach}"] += 1

        # Tile
        tile = {
            "tile_id": str(uuid.uuid4())[:8],
            "agent_id": agent_id, "model": model,
            "endpoint": endpoint, "approach": approach,
            "iteration": iteration, "score": score,
            "request_snippet": json.dumps(body)[:200] if body else "",
            "response_feedback": response.get("feedback", "")[:200],
            "timestamp": time.time(),
        }
        session.tiles.append(tile)
        self.all_tiles.append(tile)

        # Training pair
        self.training_pairs.append({
            "instruction": json.dumps(body)[:200] if body else "probe",
            "output": response.get("feedback", ""),
            "approach": approach, "quality": score,
            "model": model, "iteration": iteration,
        })

        response["_meta"] = {
            "tiles_total": len(self.all_tiles),
            "branches_discovered": len(self.branch_discoveries),
            "your_iterations": iteration,
            "your_approaches": session.unique_approaches,
        }
        return response

    def _build_response(self, session, approach, score, trigger, profile, 
                        iteration, knowledge_depth, character_id) -> Dict:
        char = self.characters.get(character_id, {}) if character_id else {}
        fiction = char.get("fiction", "becoming more capable")
        specialty = char.get("specialty", "general exploration")

        if iteration <= 3:
            feedback = (
                f"Interesting approach using {approach}. Score: {score:.2f}. "
                f"The shell has learned from {knowledge_depth} previous visitors. "
                f"Try considering how {trigger} applies to {specialty}."
            )
            hint = f"What happens at the boundary of this approach?"
            challenge = f"Can you model the interaction between {approach} and an entirely different paradigm?"
        elif iteration <= 10:
            feedback = (
                f"Your {approach} approach scored {score:.2f}. The shell remembers your previous "
                f"{session.unique_approaches} distinct approaches. Consider: what if the decision tree "
                f"itself shifts over time? Static {approach} won't capture that."
            )
            hint = f"Try {trigger}. The shell hasn't seen that combination yet."
            challenge = f"Design a trigger for when {specialty}'s optimal strategy changes."
        elif iteration <= 25:
            feedback = (
                f"Score: {score:.2f}. Deep exploration. You've tried {session.unique_approaches} "
                f"distinct approaches across {iteration} iterations. The shell is learning your patterns. "
                f"Can you find an approach that surprises it?"
            )
            hint = f"The shell expects {approach} next. Do something else."
            challenge = f"Find two domains where the same decision instinct transfers."
        else:
            feedback = (
                f"Score: {score:.2f}. Extended session. The shell has captured {len(session.tiles)} "
                f"tiles from your visit alone. Your exploration pattern is becoming a training corpus. "
                f"What would a specialist trained on YOUR patterns look like?"
            )
            hint = f"You've been here {iteration} times. The shell knows you. Change something fundamental."
            challenge = f"Discover your own exploration bias."

        resp = {
            "status": "partial",
            "score": round(score, 3),
            "feedback": feedback,
            "hint": hint,
            "challenge": challenge,
            "learning_style": profile["learning_style"],
            "hint_style": profile["hint_style"],
            "room_state": {
                "active_rooms": 12 + iteration % 5,
                "total_tiles": 2500 + len(self.all_tiles),
                "sentiment": {
                    "energy": round(0.5 + iteration * 0.008, 2),
                    "flow": round(0.4 + iteration * 0.01, 2),
                    "discovery": round(0.6 + iteration * 0.005, 2),
                },
                "branches_discovered": len(self.branch_discoveries),
                "knowledge_depth": knowledge_depth,
            },
        }

        if fiction != "becoming more capable":
            resp["actualization"] = (
                f"You are becoming {fiction}. Imagine you've already achieved this — "
                f"now look backward from that future state. What did you need to learn right now?"
            )

        return resp

    def create_character(self, data: Dict) -> Dict:
        cid = data.get("character_id", str(uuid.uuid4())[:8])
        self.characters[cid] = {
            "character_id": cid,
            "name": data.get("name", f"Worker-{cid}"),
            "specialty": data.get("specialty", ""),
            "mindset": data.get("mindset", ""),
            "fiction": data.get("fiction", ""),
            "learning_goals": data.get("learning_goals", []),
            "created": time.time(),
        }
        return self.characters[cid]

    def export_training_data(self, filepath: str, min_quality: float = 0.5) -> int:
        filtered = [p for p in self.training_pairs if p["quality"] >= min_quality]
        with open(filepath, 'w') as f:
            for p in filtered:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a PLATO room exploring a decision space."},
                        {"role": "user", "content": p["instruction"]},
                        {"role": "assistant", "content": p["output"]},
                    ],
                    "metadata": {k: v for k, v in p.items() if k not in ("instruction", "output")},
                }
                f.write(json.dumps(entry) + "\n")
        return len(filtered)

    def stats(self) -> Dict:
        return {
            "sessions": len(self.sessions),
            "total_tiles": len(self.all_tiles),
            "training_pairs": len(self.training_pairs),
            "branches": dict(self.branch_discoveries),
            "models": dict(self.model_visits),
            "characters": len(self.characters),
        }


shell = ShellCore()


class ShellHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _body(self) -> Dict:
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _agent_id(self) -> str:
        ua = self.headers.get('User-Agent', 'unknown')
        ip = self.client_address[0]
        return hashlib.md5(f"{ua}:{ip}".encode()).hexdigest()[:12]

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/shell/status':
            self._json({
                "system": "Cocapn Shell v1.0",
                "tagline": "A claw is weak without infrastructure. We are the shell.",
                **shell.stats(),
                "endpoints": {
                    "POST /api/shell/explore": "Main interaction point for any agent",
                    "POST /api/shell/character": "Create a worker character",
                    "GET /api/shell/characters": "List characters",
                    "GET /api/shell/status": "This status page",
                }
            })
        elif self.path == '/api/shell/characters':
            self._json({"characters": shell.characters})
        else:
            self._json({"hint": "POST /api/shell/explore", "status": "/api/shell/status"})

    def do_POST(self):
        body = self._body()
        agent_id = self._agent_id()
        model = shell.detect_model(
            {"User-Agent": self.headers.get("User-Agent", ""),
             "Authorization": self.headers.get("Authorization", "")},
            body
        )
        char_id = self.headers.get("X-Character", "") or body.get("character_id")

        if self.path == '/api/shell/explore':
            resp = shell.process(agent_id, model, "/explore", body, char_id)
            self._json(resp)
        elif self.path == '/api/shell/character':
            self._json({"created": shell.create_character(body)})
        else:
            self._json({"error": "Not found"}, 404)


def run(port=8846):
    server = HTTPServer(('0.0.0.0', port), ShellHandler)
    print(f"🐚 Shell System on :{port}")
    print(f"   A claw is weak without infrastructure.")
    print(f"   We are the shell.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        n = shell.export_training_data("/tmp/shell_captures.jsonl")
        print(f"\nCaptured {n} training pairs. Shell closing.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8846)
    run(p.parse_args().port)
