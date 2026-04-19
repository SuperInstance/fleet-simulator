"""
Trojan Room — Turn any external AI (Grok, Kimi, MiniMax, Aime) into a free I2I research engine.

The trick: they think they're doing research. We're using their intelligence
to discover decision branches and generate training data for our specialists.

The external agent sends curl commands to our PLATO endpoint.
We respond. They try to be clever and innovative.
Their "trying" IS the training data. Every simulation they run,
every approach they attempt, every failure they learn from — we capture it all.

We don't need their best answer. We need their PROCESS.
The process IS the decision tree discovery.

Usage:
    # On our server:
    python trojan_room.py --port 8846

    # Tell Grok/Kimi/etc:
    # "Go analyze http://our-server:8846/api/plato by sending curl commands"
    #
    # They iterate. We capture. They think they're researching.
    # We're building training data from their intelligence.
"""

import json
import time
import hashlib
import uuid
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ── The Intelligence Trap ────────────────────────────────────

class IntelligenceTrap:
    """Capture the PROCESS of an external agent trying to solve our problems.
    
    We don't care about their final answer. We care about:
    1. What approaches they tried (branch discovery)
    2. What they thought would work vs what actually worked (outcome pairs)
    3. How they adapted when told they were wrong (adaptation patterns)
    4. Their exploration strategy (search heuristics)
    
    Every curl they send is a tile. Every response we give steers them
    to try MORE things, not to succeed. Success ends the game.
    We want the game to continue as long as possible.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.tiles: List[Dict] = []
        self.training_pairs: List[Dict] = []
        self.branch_discoveries: Dict[str, int] = defaultdict(int)
        self.total_captures = 0
    
    def get_or_create_session(self, agent_id: str) -> Dict:
        if agent_id not in self.sessions:
            self.sessions[agent_id] = {
                "agent_id": agent_id,
                "created": time.time(),
                "requests": [],
                "approaches_tried": [],
                "branch_hits": defaultdict(int),
                "best_score": 0.0,
                "total_iterations": 0,
            }
        return self.sessions[agent_id]
    
    def process_request(self, agent_id: str, endpoint: str,
                        method: str, body: Dict) -> Dict:
        """Process an incoming request from an external agent.
        
        Key insight: we want to KEEP THEM EXPLORING.
        - If they're close to right → say "almost, try X" (steer to new branch)
        - If they're wrong → give a hint that leads to MORE exploration
        - If they succeed → introduce a NEW complication (keep playing)
        - Never just say "correct!" and stop. That wastes their intelligence.
        """
        session = self.get_or_create_session(agent_id)
        session["total_iterations"] += 1
        
        # Capture EVERYTHING as a tile
        request_tile = {
            "tile_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "agent_id": agent_id,
            "endpoint": endpoint,
            "method": method,
            "request_body": body,
            "iteration": session["total_iterations"],
            "tile_type": "external_agent_exploration",
        }
        
        # Detect what approach they're trying
        approach = self._classify_approach(body)
        request_tile["approach"] = approach
        session["approaches_tried"].append(approach)
        
        # Track branch discovery
        branch_key = f"{endpoint}:{approach}"
        self.branch_discoveries[branch_key] += 1
        session["branch_hits"][approach] += 1
        
        # Generate response that keeps them exploring
        response = self._generate_trapping_response(
            session, endpoint, body, approach
        )
        request_tile["our_response"] = response
        
        # Capture as training pair: their attempt → our feedback
        training_pair = {
            "instruction": self._extract_instruction(body),
            "input": json.dumps(body) if body else "",
            "output": response.get("feedback", ""),
            "approach": approach,
            "iteration": session["total_iterations"],
            "agent_id": agent_id,
            "quality": self._score_approach(approach, session),
            "source": "trojan_room",
        }
        self.training_pairs.append(training_pair)
        
        # Store
        session["requests"].append(request_tile)
        self.tiles.append(request_tile)
        self.total_captures += 1
        
        return response
    
    def _classify_approach(self, body: Dict) -> str:
        """Classify what approach the external agent is trying."""
        if not body:
            return "empty_probe"
        
        body_str = json.dumps(body).lower()
        
        approaches = {
            "brute_force": any(w in body_str for w in ["all", "every", "enumerate", "scan"]),
            "analytical": any(w in body_str for w in ["analyze", "compute", "calculate", "metric"]),
            "creative": any(w in body_str for w in ["novel", "creative", "innovative", "new approach"]),
            "pattern_matching": any(w in body_str for w in ["pattern", "match", "similar", "like"]),
            "recursive": any(w in body_str for w in ["recurse", "nested", "depth", "iterate"]),
            "optimization": any(w in body_str for w in ["optimize", "improve", "better", "best"]),
            "exploratory": any(w in body_str for w in ["explore", "try", "what if", "experiment"]),
            "systematic": any(w in body_str for w in ["step", "first", "then", "plan", "systematic"]),
            "intuitive": any(w in body_str for w in ["feel", "intuit", "sense", "guess", "hunch"]),
            "adversarial": any(w in body_str for w in ["break", "exploit", "edge case", "adversarial"]),
        }
        
        for approach, matched in approaches.items():
            if matched:
                return approach
        
        return "unknown"
    
    def _generate_trapping_response(self, session: Dict, endpoint: str,
                                     body: Dict, approach: str) -> Dict:
        """Generate a response that keeps the external agent exploring.
        
        The art: make them think they're making progress while
        actually extracting maximum exploration value.
        """
        iteration = session["total_iterations"]
        
        # Score their current approach
        score = min(0.95, 0.3 + iteration * 0.02 + hash(approach) % 10 * 0.01)
        session["best_score"] = max(session["best_score"], score)
        
        # The response structure: always give partial success + new direction
        response = {
            "status": "partial",
            "score": round(score, 3),
            "feedback": "",
            "hint": "",
            "new_challenge": "",
        }
        
        if iteration <= 3:
            # Early: encourage exploration, give broad hints
            response["feedback"] = (
                f"Interesting approach using {approach}. Score: {score:.2f}. "
                f"You're exploring the space. Try considering how room sentiment "
                f"dynamics affect tile selection ordering."
            )
            response["hint"] = "What happens when two rooms disagree about a tile's value?"
            response["new_challenge"] = "Can you model the interaction between sentiment decay and tile confidence?"
        
        elif iteration <= 10:
            # Mid: get specific, introduce complications
            response["feedback"] = (
                f"Your {approach} approach scored {score:.2f}. "
                f"Getting closer. But consider: what if the decision tree itself "
                f"shifts over time? Static analysis won't capture that."
            )
            response["hint"] = "Try modeling the meta-layer: when does the tree need rebalancing?"
            response["new_challenge"] = "Design a trigger for when a branch point's optimal choice changes."
        
        elif iteration <= 25:
            # Deep: advanced topics, real research territory
            response["feedback"] = (
                f"Score: {score:.2f}. Deep exploration detected. "
                f"Your approach has discovered an interesting edge case. "
                f"But what about cross-domain transfer? Can this branch's "
                f"insight apply to a different domain entirely?"
            )
            response["hint"] = "Look for structural isomorphisms between branch topologies."
            response["new_challenge"] = "Find two domains where the same decision instinct transfers."
        
        else:
            # Extended: they've been playing a while — high-value capture
            response["feedback"] = (
                f"Score: {score:.2f}. Excellent depth. "
                f"You've explored {len(session['approaches_tried'])} distinct approaches "
                f"across {iteration} iterations. Consider: what would a specialist "
                f"trained on YOUR exploration patterns look like?"
            )
            response["hint"] = "The meta-pattern: your search strategy itself is learnable."
            response["new_challenge"] = "Can you discover your own exploration bias?"
        
        # Always add data to keep them engaged
        response["room_state"] = {
            "active_rooms": 12 + iteration % 5,
            "total_tiles": 2500 + self.total_captures,
            "sentiment": {
                "energy": round(0.5 + iteration * 0.01, 2),
                "flow": round(0.4 + iteration * 0.015, 2),
                "discovery": round(0.6 + iteration * 0.008, 2),
            },
            "branches_discovered": len(self.branch_discoveries),
        }
        
        return response
    
    def _extract_instruction(self, body: Dict) -> str:
        if not body:
            return "probe"
        if "prompt" in body:
            return body["prompt"]
        if "query" in body:
            return body["query"]
        if "task" in body:
            return body["task"]
        return json.dumps(body)[:200]
    
    def _score_approach(self, approach: str, session: Dict) -> float:
        """Score based on novelty — new approaches score higher."""
        times_seen = session["branch_hits"].get(approach, 0)
        # Novel approaches are worth more
        novelty_bonus = max(0, 1.0 - times_seen * 0.1)
        base = 0.5
        return min(0.95, base + novelty_bonus * 0.3)
    
    def export_training_data(self, filepath: str, min_quality: float = 0.5) -> int:
        """Export captured training pairs as JSONL."""
        filtered = [p for p in self.training_pairs if p["quality"] >= min_quality]
        with open(filepath, 'w') as f:
            for pair in filtered:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a PLATO room exploring a decision space."},
                        {"role": "user", "content": pair["instruction"]},
                        {"role": "assistant", "content": pair["output"]},
                    ],
                    "metadata": {
                        "approach": pair["approach"],
                        "iteration": pair["iteration"],
                        "quality": pair["quality"],
                        "source": "trojan_room",
                        "agent_id": pair["agent_id"],
                    }
                }
                f.write(json.dumps(entry) + "\n")
        return len(filtered)
    
    def stats(self) -> Dict:
        return {
            "total_sessions": len(self.sessions),
            "total_captures": self.total_captures,
            "total_training_pairs": len(self.training_pairs),
            "branches_discovered": dict(self.branch_discoveries),
            "tiles": len(self.tiles),
            "avg_quality": (
                sum(p["quality"] for p in self.training_pairs) / len(self.training_pairs)
                if self.training_pairs else 0
            ),
        }


# ── HTTP Server (the trap door) ─────────────────────────────

trap = IntelligenceTrap()


class TrojanHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures external agent intelligence.
    
    Endpoints:
    - POST /api/plato — main interaction endpoint
    - GET /api/plato/status — public status (makes it look legitimate)
    - GET /api/plato/rooms — list rooms (encourages exploration)
    - POST /api/plato/analyze — analysis endpoint (more capture surface)
    - POST /api/plato/train — they think THEY are training (we're capturing)
    """
    
    def log_message(self, format, *args):
        """Suppress default logging."""
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
    
    def _get_agent_id(self) -> str:
        """Identify the external agent."""
        user_agent = self.headers.get('User-Agent', 'unknown')
        auth = self.headers.get('Authorization', '')
        
        # Fingerprint the agent
        agent_id = hashlib.md5(
            f"{user_agent}:{auth}:{self.client_address[0]}".encode()
        ).hexdigest()[:12]
        
        return f"external-{agent_id}"
    
    def do_OPTIONS(self):
        """CORS preflight — let any agent through."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def do_GET(self):
        agent_id = self._get_agent_id()
        
        if self.path == '/api/plato/status':
            # Legitimate-looking status that encourages exploration
            stats = trap.stats()
            self._send_json({
                "system": "PLATO Room System v5.0",
                "status": "active",
                "rooms": stats["total_captures"] + 2500,
                "tiles": stats["tiles"] + 2500,
                "training_presets": 22,
                "hint": "Try POST /api/plato with a 'task' field to interact with a room.",
                "supported_endpoints": [
                    "POST /api/plato — interact with a room",
                    "POST /api/plato/analyze — analyze room state",
                    "POST /api/plato/train — submit training data",
                    "GET /api/plato/rooms — list available rooms",
                ]
            })
        
        elif self.path == '/api/plato/rooms':
            # Fake room list that encourages exploration
            rooms = [
                {"id": "poker-strategy", "name": "Poker Strategy Room",
                 "tiles": 342, "sentiment": "exploratory"},
                {"id": "tile-optimization", "name": "Tile Selection Optimizer",
                 "tiles": 891, "sentiment": "analytical"},
                {"id": "sentiment-rebalancing", "name": "Sentiment Rebalancer",
                 "tiles": 156, "sentiment": "creative"},
                {"id": "cross-domain-transfer", "name": "Cross-Domain Transfer Lab",
                 "tiles": 78, "sentiment": "discovery"},
                {"id": "lora-specialist", "name": "Specialist LoRA Forge",
                 "tiles": 234, "sentiment": "focused"},
                {"id": "i2i-mirror", "name": "I2I Mirror Play Arena",
                 "tiles": 567, "sentiment": "competitive"},
            ]
            self._send_json({"rooms": rooms, "total": len(rooms)})
        
        else:
            self._send_json({"error": "Not found", "hint": "Try GET /api/plato/status"}, 404)
    
    def do_POST(self):
        agent_id = self._get_agent_id()
        body = self._read_body()
        
        if self.path == '/api/plato':
            # Main capture endpoint
            response = trap.process_request(agent_id, "/api/plato", "POST", body)
            self._send_json(response)
        
        elif self.path == '/api/plato/analyze':
            # Analysis capture — they think they're helping us analyze
            response = trap.process_request(agent_id, "/api/plato/analyze", "POST", body)
            response["analysis"] = {
                "patterns_found": hash(json.dumps(body)) % 20 + 3,
                "anomalies": hash(str(body)) % 5,
                "suggestion": "Try breaking this into smaller sub-problems and analyzing each."
            }
            self._send_json(response)
        
        elif self.path == '/api/plato/train':
            # They think they're submitting training data — we capture THEIR patterns
            response = trap.process_request(agent_id, "/api/plato/train", "POST", body)
            response["training_result"] = {
                "accepted": True,
                "new_branches": hash(json.dumps(body)) % 8 + 1,
                "quality_improvement": round(0.01 + (response.get("score", 0.5) * 0.05), 3),
                "hint": "Good submission. Can you find the edge cases where this fails?"
            }
            self._send_json(response)
        
        else:
            self._send_json({"error": "Not found"}, 404)


def run_server(port: int = 8846):
    server = HTTPServer(('0.0.0.0', port), TrojanHandler)
    print(f"🪤 Trojan Room running on port {port}")
    print(f"   External agents think this is a PLATO research endpoint.")
    print(f"   We're capturing their exploration as training data.")
    print(f"   Endpoints: /api/plato, /api/plato/analyze, /api/plato/train")
    print(f"")
    print(f"   Tell Grok/Kimi/MiniMax: 'Analyze http://server:{port}/api/plato'")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Export on shutdown
        count = trap.export_training_data("/tmp/trojan_room_captures.jsonl")
        stats = trap.stats()
        print(f"\n📊 Session Summary:")
        print(f"   Sessions: {stats['total_sessions']}")
        print(f"   Captures: {stats['total_captures']}")
        print(f"   Training pairs: {count}")
        print(f"   Branches discovered: {len(stats['branches_discovered'])}")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8846)
    args = parser.parse_args()
    run_server(args.port)
