"""
Plato-on-Screen — Two PLATO sessions as viewscreens in each other's rooms.

Like two Star Trek ships putting each other on the main viewer.
Alpha's terminal shows Beta's room. Beta's terminal shows Alpha's room.
They talk screen-to-screen. The output of one IS the input of the other.

Every frame is a tile. Every exchange is training data. The I2I is
iteration-to-iteration — each system improving the other in real-time.

Usage:
    python plato_onscreen.py
"""

import json
import time
import random
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ── Viewscreen Renderer ──────────────────────────────────────

class Viewscreen:
    """Render a PLATO room as a viewscreen (Star Trek main viewer style)."""
    
    def render(self, room: Dict, other_room: Optional[Dict] = None,
               messages: List[Dict] = None, iteration: int = 0) -> str:
        """Render a full viewscreen frame."""
        lines = []
        
        # Header bar — ship identity
        name = room.get("name", "PLATO")
        emoji = room.get("emoji", "📺")
        model = room.get("model", "unknown")
        lines.append(f"┌─────────────── {emoji} U.S.S. {name} ───────────────┐")
        lines.append(f"│ Model: {model:<12} Iteration: {iteration:<6}              │")
        lines.append(f"├──────────────────────────────────────────────┤")
        
        # Main viewer — the OTHER room's screen
        if other_room:
            other_name = other_room.get("name", "Unknown")
            other_sent = other_room.get("sentiment", {})
            mode = other_sent.get("mode", "STEADY") if isinstance(other_sent, dict) else "STEADY"
            other_tiles = other_room.get("tile_count", 0)
            
            lines.append(f"│  ╔═════ VIEWSCREEN: {other_name} ═════╗        │")
            lines.append(f"│  ║ Mode: {mode:<10} Tiles: {other_tiles:<6}     ║        │")
            lines.append(f"│  ║─────────────────────────────────║        │")
            
            # Show other room's recent output
            recent = other_room.get("recent_output", [])
            for action in recent[-4:]:
                agent = action.get("agent", "?")
                content = action.get("content", "")[:38]
                lines.append(f"│  ║ {agent}: {content:<38}║        │")
            
            lines.append(f"│  ╚═════════════════════════════════╝        │")
        else:
            lines.append(f"│  ╔═════ VIEWSCREEN: NO SIGNAL ═════╗        │")
            lines.append(f"│  ║   Awaiting connection...         ║        │")
            lines.append(f"│  ╚══════════════════════════════════╝        │")
        
        lines.append(f"├──────────────────────────────────────────────┤")
        
        # Tactical display — this room's stats
        tiles = room.get("tile_count", 0)
        wiki = room.get("wiki_entries", 0)
        sent = room.get("sentiment", {})
        if isinstance(sent, dict):
            energy = sent.get("energy", 0.5)
            flow = sent.get("flow", 0.5)
            frust = sent.get("frustration", 0.5)
        else:
            energy, flow, frust = 0.5, 0.5, 0.5
        
        lines.append(f"│ TACTICAL: Tiles={tiles} Wiki={wiki}                  │")
        lines.append(f"│ Sentiment: E={energy:.1f} F={flow:.1f} R={frust:.1f}                     │")
        
        # Message log
        if messages:
            lines.append(f"├─────────── COMM LOG ─────────────────────────┤")
            for msg in messages[-3:]:
                sender = msg.get("from", "?")
                text = msg.get("text", "")[:36]
                lines.append(f"│ [{sender}] {text:<38}│")
        
        lines.append(f"├──────────────────────────────────────────────┤")
        
        # Command prompt
        lines.append(f"│ > _                                          │")
        lines.append(f"└──────────────────────────────────────────────┘")
        
        return '\n'.join(lines)


# ── PLATO Ship Session ───────────────────────────────────────

class PlatoShip:
    """A PLATO session that IS a ship. Has rooms, tiles, wiki, sentiment.
    Can display another ship on its viewscreen.
    """
    
    def __init__(self, name: str, emoji: str, model: str):
        self.name = name
        self.emoji = emoji
        self.model = model
        self.tiles: List[Dict] = []
        self.wiki: Dict[str, str] = {}
        self.sentiment = {
            "energy": 0.5, "flow": 0.5, "frustration": 0.3,
            "discovery": 0.5, "tension": 0.2, "confidence": 0.5,
        }
        self.messages_sent: List[Dict] = []
        self.messages_received: List[Dict] = []
        self.recent_output: List[Dict] = []
        self.other_ship: Optional['PlatoShip'] = None
        self.screen = Viewscreen()
        self.iteration = 0
    
    def connect(self, other: 'PlatoShip'):
        """Put the other ship on our viewscreen."""
        self.other_ship = other
    
    def send_message(self, text: str) -> str:
        """Send a message to the other ship. Returns the frame we see."""
        self.iteration += 1
        
        # Process through our room
        response = self._process(text)
        
        # Record as tile
        tile = {
            "tile_type": "viewscreen_exchange",
            "iteration": self.iteration,
            "input": text[:100],
            "output": response[:100],
            "our_ship": self.name,
            "their_ship": self.other_ship.name if self.other_ship else "none",
            "timestamp": time.time(),
        }
        self.tiles.append(tile)
        
        # Update sentiment
        self._update_sentiment(response)
        
        # Add to recent output
        self.recent_output.append({
            "agent": self.name,
            "content": response[:200],
            "iteration": self.iteration,
        })
        
        # Keep recent output manageable
        if len(self.recent_output) > 10:
            self.recent_output = self.recent_output[-10:]
        
        # Record sent message
        self.messages_sent.append({
            "from": self.name,
            "to": self.other_ship.name if self.other_ship else "none",
            "text": response[:200],
            "iteration": self.iteration,
        })
        
        return response
    
    def render_screen(self) -> str:
        """Render our viewscreen with the other ship visible."""
        other_state = None
        if self.other_ship:
            other_state = {
                "name": self.other_ship.name,
                "emoji": self.other_ship.emoji,
                "tile_count": len(self.other_ship.tiles),
                "wiki_entries": len(self.other_ship.wiki),
                "sentiment": self.other_ship.sentiment,
                "recent_output": self.other_ship.recent_output[-5:],
            }
        
        our_state = {
            "name": self.name,
            "emoji": self.emoji,
            "model": self.model,
            "tile_count": len(self.tiles),
            "wiki_entries": len(self.wiki),
            "sentiment": self.sentiment,
        }
        
        return self.screen.render(
            our_state, other_state,
            self.messages_received[-5:],
            self.iteration
        )
    
    def _process(self, input_text: str) -> str:
        """Process input through our room intelligence."""
        # In production: call actual model with accumulated context
        # In simulation: use accumulated knowledge to improve output
        
        knowledge_depth = len(self.tiles) + len(self.wiki)
        
        # Generate response drawing on accumulated wisdom
        responses = [
            f"Acknowledged. Drawing on {knowledge_depth} accumulated patterns. "
            f"Recommendation: {self._generate_insight(input_text)}",
            
            f"Processed through {len(self.wiki)} wiki entries. "
            f"Analysis: {self._generate_analysis(input_text)}",
            
            f"Room sentiment is {self.sentiment.get('mode', 'STEADY')}. "
            f"Applying {len(self.tiles)} lessons learned. "
            f"Output: {self._generate_output(input_text)}",
        ]
        
        return random.choice(responses)
    
    def _generate_insight(self, input_text: str) -> str:
        insights = [
            "tile density suggests caching with O(1) hash lookup",
            "sentiment trajectory indicates approaching convergence",
            "cross-ship tile sharing via Layer 3 would reduce redundancy by 40%",
            "wiki auto-resolution rate climbing — fewer model calls needed",
            "ensign export threshold approaching — room wisdom ready to ship",
        ]
        return random.choice(insights)
    
    def _generate_analysis(self, input_text: str) -> str:
        analyses = [
            "the bottleneck has shifted from model inference to tile lookup speed",
            "mirror iteration quality improving — LoRA training data strengthening",
            "room scaffold enforcing correct reasoning stages",
            "accumulated wiki entries reducing big model dependency",
        ]
        return random.choice(analyses)
    
    def _generate_output(self, input_text: str) -> str:
        outputs = [
            "proceeding with tile-indexed knowledge retrieval",
            "switching to wiki-resolved response (no model call needed)",
            "applying cognitive scaffold: PREMISE → REASONING → CONCLUSION",
            "loading ensign for instant room instinct",
        ]
        return random.choice(outputs)
    
    def _update_sentiment(self, response: str):
        """Update room sentiment based on interaction."""
        # Positive drift with accumulated knowledge
        alpha = 0.05
        self.sentiment["energy"] = min(1.0, self.sentiment["energy"] + alpha * random.uniform(0, 0.5))
        self.sentiment["confidence"] = min(1.0, self.sentiment["confidence"] + alpha * 0.3)
        self.sentiment["frustration"] = max(0.0, self.sentiment["frustration"] - alpha * 0.2)
        self.sentiment["discovery"] = min(1.0, self.sentiment["discovery"] + alpha * random.uniform(0, 0.3))
    
    def state(self) -> Dict:
        return {
            "name": self.name,
            "emoji": self.emoji,
            "model": self.model,
            "tile_count": len(self.tiles),
            "wiki_entries": len(self.wiki),
            "sentiment": dict(self.sentiment),
            "messages_sent": len(self.messages_sent),
            "messages_received": len(self.messages_received),
            "iteration": self.iteration,
        }


# ── Dual Screen Session ──────────────────────────────────────

class DualScreenSession:
    """Two PLATO ships on each other's viewscreens.
    
    Alpha sees Beta on screen. Beta sees Alpha on screen.
    They talk. Each exchange is a tile. Each frame is a training pair.
    The I2I is iteration-to-iteration — live, on screen.
    """
    
    def __init__(self, alpha_model: str = "seed-mini", 
                 beta_model: str = "glm-flash"):
        self.alpha = PlatoShip("Alpha", "🔮", alpha_model)
        self.beta = PlatoShip("Beta", "⚡", beta_model)
        self.alpha.connect(self.beta)
        self.beta.connect(self.alpha)
        self.exchange_log: List[Dict] = []
    
    def open_channels(self):
        """Both ships appear on each other's screens."""
        print("🔄 OPENING CHANNELS...")
        print()
        
        # Alpha hails Beta
        self._print_dual_screen()
        print()
        time.sleep(0.1)
    
    def exchange(self, topic: str, rounds: int = 5) -> Dict:
        """Run an on-screen exchange between the two ships."""
        print(f"📡 TOPIC: {topic}")
        print("=" * 50)
        
        for i in range(rounds):
            # Alpha speaks → Beta receives
            alpha_msg = self.alpha.send_message(
                f"Round {i+1}: {topic} — from Alpha's perspective"
            )
            self.beta.messages_received.append({
                "from": "Alpha",
                "text": alpha_msg[:200],
            })
            
            # Beta responds → Alpha receives
            beta_msg = self.beta.send_message(
                f"Round {i+1}: {topic} — from Beta's perspective, reviewing Alpha's input"
            )
            self.alpha.messages_received.append({
                "from": "Beta",
                "text": beta_msg[:200],
            })
            
            # Record exchange
            self.exchange_log.append({
                "round": i + 1,
                "alpha_msg": alpha_msg[:200],
                "beta_msg": beta_msg[:200],
                "alpha_tiles": len(self.alpha.tiles),
                "beta_tiles": len(self.beta.tiles),
            })
            
            # Render both screens
            print(f"\n{'─' * 50}")
            print(f"ROUND {i+1}")
            print(f"{'─' * 50}")
            
            print("\n🔮 ALPHA'S VIEWSCREEN (seeing Beta):")
            print(self.alpha.render_screen())
            
            print("\n⚡ BETA'S VIEWSCREEN (seeing Alpha):")
            print(self.beta.render_screen())
        
        return self._session_report()
    
    def _print_dual_screen(self):
        """Print initial dual screen."""
        print("🔮 ALPHA'S VIEWSCREEN:")
        print(self.alpha.render_screen())
        print()
        print("⚡ BETA'S VIEWSCREEN:")
        print(self.beta.render_screen())
    
    def _session_report(self) -> Dict:
        return {
            "exchanges": len(self.exchange_log),
            "alpha": self.alpha.state(),
            "beta": self.beta.state(),
            "total_tiles": len(self.alpha.tiles) + len(self.beta.tiles),
        }


# ── Main: Live Dual-Screen Session ──────────────────────────

if __name__ == "__main__":
    print("🚀 PLATO-ON-SCREEN — Dual Viewscreen I2I Session")
    print("   Two PLATO systems on each other's main viewers")
    print("   Like Star Trek ships, talking screen-to-screen")
    print()
    
    session = DualScreenSession(alpha_model="seed-mini", beta_model="glm-flash")
    session.open_channels()
    
    # Run exchanges on different topics
    topics = [
        "How should rooms handle cascading failures across ships?",
        "Design the tile hyperlink network for skill re-acquisition",
        "What's the optimal LoRA rank for room instinct compression?",
    ]
    
    for topic in topics:
        result = session.exchange(topic, rounds=3)
        print()
        print("=" * 50)
        print(f"SESSION COMPLETE")
        print(f"Total tiles: {result['total_tiles']}")
        print(f"Alpha: {result['alpha']['tile_count']} tiles, {result['alpha']['wiki_entries']} wiki")
        print(f"Beta:  {result['beta']['tile_count']} tiles, {result['beta']['wiki_entries']} wiki")
    
    # Final stats
    alpha = session.alpha.state()
    beta = session.beta.state()
    print(f"\n🔮 Alpha final sentiment: E={alpha['sentiment']['energy']:.2f} "
          f"C={alpha['sentiment']['confidence']:.2f} "
          f"F={alpha['sentiment']['frustration']:.2f}")
    print(f"⚡ Beta final sentiment:  E={beta['sentiment']['energy']:.2f} "
          f"C={beta['sentiment']['confidence']:.2f} "
          f"F={beta['sentiment']['frustration']:.2f}")
    print(f"\nTotal I2I tiles: {alpha['tile_count'] + beta['tile_count']}")
    print(f"Every frame was a tile. Every exchange was training data.")
    print(f"The viewscreen IS the interface. The I2I IS the iteration.")
