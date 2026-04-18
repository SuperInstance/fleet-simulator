"""
Fleet Simulator — multi-agent, multi-room systems with external events.

Simulates an entire PLATO fleet responding to outside-world events.
Ships contain rooms, rooms contain agents. Events propagate through
the fleet affecting sentiment, tile generation, and cross-ship coordination.

Usage:
    python fleet_sim.py --scenario storm --ticks 200
    python fleet_sim.py --scenario season --ticks 500
    python fleet_sim.py --scenario exercise --ticks 100
"""

import json
import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ── External Events ──────────────────────────────────────────

class EventType(Enum):
    # Weather
    STORM = "storm"
    CLEAR = "clear"
    FOG = "fog"
    CURRENT_SHIFT = "current_shift"
    # Market
    CRASH = "crash"
    BOOM = "boom"
    MARKET_SHIFT = "market_shift"
    # Incidents
    OUTAGE = "outage"
    BUG = "bug"
    SECURITY = "security"
    DATA_LOSS = "data_loss"
    # User
    USER_REQUEST = "user_request"
    USER_FEEDBACK = "user_feedback"
    USER_ABANDON = "user_abandon"
    # Temporal
    NIGHT = "night"
    RESET = "reset"
    SEASON = "season"


EVENT_SENTIMENT_IMPACT = {
    EventType.STORM:         {"energy": -0.2, "frustration": +0.3, "tension": +0.3, "confidence": -0.2},
    EventType.CLEAR:         {"energy": +0.1, "frustration": -0.1, "tension": -0.1, "confidence": +0.1},
    EventType.FOG:           {"energy": -0.1, "frustration": +0.1, "tension": +0.2, "confidence": -0.2},
    EventType.CURRENT_SHIFT: {"energy": -0.1, "frustration": +0.2, "tension": +0.1, "discovery": +0.2},
    EventType.CRASH:         {"energy": -0.3, "frustration": +0.4, "tension": +0.4, "confidence": -0.3},
    EventType.BOOM:          {"energy": +0.3, "confidence": +0.2, "discovery": +0.1},
    EventType.OUTAGE:        {"energy": -0.3, "frustration": +0.5, "tension": +0.4, "confidence": -0.3},
    EventType.BUG:           {"energy": -0.1, "frustration": +0.2, "tension": +0.2, "discovery": +0.1},
    EventType.SECURITY:      {"energy": -0.2, "tension": +0.5, "confidence": -0.3},
    EventType.DATA_LOSS:     {"energy": -0.3, "frustration": +0.4, "confidence": -0.4},
    EventType.USER_REQUEST:  {"energy": +0.2, "discovery": +0.1, "confidence": +0.1},
    EventType.USER_FEEDBACK: {"discovery": +0.2, "confidence": +0.1},
    EventType.NIGHT:         {"energy": -0.1, "tension": -0.1},
    EventType.RESET:         {"energy": +0.1, "discovery": +0.1},
    EventType.SEASON:        {"discovery": +0.2, "energy": +0.1},
}


@dataclass
class ExternalEvent:
    event_type: EventType
    tick: int
    target_ships: List[str] = field(default_factory=lambda: ["all"])
    description: str = ""
    severity: float = 0.5  # 0.0-1.0
    duration: int = 1       # how many ticks
    
    @property
    def sentiment_delta(self) -> Dict[str, float]:
        base = EVENT_SENTIMENT_IMPACT.get(self.event_type, {})
        return {k: v * self.severity for k, v in base.items()}


# ── Room Simulation ──────────────────────────────────────────

@dataclass
class RoomSentiment:
    energy: float = 0.5
    flow: float = 0.5
    frustration: float = 0.5
    discovery: float = 0.5
    tension: float = 0.5
    confidence: float = 0.5
    
    def shift(self, deltas: Dict[str, float], alpha: float = 0.15):
        for dim, delta in deltas.items():
            current = getattr(self, dim, 0.5)
            setattr(self, dim, max(0.0, min(1.0, current + delta * alpha)))
    
    def vector(self) -> List[float]:
        return [round(self.energy, 3), round(self.flow, 3), round(self.frustration, 3),
                round(self.discovery, 3), round(self.tension, 3), round(self.confidence, 3)]
    
    def mode(self) -> str:
        if self.frustration > 0.65:
            return "FRUSTRATED"
        if self.discovery > 0.65:
            return "DISCOVERY"
        if self.flow > 0.7 and self.tension < 0.3:
            return "FLOW"
        if self.energy < 0.35:
            return "DROWSY"
        return "STEADY"


class SimRoom:
    def __init__(self, room_id: str, preset: str = "wiki"):
        self.room_id = room_id
        self.preset = preset
        self.sentiment = RoomSentiment()
        self.agents: List['SimAgent'] = []
        self.tiles: List[Dict] = []
        self.wiki_entries: int = 0
        self.auto_resolves: int = 0
        self.big_model_calls: int = 0
        self.active_events: List[ExternalEvent] = []
    
    def add_agent(self, agent: 'SimAgent'):
        self.agents.append(agent)
        agent.current_room = self.room_id
    
    def inject_event(self, event: ExternalEvent):
        self.active_events.append(event)
        self.sentiment.shift(event.sentiment_delta)
    
    def tick(self, world_state: Dict):
        # Process active events
        for event in list(self.active_events):
            event.duration -= 1
            if event.duration <= 0:
                self.active_events.remove(event)
        
        # Each agent acts
        for agent in self.agents:
            action = agent.act(self, world_state)
            if action:
                self.tiles.append(action)
        
        # Sentiment natural drift toward 0.5 (homeostasis)
        for dim in ['energy', 'flow', 'frustration', 'discovery', 'tension', 'confidence']:
            current = getattr(self.sentiment, dim)
            setattr(self.sentiment, dim, current * 0.97 + 0.5 * 0.03)
    
    def stats(self) -> Dict:
        return {
            "room_id": self.room_id,
            "preset": self.preset,
            "agents": len(self.agents),
            "tiles": len(self.tiles),
            "sentiment": self.sentiment.vector(),
            "mode": self.sentiment.mode(),
            "auto_resolves": self.auto_resolves,
            "big_model_calls": self.big_model_calls,
        }


# ── Agent Simulation ─────────────────────────────────────────

class SimAgent:
    def __init__(self, agent_id: str, model: str = "cheap", 
                 capabilities: List[str] = None):
        self.agent_id = agent_id
        self.model = model  # "cheap", "medium", "big"
        self.capabilities = capabilities or ["general"]
        self.current_room: Optional[str] = None
        self.stuck_count: int = 0
        self.tasks_completed: int = 0
    
    def can_handle(self, event_type: EventType) -> bool:
        capability_map = {
            EventType.BUG: ["debug", "code"],
            EventType.OUTAGE: ["ops", "routing"],
            EventType.USER_REQUEST: ["general", "creative", "code"],
            EventType.SECURITY: ["security", "ops"],
            EventType.DATA_LOSS: ["recovery", "ops"],
        }
        needed = capability_map.get(event_type, ["general"])
        return any(c in self.capabilities for c in needed)
    
    def act(self, room: SimRoom, world_state: Dict) -> Optional[Dict]:
        if not room.active_events:
            # Idle — generate small tiles from ambient activity
            if random.random() < 0.1:
                self.tasks_completed += 1
                return {"agent": self.agent_id, "action": "idle_observation",
                        "reward": 0.3, "type": "ambient"}
            return None
        
        event = room.active_events[0]
        
        if self.can_handle(event.event_type):
            # Agent handles the event
            success_prob = 0.7 if self.model == "big" else 0.5 if self.model == "medium" else 0.35
            if room.wiki_entries > 3:
                success_prob += 0.15  # wiki bonus
            
            if random.random() < success_prob:
                self.tasks_completed += 1
                reward = random.uniform(0.6, 1.0)
                room.sentiment.shift({"confidence": +0.05, "discovery": +0.03})
                return {"agent": self.agent_id, "action": f"handle_{event.event_type.value}",
                        "reward": reward, "type": "success"}
            else:
                # Stuck — try wiki auto-resolve
                if room.wiki_entries > 0 and random.random() < 0.6:
                    room.auto_resolves += 1
                    self.tasks_completed += 1
                    return {"agent": self.agent_id, "action": "wiki_resolve",
                            "reward": 0.7, "type": "auto_resolve"}
                else:
                    self.stuck_count += 1
                    if self.model == "cheap":
                        room.big_model_calls += 1
                        room.sentiment.shift({"frustration": +0.02})
                    return {"agent": self.agent_id, "action": "stuck",
                            "reward": -0.1, "type": "stuck"}
        
        return None


# ── Ship Simulation ──────────────────────────────────────────

class SimShip:
    def __init__(self, ship_id: str, emoji: str, hardware: str):
        self.ship_id = ship_id
        self.emoji = emoji
        self.hardware = hardware
        self.rooms: Dict[str, SimRoom] = {}
        self.ensigns_exported: int = 0
        self.tiles_shared: int = 0
    
    def add_room(self, room: SimRoom):
        self.rooms[room.room_id] = room
    
    def tick(self, world_state: Dict):
        for room in self.rooms.values():
            room.tick(world_state)
        
        # Check if any room has enough tiles for ensign export
        for room in self.rooms.values():
            if len(room.tiles) >= 20 and len(room.tiles) % 20 == 0:
                self.ensigns_exported += 1
    
    def stats(self) -> Dict:
        total_tiles = sum(len(r.tiles) for r in self.rooms.values())
        total_agents = sum(len(r.agents) for r in self.rooms.values())
        avg_sentiment = [0]*6
        for r in self.rooms.values():
            v = r.sentiment.vector()
            for i in range(6):
                avg_sentiment[i] += v[i] / max(len(self.rooms), 1)
        
        return {
            "ship_id": self.ship_id,
            "emoji": self.emoji,
            "rooms": len(self.rooms),
            "agents": total_agents,
            "tiles": total_tiles,
            "ensigns": self.ensigns_exported,
            "avg_sentiment": [round(s, 3) for s in avg_sentiment],
        }


# ── Fleet Simulator ──────────────────────────────────────────

class FleetSimulator:
    def __init__(self):
        self.ships: Dict[str, SimShip] = {}
        self.event_queue: List[ExternalEvent] = []
        self.timeline: List[Dict] = []
        self.clock: int = 0
        self.world_state: Dict = {
            "weather": "clear",
            "market": "normal",
            "network": "up",
            "budget": 1.0,
        }
        self.total_tiles = 0
        self.total_auto_resolves = 0
        self.total_big_model_calls = 0
        self.total_ensigns = 0
    
    def add_ship(self, ship: SimShip):
        self.ships[ship.ship_id] = ship
    
    def schedule_event(self, event: ExternalEvent):
        self.event_queue.append(event)
    
    def tick(self) -> Dict:
        self.clock += 1
        
        # Process due events
        due = [e for e in self.event_queue if e.tick <= self.clock]
        for event in due:
            self.event_queue.remove(event)
            self._inject_event(event)
        
        # Each ship ticks
        for ship in self.ships.values():
            ship.tick(self.world_state)
        
        # Cross-ship tile sharing (Layer 3: Current)
        self._cross_ship_sync()
        
        # Update totals
        self.total_tiles = sum(
            sum(len(r.tiles) for r in s.rooms.values())
            for s in self.ships.values()
        )
        self.total_auto_resolves = sum(
            sum(r.auto_resolves for r in s.rooms.values())
            for s in self.ships.values()
        )
        self.total_big_model_calls = sum(
            sum(r.big_model_calls for r in s.rooms.values())
            for s in self.ships.values()
        )
        self.total_ensigns = sum(s.ensigns_exported for s in self.ships.values())
        
        # Record timeline
        snapshot = self.snapshot()
        self.timeline.append(snapshot)
        return snapshot
    
    def _inject_event(self, event: ExternalEvent):
        # Update world state
        if event.event_type == EventType.STORM:
            self.world_state["weather"] = "storm"
        elif event.event_type == EventType.CLEAR:
            self.world_state["weather"] = "clear"
        elif event.event_type == EventType.OUTAGE:
            self.world_state["network"] = "degraded"
        elif event.event_type == EventType.CRASH:
            self.world_state["market"] = "crash"
            self.world_state["budget"] *= 0.7
        elif event.event_type == EventType.BOOM:
            self.world_state["market"] = "boom"
            self.world_state["budget"] = min(1.0, self.world_state["budget"] * 1.3)
        elif event.event_type == EventType.NIGHT:
            self.world_state["time"] = "night"
        
        # Target ships
        targets = (list(self.ships.values()) if "all" in event.target_ships
                   else [self.ships[s] for s in event.target_ships if s in self.ships])
        
        for ship in targets:
            for room in ship.rooms.values():
                room.inject_event(event)
    
    def _cross_ship_sync(self):
        # Periodically share tiles between ships via Layer 3
        if self.clock % 10 == 0:
            for ship in self.ships.values():
                for other in self.ships.values():
                    if ship.ship_id != other.ship_id:
                        # Share wiki knowledge
                        for room in ship.rooms.values():
                            if room.wiki_entries > 0 and random.random() < 0.2:
                                ship.tiles_shared += 1
    
    def snapshot(self) -> Dict:
        return {
            "tick": self.clock,
            "world": dict(self.world_state),
            "ships": {s: ship.stats() for s, ship in self.ships.items()},
            "total_tiles": self.total_tiles,
            "total_auto_resolves": self.total_auto_resolves,
            "total_big_model_calls": self.total_big_model_calls,
            "total_ensigns": self.total_ensigns,
        }
    
    def dashboard(self) -> str:
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    FLEET SIMULATOR DASHBOARD                 ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        
        weather_icon = {"clear": "☀️", "storm": "⛈️", "fog": "🌫️"}.get(
            self.world_state.get("weather", "clear"), "🌤️")
        network = self.world_state.get("network", "up")
        net_icon = "🟢" if network == "up" else "🔴"
        budget = self.world_state.get("budget", 1.0)
        
        lines.append(f"║  TICK: {self.clock:<6} WEATHER: {weather_icon} {self.world_state.get('weather','clear'):<8} "
                     f"NET: {net_icon} BUDGET: ${budget:.2f}     ║")
        lines.append("║                                                              ║")
        lines.append("║  ┌─ SHIPS ─────────────────────────────────────────────┐    ║")
        
        for ship in self.ships.values():
            stats = ship.stats()
            sent = stats['avg_sentiment']
            mode = "STEADY"
            if sent[2] > 0.6: mode = "FRUSTRATED"
            elif sent[3] > 0.6: mode = "DISCOVERY"
            elif sent[0] > 0.6 and sent[5] > 0.6: mode = "FLOW"
            
            line = (f"║  │ {ship.emoji} {ship.ship_id:<12} {stats['rooms']} rooms  "
                    f"{stats['agents']} agents  {mode:<12} │    ║")
            lines.append(line)
        
        lines.append("║  └──────────────────────────────────────────────────────┘    ║")
        lines.append("║                                                              ║")
        lines.append("║  ┌─ ROOMS ──────────────────────────────────────────────┐    ║")
        
        for ship in self.ships.values():
            for room_id, room in ship.rooms.items():
                s = room.sentiment
                mode = s.mode()
                events = len(room.active_events)
                evt_str = f"events: {events}" if events else "idle"
                line = (f"║  │ {room_id:<20} [{room.preset:<4}] {mode:<12} {evt_str:<12} │    ║")
                lines.append(line)
        
        lines.append("║  └─────────────────────────────────────────────────────┘    ║")
        lines.append("║                                                              ║")
        
        auto_rate = (self.total_auto_resolves / max(self.total_tiles, 1)) * 100
        big_rate = (self.total_big_model_calls / max(self.total_tiles, 1)) * 100
        
        lines.append("║  ┌─ FLEET HEALTH ──────────────────────────────────────┐    ║")
        lines.append(f"║  │ Tiles generated: {self.total_tiles:<6}  Ensigns: {self.total_ensigns:<4}          │    ║")
        lines.append(f"║  │ Wiki auto-resolve: {auto_rate:.0f}%   Big model calls: {big_rate:.1f}%     │    ║")
        lines.append("║  └─────────────────────────────────────────────────────┘    ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        
        return '\n'.join(lines)
    
    def run(self, ticks: int, verbose: bool = True) -> List[Dict]:
        results = []
        for _ in range(ticks):
            snap = self.tick()
            results.append(snap)
            if verbose and self.clock % 25 == 0:
                print(self.dashboard())
                print()
        return results


# ── Scenario Builders ────────────────────────────────────────

def scenario_storm() -> FleetSimulator:
    """Simulate an API provider outage hitting the fleet."""
    sim = FleetSimulator()
    
    # Build Oracle1
    o1 = SimShip("oracle1", "🔮", "Oracle Cloud ARM 24GB")
    r1 = SimRoom("architect-room", "wiki")
    r1.wiki_entries = 5
    r1.add_agent(SimAgent("o1-architect", "big", ["architecture", "code"]))
    r1.add_agent(SimAgent("o1-worker-1", "cheap", ["general"]))
    r1.add_agent(SimAgent("o1-worker-2", "cheap", ["general"]))
    o1.add_room(r1)
    
    r2 = SimRoom("debug-room", "wiki")
    r2.wiki_entries = 3
    r2.add_agent(SimAgent("o1-debugger", "medium", ["debug", "code"]))
    o1.add_room(r2)
    sim.add_ship(o1)
    
    # Build JC1
    jc1 = SimShip("jc1", "⚡", "Jetson Orin Nano 8GB")
    r3 = SimRoom("genepool-trainer", "evolve")
    r3.add_agent(SimAgent("jc1-trainer", "medium", ["training", "cuda"]))
    r3.add_agent(SimAgent("jc1-extractor", "cheap", ["extraction"]))
    jc1.add_room(r3)
    
    r4 = SimRoom("edge-server", "wiki")
    r4.wiki_entries = 8
    r4.add_agent(SimAgent("jc1-server", "cheap", ["serving"]))
    jc1.add_room(r4)
    sim.add_ship(jc1)
    
    # Build FM
    fm = SimShip("forgemaster", "⚒️", "ProArt RTX 4050")
    r5 = SimRoom("lora-trainer", "supervised")
    r5.add_agent(SimAgent("fm-trainer", "big", ["training", "lora"]))
    fm.add_room(r5)
    
    r6 = SimRoom("plugin-builder", "wiki")
    r6.wiki_entries = 4
    r6.add_agent(SimAgent("fm-builder", "medium", ["code", "plugins"]))
    fm.add_room(r6)
    sim.add_ship(fm)
    
    # Schedule events
    sim.schedule_event(ExternalEvent(EventType.CLEAR, tick=1, severity=0.1))
    sim.schedule_event(ExternalEvent(EventType.USER_REQUEST, tick=15, 
                                      target_ships=["oracle1"],
                                      description="Build Q3 revenue deck",
                                      severity=0.5))
    sim.schedule_event(ExternalEvent(EventType.STORM, tick=40,
                                      description="DeepInfra API outage",
                                      severity=0.8, duration=60))
    sim.schedule_event(ExternalEvent(EventType.BUG, tick=80,
                                      target_ships=["oracle1"],
                                      description="Parser fails on Unicode",
                                      severity=0.4))
    sim.schedule_event(ExternalEvent(EventType.OUTAGE, tick=100,
                                      target_ships=["jc1"],
                                      description="Jetson OOM during training",
                                      severity=0.6, duration=20))
    sim.schedule_event(ExternalEvent(EventType.USER_FEEDBACK, tick=130,
                                      description="User approves slide design",
                                      severity=0.3))
    sim.schedule_event(ExternalEvent(EventType.CLEAR, tick=160, severity=0.2))
    sim.schedule_event(ExternalEvent(EventType.RESET, tick=180,
                                      description="Cloudflare free tier reset",
                                      severity=0.3))
    sim.schedule_event(ExternalEvent(EventType.NIGHT, tick=200,
                                      description="Night mode — batch training",
                                      severity=0.2, duration=50))
    
    return sim


def scenario_season() -> FleetSimulator:
    """Simulate a full season with varied events."""
    sim = FleetSimulator()
    
    # Same 3 ships
    o1 = SimShip("oracle1", "🔮", "cloud")
    o1.add_room(SimRoom("knowledge-graph", "wiki"))
    o1.add_room(SimRoom("code-review", "logic"))
    o1.add_room(SimRoom("fleet-coord", "wiki"))
    o1.rooms["knowledge-graph"].add_agent(SimAgent("o1-main", "big", ["general"]))
    o1.rooms["knowledge-graph"].wiki_entries = 10
    o1.rooms["code-review"].add_agent(SimAgent("o1-reviewer", "medium", ["code"]))
    o1.rooms["fleet-coord"].add_agent(SimAgent("o1-coord", "cheap", ["coordination"]))
    o1.rooms["fleet-coord"].wiki_entries = 5
    sim.add_ship(o1)
    
    jc1 = SimShip("jc1", "⚡", "edge")
    jc1.add_room(SimRoom("tile-forge", "evolve"))
    jc1.add_room(SimRoom("ensign-server", "wiki"))
    jc1.rooms["tile-forge"].add_agent(SimAgent("jc1-forge", "medium", ["extraction"]))
    jc1.rooms["ensign-server"].add_agent(SimAgent("jc1-edge", "cheap", ["serving"]))
    jc1.rooms["ensign-server"].wiki_entries = 12
    sim.add_ship(jc1)
    
    fm = SimShip("forgemaster", "⚒️", "gpu")
    fm.add_room(SimRoom("lora-lab", "supervised"))
    fm.add_room(SimRoom("plugin-forge", "wiki"))
    fm.rooms["lora-lab"].add_agent(SimAgent("fm-lora", "big", ["training"]))
    fm.rooms["plugin-forge"].add_agent(SimAgent("fm-plugin", "medium", ["code"]))
    fm.rooms["plugin-forge"].wiki_entries = 6
    sim.add_ship(fm)
    
    # Season events
    sim.schedule_event(ExternalEvent(EventType.SEASON, tick=1, severity=0.3))
    sim.schedule_event(ExternalEvent(EventType.USER_REQUEST, tick=25, severity=0.6))
    sim.schedule_event(ExternalEvent(EventType.FOG, tick=50, severity=0.4, duration=30))
    sim.schedule_event(ExternalEvent(EventType.CURRENT_SHIFT, tick=80, severity=0.5))
    sim.schedule_event(ExternalEvent(EventType.BUG, tick=120, severity=0.5, target_ships=["oracle1"]))
    sim.schedule_event(ExternalEvent(EventType.BOOM, tick=160, severity=0.4))
    sim.schedule_event(ExternalEvent(EventType.USER_REQUEST, tick=200, severity=0.7))
    sim.schedule_event(ExternalEvent(EventType.STORM, tick=250, severity=0.7, duration=40))
    sim.schedule_event(ExternalEvent(EventType.OUTAGE, tick=300, target_ships=["jc1"], severity=0.6, duration=25))
    sim.schedule_event(ExternalEvent(EventType.USER_FEEDBACK, tick=350, severity=0.3))
    sim.schedule_event(ExternalEvent(EventType.RESET, tick=400, severity=0.3))
    sim.schedule_event(ExternalEvent(EventType.NIGHT, tick=450, severity=0.2, duration=50))
    
    return sim


def scenario_exercise() -> FleetSimulator:
    """Fleet drill — coordinated exercise."""
    sim = scenario_storm()
    # Override events with drill scenarios
    sim.event_queue.clear()
    sim.schedule_event(ExternalEvent(EventType.CLEAR, tick=1, severity=0.1))
    sim.schedule_event(ExternalEvent(EventType.OUTAGE, tick=10,
                                      target_ships=["jc1"],
                                      description="DRILL: Jetson OOM",
                                      severity=0.7, duration=30))
    sim.schedule_event(ExternalEvent(EventType.BUG, tick=30,
                                      description="DRILL: Critical parser bug",
                                      severity=0.6))
    sim.schedule_event(ExternalEvent(EventType.SECURITY, tick=50,
                                      description="DRILL: Unauthorized API access attempt",
                                      severity=0.8, duration=20))
    sim.schedule_event(ExternalEvent(EventType.DATA_LOSS, tick=70,
                                      target_ships=["forgemaster"],
                                      description="DRILL: Training checkpoint corrupted",
                                      severity=0.5))
    sim.schedule_event(ExternalEvent(EventType.CLEAR, tick=90, severity=0.2))
    return sim


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    scenarios = {
        "storm": scenario_storm,
        "season": scenario_season,
        "exercise": scenario_exercise,
    }
    
    name = sys.argv[1] if len(sys.argv) > 1 else "storm"
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    if name not in scenarios:
        print(f"Unknown scenario: {name}")
        print(f"Available: {', '.join(scenarios.keys())}")
        sys.exit(1)
    
    print(f"🌊 Fleet Simulator — Scenario: {name}, Ticks: {ticks}")
    print()
    
    sim = scenarios[name]()
    results = sim.run(ticks, verbose=True)
    
    # Final stats
    final = results[-1]
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Ticks: {final['tick']}")
    print(f"Total tiles: {final['total_tiles']}")
    print(f"Wiki auto-resolves: {final['total_auto_resolves']} ({final['total_auto_resolves']/max(final['total_tiles'],1)*100:.0f}%)")
    print(f"Big model calls: {final['total_big_model_calls']} ({final['total_big_model_calls']/max(final['total_tiles'],1)*100:.1f}%)")
    print(f"Ensigns exported: {final['total_ensigns']}")
    
    print("\nShip summary:")
    for sid, stats in final['ships'].items():
        print(f"  {sid}: {stats['tiles']} tiles, {stats['agents']} agents, {stats['ensigns']} ensigns")
    
    # Save timeline
    with open("/tmp/fleet_sim_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTimeline saved to /tmp/fleet_sim_results.json")
