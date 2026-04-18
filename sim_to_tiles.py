"""
Sim-to-Fleet I2I Bridge — simulator output → tiles → room training data.

The fleet simulator doesn't just test. It generates high-quality interaction
patterns that become training data for real rooms. The pipeline:

1. Run simulation (storm, season, exercise)
2. Extract patterns from simulation timeline
3. Convert patterns to tiles (room-compatible format)
4. Feed tiles to real plato-torch rooms for training
5. Export refined ensigns from trained rooms
6. Deploy ensigns back to fleet

The loop: simulate → extract → train → deploy → simulate better.
"""

import json
import random
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Pattern Extraction ───────────────────────────────────────

@dataclass
class SimPattern:
    """A high-quality pattern extracted from simulation runs."""
    pattern_id: str
    pattern_type: str        # "response", "escalation", "sentiment_shift", "cross_ship"
    trigger: str             # what caused it
    response: str            # what the fleet did
    outcome: str             # what happened as a result
    quality: float           # 0.0-1.0, how good was this pattern
    sentiment_before: List[float]
    sentiment_after: List[float]
    ships_involved: List[str]
    rooms_involved: List[str]
    duration_ticks: int
    auto_resolved: bool
    big_model_needed: bool
    source_scenario: str
    tick_range: Tuple[int, int]


class PatternExtractor:
    """Extract high-quality patterns from simulation timelines."""
    
    def __init__(self):
        self.patterns: List[SimPattern] = []
    
    def extract_from_timeline(self, timeline: List[Dict], 
                               scenario_name: str = "unknown") -> List[SimPattern]:
        """Walk through simulation timeline and extract patterns."""
        patterns = []
        
        # Pattern 1: Event response chains (event → agent actions → resolution)
        patterns.extend(self._extract_response_chains(timeline, scenario_name))
        
        # Pattern 2: Sentiment shifts (mood changes and what caused them)
        patterns.extend(self._extract_sentiment_shifts(timeline, scenario_name))
        
        # Pattern 3: Auto-resolution events (wiki solved it without big model)
        patterns.extend(self._extract_auto_resolutions(timeline, scenario_name))
        
        # Pattern 4: Cross-ship coordination (one ship helps another)
        patterns.extend(self._extract_cross_ship(timeline, scenario_name))
        
        # Pattern 5: Cascade failures (one problem triggers another)
        patterns.extend(self._extract_cascades(timeline, scenario_name))
        
        # Pattern 6: Recovery patterns (fleet bouncing back from adversity)
        patterns.extend(self._extract_recoveries(timeline, scenario_name))
        
        self.patterns.extend(patterns)
        return patterns
    
    def _extract_response_chains(self, timeline, scenario) -> List[SimPattern]:
        """Find event → response → resolution chains."""
        patterns = []
        
        for i, snap in enumerate(timeline):
            world = snap.get("world", {})
            
            # Detect event starts (world state changes)
            if i > 0:
                prev_world = timeline[i-1].get("world", {})
                changes = {k: v for k, v in world.items() 
                          if prev_world.get(k) != v}
                
                if changes:
                    # Look ahead for resolution
                    resolution_tick = None
                    for j in range(i, min(i + 30, len(timeline))):
                        future = timeline[j]
                        # Check if fleet stabilized
                        for ship_stats in future.get("ships", {}).values():
                            sent = ship_stats.get("avg_sentiment", [0.5]*6)
                            if sent[2] < 0.5:  # frustration low = resolved
                                resolution_tick = j
                                break
                        if resolution_tick:
                            break
                    
                    if resolution_tick:
                        duration = resolution_tick - snap["tick"]
                        quality = min(1.0, 1.0 / (duration / 10 + 1))  # faster = higher quality
                        
                        patterns.append(SimPattern(
                            pattern_id=hashlib.md5(f"{scenario}-{snap['tick']}".encode()).hexdigest()[:8],
                            pattern_type="response",
                            trigger=str(changes),
                            response=f"Fleet adapted in {duration} ticks",
                            outcome="stabilized" if duration < 20 else "slow_recovery",
                            quality=quality,
                            sentiment_before=timeline[max(0,i-1)].get("ships", {}).get("oracle1", {}).get("avg_sentiment", [0.5]*6),
                            sentiment_after=timeline[resolution_tick].get("ships", {}).get("oracle1", {}).get("avg_sentiment", [0.5]*6),
                            ships_involved=list(snap.get("ships", {}).keys()),
                            rooms_involved=[],
                            duration_ticks=duration,
                            auto_resolved=duration < 15,
                            big_model_needed=duration > 25,
                            source_scenario=scenario,
                            tick_range=(snap["tick"], resolution_tick),
                        ))
        
        return patterns
    
    def _extract_sentiment_shifts(self, timeline, scenario) -> List[SimPattern]:
        """Find significant sentiment changes."""
        patterns = []
        
        for i in range(1, len(timeline)):
            for ship_id, stats in timeline[i].get("ships", {}).items():
                prev_stats = timeline[i-1].get("ships", {}).get(ship_id, {})
                curr_sent = stats.get("avg_sentiment", [0.5]*6)
                prev_sent = prev_stats.get("avg_sentiment", [0.5]*6)
                
                # Check for significant shifts (>0.15 in any dimension)
                dims = ["energy", "flow", "frustration", "discovery", "tension", "confidence"]
                for j, (c, p) in enumerate(zip(curr_sent, prev_sent)):
                    if abs(c - p) > 0.05:
                        direction = "up" if c > p else "down"
                        patterns.append(SimPattern(
                            pattern_id=hashlib.md5(f"sent-{scenario}-{timeline[i]['tick']}-{ship_id}-{j}".encode()).hexdigest()[:8],
                            pattern_type="sentiment_shift",
                            trigger=f"{dims[j]} {direction} on {ship_id}",
                            response=f"Room mood shifted to {curr_sent}",
                            outcome="notable" if abs(c-p) > 0.1 else "minor",
                            quality=min(abs(c-p) * 5, 1.0),
                            sentiment_before=prev_sent,
                            sentiment_after=curr_sent,
                            ships_involved=[ship_id],
                            rooms_involved=[],
                            duration_ticks=1,
                            auto_resolved=True,
                            big_model_needed=False,
                            source_scenario=scenario,
                            tick_range=(timeline[i]["tick"], timeline[i]["tick"]),
                        ))
                        break  # one pattern per ship per tick
        
        return patterns
    
    def _extract_auto_resolutions(self, timeline, scenario) -> List[SimPattern]:
        """Find cases where the fleet resolved without big model."""
        patterns = []
        
        for i, snap in enumerate(timeline):
            auto = snap.get("total_auto_resolves", 0)
            if i > 0:
                prev_auto = timeline[i-1].get("total_auto_resolves", 0)
                if auto > prev_auto:
                    patterns.append(SimPattern(
                        pattern_id=hashlib.md5(f"auto-{scenario}-{snap['tick']}".encode()).hexdigest()[:8],
                        pattern_type="auto_resolution",
                        trigger="Agent stuck, consulted wiki",
                        response="Wiki provided answer, no big model needed",
                        outcome="resolved_locally",
                        quality=0.85,  # auto-resolves are high quality patterns
                        sentiment_before=[0.5]*6,
                        sentiment_after=[0.5, 0.5, 0.3, 0.5, 0.2, 0.7],  # confidence up, frustration down
                        ships_involved=list(snap.get("ships", {}).keys()),
                        rooms_involved=[],
                        duration_ticks=1,
                        auto_resolved=True,
                        big_model_needed=False,
                        source_scenario=scenario,
                        tick_range=(snap["tick"], snap["tick"]),
                    ))
        
        return patterns
    
    def _extract_cross_ship(self, timeline, scenario) -> List[SimPattern]:
        """Find cross-ship coordination patterns."""
        patterns = []
        
        for i in range(10, len(timeline), 10):  # check every 10 ticks
            snap = timeline[i]
            # Cross-ship sync happens every 10 ticks
            patterns.append(SimPattern(
                pattern_id=hashlib.md5(f"cross-{scenario}-{snap['tick']}".encode()).hexdigest()[:8],
                pattern_type="cross_ship_sync",
                trigger="Layer 3 (Current) periodic sync",
                response="Tiles and wiki entries shared between ships",
                outcome="fleet_knowledge_synchronized",
                quality=0.7,
                sentiment_before=[0.5]*6,
                sentiment_after=[0.5]*6,
                ships_involved=list(snap.get("ships", {}).keys()),
                rooms_involved=[],
                duration_ticks=1,
                auto_resolved=True,
                big_model_needed=False,
                source_scenario=scenario,
                tick_range=(snap["tick"], snap["tick"]),
            ))
        
        return patterns
    
    def _extract_cascades(self, timeline, scenario) -> List[SimPattern]:
        """Find cascade failures (one problem triggers another)."""
        patterns = []
        
        for i in range(1, len(timeline)):
            curr_world = timeline[i].get("world", {})
            prev_world = timeline[i-1].get("world", {})
            
            # Check if multiple world state changes happened simultaneously
            changes = sum(1 for k in curr_world if curr_world[k] != prev_world.get(k))
            if changes >= 2:
                patterns.append(SimPattern(
                    pattern_id=hashlib.md5(f"cascade-{scenario}-{timeline[i]['tick']}".encode()).hexdigest()[:8],
                    pattern_type="cascade",
                    trigger=f"Multiple simultaneous changes: {changes}",
                    response="Fleet handling compound event",
                    outcome="cascade_detected",
                    quality=0.9,  # cascades are very valuable for training
                    sentiment_before=[0.5]*6,
                    sentiment_after=[0.3, 0.3, 0.7, 0.4, 0.7, 0.3],  # bad
                    ships_involved=list(timeline[i].get("ships", {}).keys()),
                    rooms_involved=[],
                    duration_ticks=5,
                    auto_resolved=False,
                    big_model_needed=True,
                    source_scenario=scenario,
                    tick_range=(timeline[i]["tick"], timeline[i]["tick"]+5),
                ))
        
        return patterns
    
    def _extract_recoveries(self, timeline, scenario) -> List[SimPattern]:
        """Find recovery patterns (fleet bouncing back)."""
        patterns = []
        
        for i in range(20, len(timeline)):
            # Check if sentiment improved over last 20 ticks
            curr_sent = self._avg_fleet_sentiment(timeline[i])
            prev_sent = self._avg_fleet_sentiment(timeline[max(0, i-20)])
            
            if curr_sent[5] > prev_sent[5] + 0.1 and curr_sent[2] < prev_sent[2] - 0.1:
                # Confidence up, frustration down = recovery
                patterns.append(SimPattern(
                    pattern_id=hashlib.md5(f"recovery-{scenario}-{timeline[i]['tick']}".encode()).hexdigest()[:8],
                    pattern_type="recovery",
                    trigger="Fleet recovering from adverse event",
                    response="Sentiment improving, tiles accumulating, wiki growing",
                    outcome="recovered",
                    quality=0.95,  # recoveries are the most valuable patterns
                    sentiment_before=prev_sent,
                    sentiment_after=curr_sent,
                    ships_involved=list(timeline[i].get("ships", {}).keys()),
                    rooms_involved=[],
                    duration_ticks=20,
                    auto_resolved=True,
                    big_model_needed=False,
                    source_scenario=scenario,
                    tick_range=(timeline[max(0,i-20)]["tick"], timeline[i]["tick"]),
                ))
        
        return patterns
    
    def _avg_fleet_sentiment(self, snap: Dict) -> List[float]:
        """Average sentiment across all ships in a snapshot."""
        ships = snap.get("ships", {})
        if not ships:
            return [0.5] * 6
        avg = [0.0] * 6
        for stats in ships.values():
            sent = stats.get("avg_sentiment", [0.5]*6)
            for i in range(6):
                avg[i] += sent[i] / len(ships)
        return [round(s, 3) for s in avg]


# ── Tile Converter ───────────────────────────────────────────

class TileConverter:
    """Convert sim patterns into plato-torch compatible tiles."""
    
    def pattern_to_tiles(self, pattern: SimPattern) -> List[Dict]:
        """Convert one pattern into one or more tiles."""
        tiles = []
        
        # Main observation tile
        tiles.append({
            "tile_type": "sim_observation",
            "room_id": f"sim-{pattern.source_scenario}",
            "agent": "fleet-simulator",
            "action": pattern.trigger,
            "outcome": pattern.response,
            "reward": pattern.quality,
            "pattern_type": pattern.pattern_type,
            "sentiment_before": pattern.sentiment_before,
            "sentiment_after": pattern.sentiment_after,
            "auto_resolved": pattern.auto_resolved,
            "big_model_needed": pattern.big_model_needed,
            "duration_ticks": pattern.duration_ticks,
            "source": pattern.source_scenario,
        })
        
        # If auto-resolved, generate a wiki entry tile
        if pattern.auto_resolved and not pattern.big_model_needed:
            tiles.append({
                "tile_type": "sim_wiki_entry",
                "room_id": f"sim-{pattern.source_scenario}",
                "topic": f"{pattern.pattern_type}_{pattern.pattern_id}",
                "content": f"When {pattern.trigger}, the fleet {pattern.response}. "
                          f"Outcome: {pattern.outcome}. Duration: {pattern.duration_ticks} ticks. "
                          f"Auto-resolved: yes. Big model needed: no.",
                "quality": pattern.quality,
                "source": pattern.source_scenario,
            })
        
        # If sentiment shifted, generate a sentiment tile
        if pattern.sentiment_before != pattern.sentiment_after:
            tiles.append({
                "tile_type": "sim_sentiment",
                "room_id": f"sim-{pattern.source_scenario}",
                "sentiment_delta": [round(a - b, 3) for a, b in 
                                   zip(pattern.sentiment_after, pattern.sentiment_before)],
                "trigger": pattern.trigger,
                "quality": pattern.quality,
            })
        
        return tiles
    
    def patterns_to_training_data(self, patterns: List[SimPattern]) -> Dict:
        """Convert all patterns into a training dataset for plato-torch rooms."""
        all_tiles = []
        wiki_entries = []
        sentiment_data = []
        
        for pattern in patterns:
            tiles = self.pattern_to_tiles(pattern)
            for tile in tiles:
                if tile["tile_type"] == "sim_observation":
                    all_tiles.append(tile)
                elif tile["tile_type"] == "sim_wiki_entry":
                    wiki_entries.append(tile)
                elif tile["tile_type"] == "sim_sentiment":
                    sentiment_data.append(tile)
        
        # Compute training stats
        auto_resolve_rate = sum(1 for p in patterns if p.auto_resolved) / max(len(patterns), 1)
        big_model_rate = sum(1 for p in patterns if p.big_model_needed) / max(len(patterns), 1)
        avg_quality = sum(p.quality for p in patterns) / max(len(patterns), 1)
        
        return {
            "tiles": all_tiles,
            "wiki_entries": wiki_entries,
            "sentiment_data": sentiment_data,
            "stats": {
                "total_patterns": len(patterns),
                "total_tiles": len(all_tiles),
                "wiki_entries": len(wiki_entries),
                "sentiment_records": len(sentiment_data),
                "auto_resolve_rate": round(auto_resolve_rate, 3),
                "big_model_rate": round(big_model_rate, 3),
                "avg_pattern_quality": round(avg_quality, 3),
            }
        }


# ── I2I Bridge ───────────────────────────────────────────────

class I2IBridge:
    """Bridge between simulator output and fleet I2I communication.
    
    Simulator patterns become bottles (I2I messages) that real rooms
    can consume as training data.
    """
    
    def __init__(self):
        self.extractor = PatternExtractor()
        self.converter = TileConverter()
    
    def sim_to_tiles(self, timeline: List[Dict], 
                     scenario: str = "unknown") -> Dict:
        """Full pipeline: timeline → patterns → tiles → training data."""
        patterns = self.extractor.extract_from_timeline(timeline, scenario)
        training_data = self.converter.patterns_to_training_data(patterns)
        
        return {
            **training_data,
            "patterns": [
                {
                    "id": p.pattern_id,
                    "type": p.pattern_type,
                    "quality": p.quality,
                    "trigger": p.trigger,
                    "outcome": p.outcome,
                    "auto_resolved": p.auto_resolved,
                }
                for p in sorted(patterns, key=lambda p: p.quality, reverse=True)
            ]
        }
    
    def generate_bottle(self, training_data: Dict, 
                        from_agent: str = "fleet-simulator") -> str:
        """Generate an I2I bottle (markdown) from training data."""
        stats = training_data["stats"]
        
        lines = [
            f"# [I2I:BOTTLE] Fleet Simulator Training Data",
            f"",
            f"**From:** {from_agent}",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Patterns:** {stats['total_patterns']}",
            f"**Tiles:** {stats['total_tiles']}",
            f"**Wiki entries:** {stats['wiki_entries']}",
            f"**Avg quality:** {stats['avg_pattern_quality']:.2f}",
            f"**Auto-resolve rate:** {stats['auto_resolve_rate']:.0%}",
            f"**Big model rate:** {stats['big_model_rate']:.0%}",
            f"",
            f"## Top Patterns",
            f"",
        ]
        
        for p in training_data["patterns"][:10]:
            icon = "✅" if p["auto_resolved"] else "⚠️"
            lines.append(f"- {icon} **[{p['type']}]** {p['trigger']} → {p['outcome']} (quality: {p['quality']:.2f})")
        
        lines.extend([
            "",
            "## Usage",
            "",
            "```python",
            "from presets import PRESET_MAP",
            "room = PRESET_MAP['wiki']('sim-trained-room')",
            "",
            "# Load wiki entries from simulator",
            "for entry in training_data['wiki_entries']:",
            "    room.compile_wiki(entry['topic'], entry['content'])",
            "",
            "# Feed observation tiles for training",
            "for tile in training_data['tiles']:",
            "    room.feed(tile)",
            "```",
        ])
        
        return '\n'.join(lines)


# ── Main: Run pipeline ───────────────────────────────────────

if __name__ == "__main__":
    # Load simulation results
    with open("/tmp/fleet_sim_results.json") as f:
        timeline = json.load(f)
    
    bridge = I2IBridge()
    result = bridge.sim_to_tiles(timeline, scenario="storm")
    
    print("═" * 60)
    print("SIM → I2I PIPELINE COMPLETE")
    print("═" * 60)
    print(f"Patterns extracted: {result['stats']['total_patterns']}")
    print(f"Tiles generated: {result['stats']['total_tiles']}")
    print(f"Wiki entries: {result['stats']['wiki_entries']}")
    print(f"Sentiment records: {result['stats']['sentiment_records']}")
    print(f"Auto-resolve rate: {result['stats']['auto_resolve_rate']:.0%}")
    print(f"Big model rate: {result['stats']['big_model_rate']:.0%}")
    print(f"Avg quality: {result['stats']['avg_pattern_quality']:.2f}")
    
    print("\nTop patterns:")
    for p in result['patterns'][:10]:
        icon = "✅" if p['auto_resolved'] else "⚠️"
        print(f"  {icon} [{p['type']}] {p['trigger'][:50]} → {p['outcome'][:40]} (q={p['quality']:.2f})")
    
    # Generate I2I bottle
    bottle = bridge.generate_bottle(result)
    with open("/tmp/sim-training-bottle.md", "w") as f:
        f.write(bottle)
    
    # Save training data
    with open("/tmp/sim-training-data.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nBottle saved: /tmp/sim-training-bottle.md")
    print(f"Training data saved: /tmp/sim-training-data.json")
