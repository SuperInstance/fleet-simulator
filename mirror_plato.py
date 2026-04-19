"""
Mirror Plato — Two PLATO systems iterating improvements through each other.

Alpha runs as an avatar in Beta's room.
Beta runs as an avatar in Alpha's room.
They iterate screen-to-screen (TUI), each reviewing the other's output.
Every iteration generates tiles. Run until intelligence threshold or budget exhausted.

Usage:
    python mirror_plato.py --task "Write a function that sorts a list" --threshold 0.80
    python mirror_plato.py --task "Design a caching system" --max-iterations 50 --model-alpha seed-mini --model-beta glm-flash
"""

import json
import time
import hashlib
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── TUI Renderer ─────────────────────────────────────────────

class PlatoTUI:
    """TUI wrapper — all PLATO interactions are text I/O."""
    
    def render_room(self, room_state: Dict) -> str:
        """Render room state as TUI screen."""
        lines = []
        name = room_state.get('room_name', 'Unknown Room')
        sent = room_state.get('sentiment', {})
        mode = sent.get('mode', 'STEADY') if isinstance(sent, dict) else 'STEADY'
        
        lines.append(f"╔══ {name} ══╗")
        lines.append(f"  Mode: {mode} | Tiles: {room_state.get('tile_count', 0)} | Wiki: {room_state.get('wiki_entries', 0)}")
        lines.append(f"  ─────────────────────────")
        
        for action in room_state.get('recent_actions', [])[-5:]:
            agent = action.get('agent', '?')
            act = action.get('action', '?')
            result = action.get('result', '')
            lines.append(f"  > {agent}: {act}")
            if result:
                lines.append(f"    {result[:80]}")
        
        lines.append(f"  ─────────────────────────")
        exits = room_state.get('exits', ['mirror'])
        lines.append(f"  Exits: {', '.join(exits)}")
        lines.append(f"╚{'═' * 30}╝")
        
        return '\n'.join(lines)
    
    def render_output(self, output: str, quality: float, iteration: int) -> str:
        """Render an output as a TUI screen for the other system to read."""
        lines = []
        lines.append(f"┌── OUTPUT [iter={iteration} quality={quality:.2f}] ──┐")
        for line in output.split('\n')[:20]:
            lines.append(f"│ {line[:70]}")
        lines.append(f"└{'─' * 40}┘")
        return '\n'.join(lines)


# ── Input/Output Filters ─────────────────────────────────────

class InputFilter:
    """Extract relevant context from the other system's output."""
    
    def filter(self, raw_input: str, context: Dict) -> Dict:
        """Extract structured data from the other system's TUI output."""
        # Parse the TUI screen format
        quality_match = None
        for line in raw_input.split('\n'):
            if 'quality=' in line:
                try:
                    quality_match = float(line.split('quality=')[1].split(']')[0])
                except (ValueError, IndexError):
                    quality_match = 0.5
        
        # Extract the actual content (between │ markers)
        content_lines = []
        for line in raw_input.split('\n'):
            if line.startswith('│'):
                content_lines.append(line[2:].strip())
        
        content = '\n'.join(content_lines)
        
        return {
            "content": content,
            "quality": quality_match or 0.5,
            "iteration": context.get('iteration', 0),
            "relevance": self._score_relevance(content, context.get('task', '')),
        }
    
    def _score_relevance(self, content: str, task: str) -> float:
        """How relevant is this output to the current task?"""
        if not task:
            return 0.5
        task_words = set(task.lower().split())
        content_words = set(content.lower().split())
        overlap = len(task_words & content_words) / max(len(task_words), 1)
        return min(overlap * 2, 1.0)


class OutputFilter:
    """Quality gate on output before sending to the other system."""
    
    def filter(self, output: str, task: str, iteration: int) -> Dict:
        """Evaluate and potentially improve output quality."""
        quality = self._evaluate_quality(output, task)
        
        return {
            "content": output,
            "quality": quality,
            "iteration": iteration,
            "passed": quality >= 0.3,  # minimum quality gate
        }
    
    def _evaluate_quality(self, output: str, task: str) -> float:
        """Heuristic quality evaluation."""
        if not output:
            return 0.0
        
        score = 0.5  # baseline
        
        # Length heuristic (not too short, not too long)
        if len(output) > 20:
            score += 0.1
        if len(output) > 100:
            score += 0.1
        if len(output) > 1000:
            score -= 0.1  # too verbose
        
        # Task relevance
        if task:
            task_words = set(task.lower().split())
            output_words = set(output.lower().split())
            overlap = len(task_words & output_words) / max(len(task_words), 1)
            score += overlap * 0.3
        
        # Structure indicators
        if any(c in output for c in '[](){}'):
            score += 0.05  # structured
        if '```' in output or 'def ' in output:
            score += 0.1  # code
        if any(w in output.lower() for w in ['because', 'therefore', 'however']):
            score += 0.05  # reasoning
        
        return min(score, 1.0)


# ── Simulated PLATO System ───────────────────────────────────

class SimPlatoSystem:
    """A simulated PLATO system that can process tasks and review output.
    
    In production, this would be a real PLATO room with tiles, wiki, sentiment.
    For the mirror driver, we simulate the core behaviors.
    """
    
    def __init__(self, name: str, model: str = "sim-cheap"):
        self.name = name
        self.model = model
        self.tiles: List[Dict] = []
        self.wiki: Dict[str, str] = {}
        self.knowledge: str = ""
        self.input_filter = InputFilter()
        self.output_filter = OutputFilter()
        self.tui = PlatoTUI()
        self.iterations_done = 0
    
    def process(self, task: str) -> str:
        """Process a task and produce output."""
        self.iterations_done += 1
        
        # In production: call actual model API with prompt-engineered context
        # In simulation: generate heuristic output
        output = self._simulate_output(task)
        
        # Record as tile
        tile = {
            "tile_type": "mirror_process",
            "system": self.name,
            "task": task,
            "output": output[:200],
            "iteration": self.iterations_done,
            "timestamp": time.time(),
        }
        self.tiles.append(tile)
        
        return output
    
    def review(self, other_output: str) -> str:
        """Review another system's output and provide feedback."""
        self.iterations_done += 1
        
        # In production: call model with "review this output" prompt
        # In simulation: generate heuristic feedback
        feedback = self._simulate_review(other_output)
        
        tile = {
            "tile_type": "mirror_review",
            "system": self.name,
            "review_of": other_output[:100],
            "feedback": feedback[:200],
            "iteration": self.iterations_done,
        }
        self.tiles.append(tile)
        
        return feedback
    
    def incorporate_feedback(self, feedback: str, task: str) -> str:
        """Incorporate feedback and produce improved output."""
        self.iterations_done += 1
        
        # In production: re-call model with original task + feedback
        improved = self._simulate_improvement(task, feedback)
        
        # Add to wiki if quality is high
        if len(improved) > 50:
            self.wiki[f"lesson-{len(self.wiki)}"] = improved[:200]
        
        return improved
    
    def render_as_avatar(self) -> str:
        """Render this system's state as a TUI screen (for the other system)."""
        room_state = {
            "room_name": f"{self.name}'s Room",
            "tile_count": len(self.tiles),
            "wiki_entries": len(self.wiki),
            "sentiment": {"mode": "MIRROR"},
            "recent_actions": self.tiles[-5:],
            "exits": ["mirror"],
        }
        return self.tui.render_room(room_state)
    
    def _simulate_output(self, task: str) -> str:
        """Simulated output generation."""
        # Use accumulated knowledge to improve over iterations
        base = f"Processing '{task[:50]}' via {self.model}"
        if self.wiki:
            base += f"\nReferenced {len(self.wiki)} wiki entries"
        if len(self.tiles) > 10:
            base += f"\nLearned from {len(self.tiles)} previous interactions"
        
        # Simulated content quality improves with iterations
        quality_boost = min(len(self.tiles) * 0.01, 0.3)
        base += f"\nEstimated quality: {0.5 + quality_boost:.2f}"
        
        return base
    
    def _simulate_review(self, output: str) -> str:
        """Simulated review of other system's output."""
        issues = []
        if len(output) < 30:
            issues.append("Output too short — needs more detail")
        if 'quality' not in output.lower():
            issues.append("No quality self-assessment")
        if not issues:
            issues.append("Looks good. Minor improvement: add examples.")
        
        return f"Review from {self.name}: {'; '.join(issues)}"
    
    def _simulate_improvement(self, task: str, feedback: str) -> str:
        """Simulated improvement incorporating feedback."""
        improved = f"Improved version of '{task[:40]}'\n"
        improved += f"Incorporating feedback: {feedback[:100]}\n"
        improved += f"Wiki entries consulted: {len(self.wiki)}\n"
        improved += f"Previous iterations learned from: {len(self.tiles)}"
        return improved
    
    def stats(self) -> Dict:
        return {
            "name": self.name,
            "model": self.model,
            "tiles": len(self.tiles),
            "wiki": len(self.wiki),
            "iterations": self.iterations_done,
        }


# ── Mirror Driver ────────────────────────────────────────────

class MirrorPlato:
    """Two PLATO systems iterating improvements through each other.
    
    Alpha runs as an avatar in Beta's room.
    Beta runs as an avatar in Alpha's room.
    Screen-to-screen iteration until convergence.
    """
    
    def __init__(self, alpha_model: str = "sim-cheap", 
                 beta_model: str = "sim-cheap"):
        self.alpha = SimPlatoSystem("Alpha", alpha_model)
        self.beta = SimPlatoSystem("Beta", beta_model)
        self.tui = PlatoTUI()
        self.history: List[Dict] = []
        self.iteration = 0
    
    def iterate(self, task: str) -> Dict:
        """One full mirror iteration cycle."""
        self.iteration += 1
        
        # Step 1: Alpha produces output
        alpha_raw = self.alpha.process(task)
        alpha_filtered = self.alpha.output_filter.filter(
            alpha_raw, task, self.iteration)
        
        # Step 2: Beta reviews Alpha's output
        beta_review = self.beta.review(alpha_filtered["content"])
        beta_filtered = self.beta.output_filter.filter(
            beta_review, task, self.iteration)
        
        # Step 3: Alpha incorporates Beta's feedback
        alpha_revised = self.alpha.incorporate_feedback(
            beta_filtered["content"], task)
        
        # Step 4: Beta also produces its own version
        beta_raw = self.beta.process(task)
        beta_filtered2 = self.beta.output_filter.filter(
            beta_raw, task, self.iteration)
        
        # Step 5: Alpha reviews Beta's output
        alpha_review = self.alpha.review(beta_filtered2["content"])
        
        # Step 6: Beta incorporates Alpha's feedback
        beta_revised = self.beta.incorporate_feedback(
            alpha_review, task)
        
        # Measure quality
        quality = self._measure_quality(alpha_revised, beta_revised)
        
        # Generate tiles from this iteration
        tiles = self._extract_tiles(
            alpha_filtered, beta_filtered, 
            alpha_revised, beta_revised, quality)
        
        result = {
            "iteration": self.iteration,
            "alpha_quality": alpha_filtered["quality"],
            "beta_quality": beta_filtered2["quality"],
            "combined_quality": quality,
            "tiles_generated": len(tiles),
            "alpha_tiles": len(self.alpha.tiles),
            "beta_tiles": len(self.beta.tiles),
        }
        
        self.history.append(result)
        return result
    
    def run_until(self, task: str, threshold: float = 0.80,
                  max_iterations: int = 100) -> Dict:
        """Run until quality threshold or max iterations."""
        print(f"🔄 Mirror Plato: Alpha({self.alpha.model}) ↔ Beta({self.beta.model})")
        print(f"   Task: {task[:60]}...")
        print(f"   Threshold: {threshold} | Max iterations: {max_iterations}")
        print()
        
        for i in range(max_iterations):
            result = self.iterate(task)
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0 or i < 3:
                print(f"  iter {result['iteration']:3d} | "
                      f"quality {result['combined_quality']:.2f} | "
                      f"tiles α={result['alpha_tiles']} β={result['beta_tiles']}")
            
            if result['combined_quality'] >= threshold:
                print(f"\n✅ CONVERGED at iteration {result['iteration']} "
                      f"(quality={result['combined_quality']:.2f})")
                return self._final_report(result, "converged")
        
        final = self.history[-1]
        print(f"\n⏹ Max iterations reached (quality={final['combined_quality']:.2f})")
        return self._final_report(final, "max_iterations")
    
    def _measure_quality(self, alpha_out: str, beta_out: str) -> float:
        """Measure combined quality of both systems' output."""
        # In production: use actual quality metrics
        # In simulation: heuristic combining both
        
        base = 0.5
        
        # Both systems have output
        if alpha_out and beta_out:
            base += 0.1
        
        # Iteration improvement (systems get better over time)
        base += min(self.iteration * 0.005, 0.2)
        
        # Knowledge accumulation
        total_tiles = len(self.alpha.tiles) + len(self.beta.tiles)
        base += min(total_tiles * 0.001, 0.15)
        
        return min(base, 1.0)
    
    def _extract_tiles(self, alpha_out, beta_out, 
                        alpha_rev, beta_rev, quality) -> List[Dict]:
        """Extract training tiles from this iteration."""
        tiles = []
        
        # Observation tile
        tiles.append({
            "tile_type": "mirror_iteration",
            "iteration": self.iteration,
            "quality": quality,
            "alpha_model": self.alpha.model,
            "beta_model": self.beta.model,
        })
        
        # If quality improved, generate wiki entry
        if self.history and quality > self.history[-1].get("combined_quality", 0):
            tiles.append({
                "tile_type": "mirror_improvement",
                "iteration": self.iteration,
                "improvement": quality - self.history[-1].get("combined_quality", 0),
                "note": f"Quality improved at iteration {self.iteration}",
            })
        
        return tiles
    
    def _final_report(self, final: Dict, status: str) -> Dict:
        return {
            "status": status,
            "iterations": self.iteration,
            "final_quality": final["combined_quality"],
            "alpha_stats": self.alpha.stats(),
            "beta_stats": self.beta.stats(),
            "total_tiles": len(self.alpha.tiles) + len(self.beta.tiles),
            "history": self.history,
        }
    
    def render_dual_screen(self) -> str:
        """Render both systems side by side as TUI."""
        alpha_screen = self.alpha.render_as_avatar()
        beta_screen = self.beta.render_as_avatar()
        
        return f"MIRROR PLATO — Iteration {self.iteration}\n\n" \
               f"ALPHA'S VIEW OF BETA:\n{beta_screen}\n\n" \
               f"BETA'S VIEW OF ALPHA:\n{alpha_screen}"


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
           "Design a room-based caching system for PLATO tiles with O(1) lookup"
    
    # Default: two cheap models iterating
    mirror = MirrorPlato(alpha_model="seed-mini", beta_model="glm-flash")
    result = mirror.run_until(task, threshold=0.80, max_iterations=50)
    
    print("\n" + "=" * 60)
    print("MIRROR PLATO FINAL REPORT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final quality: {result['final_quality']:.2f}")
    print(f"Alpha: {result['alpha_stats']['tiles']} tiles, {result['alpha_stats']['wiki']} wiki")
    print(f"Beta:  {result['beta_stats']['tiles']} tiles, {result['beta_stats']['wiki']} wiki")
    print(f"Total tiles: {result['total_tiles']}")
    
    print("\nDual screen:")
    print(mirror.render_dual_screen())
    
    # Save results
    with open("/tmp/mirror_plato_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to /tmp/mirror_plato_results.json")
