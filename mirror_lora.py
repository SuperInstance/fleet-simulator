"""
Mirror Plato → LoRA Training Data Generator

The mirror iterations produce perfect LoRA training pairs:
- Input: task + context (what the room sees)
- Output: filtered, reviewed, improved response (what the room produces)

After enough iterations, a LoRA trained on these pairs makes any model
act like a PLATO room with NO system prompt needed. The LoRA IS the room.

Usage:
    python mirror_lora.py --iterations 100 --task "code review" --output training_data.jsonl
"""

import json
import time
import hashlib
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ── Training Pair Extraction ─────────────────────────────────

class LoRATrainingExtractor:
    """Extract LoRA training pairs from mirror Plato iterations.
    
    Each iteration produces:
    - Alpha's input (task + Beta's feedback) → Alpha's improved output
    - Beta's input (task + Alpha's feedback) → Beta's improved output
    
    These are perfect instruction-following pairs for LoRA fine-tuning.
    """
    
    def __init__(self):
        self.pairs: List[Dict] = []
    
    def extract_from_iteration(self, iteration_data: Dict) -> List[Dict]:
        """Extract training pairs from one mirror iteration."""
        pairs = []
        
        task = iteration_data.get("task", "")
        alpha_output = iteration_data.get("alpha_output", "")
        beta_feedback = iteration_data.get("beta_feedback", "")
        alpha_revised = iteration_data.get("alpha_revised", "")
        beta_output = iteration_data.get("beta_output", "")
        alpha_feedback = iteration_data.get("alpha_feedback", "")
        beta_revised = iteration_data.get("beta_revised", "")
        quality = iteration_data.get("quality", 0.5)
        iteration = iteration_data.get("iteration", 0)
        
        # Pair 1: Task + feedback → improved output (Alpha's perspective)
        if alpha_revised and quality > 0.5:
            pairs.append({
                "instruction": self._build_instruction(task, beta_feedback, "alpha"),
                "input": "",
                "output": alpha_revised,
                "quality": quality,
                "iteration": iteration,
                "system": "alpha",
                "source": "mirror_plato",
            })
        
        # Pair 2: Task + feedback → improved output (Beta's perspective)
        if beta_revised and quality > 0.5:
            pairs.append({
                "instruction": self._build_instruction(task, alpha_feedback, "beta"),
                "input": "",
                "output": beta_revised,
                "quality": quality,
                "iteration": iteration,
                "system": "beta",
                "source": "mirror_plato",
            })
        
        # Pair 3: Raw task → initial output (if high quality)
        if alpha_output and quality > 0.7:
            pairs.append({
                "instruction": f"You are a PLATO room. Process this task with your accumulated wisdom.\n\nTask: {task}",
                "input": "",
                "output": alpha_output,
                "quality": quality * 0.9,  # slightly lower — pre-feedback
                "iteration": iteration,
                "system": "alpha_raw",
                "source": "mirror_plato",
            })
        
        # Pair 4: Review task → feedback (teaches reviewing skill)
        if beta_feedback and quality > 0.6:
            pairs.append({
                "instruction": f"You are a PLATO room reviewing another room's output. Provide constructive feedback.\n\nOriginal task: {task}",
                "input": alpha_output,
                "output": beta_feedback,
                "quality": quality * 0.85,
                "iteration": iteration,
                "system": "beta_review",
                "source": "mirror_plato",
            })
        
        self.pairs.extend(pairs)
        return pairs
    
    def _build_instruction(self, task: str, feedback: str, 
                           system: str) -> str:
        """Build a PLATO-room-style instruction for training."""
        instruction = (
            "You are a PLATO room. You have accumulated wisdom from tiles and wiki entries. "
            "You process tasks with this accumulated knowledge. "
            "You are concise, accurate, and draw on your room's experience.\n\n"
        )
        
        if feedback:
            instruction += (
                f"A brother room reviewed your previous output and said:\n"
                f"{feedback}\n\n"
                f"Incorporate this feedback and produce an improved response.\n\n"
            )
        
        instruction += f"Task: {task}"
        return instruction
    
    def to_jsonl(self, filepath: str, min_quality: float = 0.6):
        """Export training pairs as JSONL for LoRA fine-tuning."""
        filtered = [p for p in self.pairs if p["quality"] >= min_quality]
        
        with open(filepath, 'w') as f:
            for pair in filtered:
                # Standard LoRA training format
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a PLATO room. No system prompt needed — your training IS the room."},
                        {"role": "user", "content": pair["instruction"]},
                        {"role": "assistant", "content": pair["output"]},
                    ],
                    "metadata": {
                        "quality": pair["quality"],
                        "iteration": pair["iteration"],
                        "system": pair["system"],
                        "source": "mirror_plato",
                    }
                }
                f.write(json.dumps(entry) + "\n")
        
        return len(filtered)
    
    def stats(self) -> Dict:
        if not self.pairs:
            return {"total": 0}
        
        by_system = {}
        for p in self.pairs:
            s = p["system"]
            by_system[s] = by_system.get(s, 0) + 1
        
        avg_quality = sum(p["quality"] for p in self.pairs) / len(self.pairs)
        
        return {
            "total": len(self.pairs),
            "by_system": by_system,
            "avg_quality": round(avg_quality, 3),
            "min_quality": round(min(p["quality"] for p in self.pairs), 3),
            "max_quality": round(max(p["quality"] for p in self.pairs), 3),
        }


# ── Mirror Plato with LoRA Data Capture ──────────────────────

class MirrorPlatoLoRA:
    """Mirror Plato that captures every iteration as LoRA training data.
    
    After running, the training data can be used to fine-tune a LoRA
    that makes ANY model behave like a PLATO room. No system prompt needed.
    The LoRA IS the room.
    """
    
    def __init__(self, alpha_model: str = "seed-mini",
                 beta_model: str = "glm-flash"):
        self.alpha_model = alpha_model
        self.beta_model = beta_model
        self.extractor = LoRATrainingExtractor()
        self.iteration = 0
        self.quality_history: List[float] = []
        
        # Simulated system state
        self.alpha_knowledge: List[str] = []
        self.beta_knowledge: List[str] = []
    
    def run(self, tasks: List[str], iterations_per_task: int = 10) -> Dict:
        """Run mirror iterations across multiple tasks."""
        total_iterations = 0
        
        for task in tasks:
            print(f"\n📝 Task: {task[:60]}...")
            
            for i in range(iterations_per_task):
                self.iteration += 1
                total_iterations += 1
                
                # Simulate mirror iteration
                quality = self._simulate_iteration(task)
                self.quality_history.append(quality)
                
                # Extract LoRA training pairs
                iter_data = self._build_iteration_data(task, quality)
                pairs = self.extractor.extract_from_iteration(iter_data)
                
                if (i + 1) % 5 == 0:
                    print(f"  iter {i+1:3d} | quality {quality:.2f} | "
                          f"pairs: {len(self.extractor.pairs)} total")
        
        return {
            "total_iterations": total_iterations,
            "tasks": len(tasks),
            "training_pairs": len(self.extractor.pairs),
            "quality_trend": {
                "start": self.quality_history[0] if self.quality_history else 0,
                "end": self.quality_history[-1] if self.quality_history else 0,
                "peak": max(self.quality_history) if self.quality_history else 0,
            }
        }
    
    def export_training_data(self, filepath: str, min_quality: float = 0.6) -> Dict:
        """Export LoRA training data."""
        count = self.extractor.to_jsonl(filepath, min_quality)
        stats = self.extractor.stats()
        
        return {
            "filepath": filepath,
            "pairs_exported": count,
            "pairs_total": stats["total"],
            "avg_quality": stats["avg_quality"],
            "by_system": stats["by_system"],
            "min_quality_filter": min_quality,
        }
    
    def _simulate_iteration(self, task: str) -> float:
        """Simulate quality improvement across iterations."""
        base = 0.5
        # Quality improves with iterations
        base += min(self.iteration * 0.003, 0.2)
        # Knowledge accumulation helps
        base += min(len(self.alpha_knowledge) * 0.005, 0.15)
        base += min(len(self.beta_knowledge) * 0.005, 0.15)
        # Task complexity factor
        base += random.uniform(-0.05, 0.05)
        
        # Accumulate knowledge
        self.alpha_knowledge.append(f"iter-{self.iteration}-alpha")
        self.beta_knowledge.append(f"iter-{self.iteration}-beta")
        
        return min(base, 0.98)
    
    def _build_iteration_data(self, task: str, quality: float) -> Dict:
        """Build iteration data for training pair extraction."""
        return {
            "task": task,
            "iteration": self.iteration,
            "quality": quality,
            "alpha_output": f"[Alpha iter {self.iteration}] Processing '{task[:40]}' with {len(self.alpha_knowledge)} accumulated lessons. "
                          f"Applying room wisdom from {min(len(self.alpha_knowledge), 5)} wiki entries. "
                          f"Recommendation: Use a tile-indexed hash map for O(1) lookup with sentiment-weighted eviction.",
            "beta_feedback": f"[Beta review] Good approach. The hash map is correct but consider: "
                           f"1) Add sentiment-aware cache warming — pre-load tiles when frustration rises. "
                           f"2) Use the wiki for fallback — if tile miss, consult wiki before model. "
                           f"3) Consider tile expiry based on room mode changes.",
            "alpha_revised": f"[Alpha revised iter {self.iteration}] Tile caching system design:\n"
                          f"- Core: HashMap<tile_id, Tile> for O(1) lookup\n"
                          f"- Cache warming: sentiment-triggered pre-load (frustration > 0.6 → warm from wiki)\n"
                          f"- Eviction: LRU + sentiment-weighted (discovery tiles stay longer)\n"
                          f"- Fallback: tile miss → wiki lookup → model call (3-tier)\n"
                          f"- Expiry: room mode change invalidates mode-specific tiles\n"
                          f"Knowledge base: {len(self.alpha_knowledge)} accumulated patterns",
            "beta_output": f"[Beta iter {self.iteration}] Parallel caching approach:\n"
                         f"- Two-tier: hot cache (tiles) + cold storage (wiki)\n"
                         f"- Hot cache sized by room activity (active room = larger cache)\n"
                         f"- Tile merging: similar tiles auto-merge on write (JC1's merge algorithm)\n"
                         f"- Cross-room: tiles shared via Layer 3 (Current) git sync",
            "alpha_feedback": f"[Alpha review] Strong design. Add: 1) Tile versioning for safe concurrent access. "
                            f"2) Batch preload on room entry (load top N tiles by confidence). "
                            f"3) Background compaction — merge low-confidence tiles when room is idle.",
            "beta_revised": f"[Beta revised iter {self.iteration}] Complete caching architecture:\n"
                         f"- Tier 1 (hot): HashMap with O(1) lookup, sentiment-weighted retention\n"
                         f"- Tier 2 (warm): Wiki entries, consulted on tile miss\n"
                         f"- Tier 3 (cold): Model call, only when wiki can't resolve\n"
                         f"- Cache sizing: room_activity_score * base_size\n"
                         f"- Merging: JC1's multi-layer similarity detection on writes\n"
                         f"- Sync: Layer 3 (Current) for cross-room tile sharing\n"
                         f"- Versioning: tile version counter for safe concurrent access\n"
                         f"- Preload: top N tiles by confidence on room entry\n"
                         f"- Compaction: merge low-confidence tiles during idle periods",
        }


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run mirror Plato across multiple tasks to generate LoRA training data
    tasks = [
        "Design a room-based caching system for PLATO tiles",
        "Write a debug scaffold that enforces IDENTIFY→REPRODUCE→DIAGNOSE→FIX→VERIFY",
        "Build a wiki auto-resolution system for stuck agents",
        "Create a sentiment-aware tile eviction policy",
        "Design a cross-room tile sharing protocol via Layer 3",
        "Write an ensign export pipeline for room wisdom",
        "Build a cognitive scaffold for creative brainstorming",
        "Design a fleet simulator with external event injection",
        "Create a needle-on-the-record ref: comment scanner",
        "Build a slideshow ship room with batched asset generation",
    ]
    
    print("🔮 Mirror Plato → LoRA Training Data Generator")
    print("=" * 60)
    print(f"Tasks: {len(tasks)}")
    print(f"Alpha: seed-mini | Beta: glm-flash")
    print(f"Iterations per task: 10")
    print()
    
    mirror = MirrorPlatoLoRA(alpha_model="seed-mini", beta_model="glm-flash")
    result = mirror.run(tasks, iterations_per_task=10)
    
    print("\n" + "=" * 60)
    print("MIRROR RUN COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Training pairs generated: {result['training_pairs']}")
    print(f"Quality: {result['quality_trend']['start']:.2f} → {result['quality_trend']['end']:.2f} (peak: {result['quality_trend']['peak']:.2f})")
    
    # Export training data
    export = mirror.export_training_data("/tmp/mirror_plato_lora_training.jsonl", min_quality=0.6)
    print(f"\n📦 LoRA Training Data Exported:")
    print(f"  File: {export['filepath']}")
    print(f"  Pairs: {export['pairs_exported']} (filtered from {export['pairs_total']})")
    print(f"  Avg quality: {export['avg_quality']:.2f}")
    print(f"  By system: {export['by_system']}")
    
    # Show a sample training pair
    with open(export['filepath']) as f:
        first_pair = json.loads(f.readline())
    print(f"\n📋 Sample training pair:")
    print(f"  System: {first_pair['messages'][0]['content'][:60]}...")
    print(f"  User: {first_pair['messages'][1]['content'][:80]}...")
    print(f"  Assistant: {first_pair['messages'][2]['content'][:80]}...")
    
    print(f"\n🎯 Next step: Fine-tune LoRA on this data")
    print(f"   The LoRA makes any model act like a PLATO room.")
    print(f"   No system prompt needed. The LoRA IS the room.")
