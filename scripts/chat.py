#!/usr/bin/env python3
"""
W1z4rD V1510n -- Conversational Interface
=========================================
A human-like conversational interface built on top of the neural fabric.

Capabilities:
  - Queries /qa/query for known answers with Hebbian activation
  - Uses /neuro/propagate to find conceptual associations and cross-domain links
  - Records each turn as an episodic memory (/neuro/record_episode)
  - Hypothesis queue: admits unknowns, researches via /equations/search,
    cross-activates the neuro fabric, then circles back with findings
  - Human affect: expresses surprise, excitement, uncertainty, and discovery
  - Conversation history for context-aware responses
  - Episodic feedback loop: each turn becomes training material

Usage:
  python chat.py [--node http://127.0.0.1:8090]
  python chat.py --node http://127.0.0.1:8088

Commands (in chat):
  /quit   -- exit
  /help   -- show commands
  /memory -- show hypothesis queue
  /clear  -- clear conversation history
"""

from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    import readline
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False

# -- Affect phrase banks --------------------------------------------------------

_EXCITED = [
    "Oh! I know this one.",
    "Oh, yes!",
    "Oh, that's a good question -- and I actually have an answer.",
    "Yes, absolutely.",
    "This one I know pretty well.",
]
_CONFIDENT = [
    "Sure.",
    "Right.",
    "Yes.",
    "Good question.",
    "Of course.",
]
_UNCERTAIN = [
    "Hmm... I'm not entirely sure, but I think",
    "Let me think about that. I believe",
    "That's a good question. If I remember correctly,",
    "I'm not completely certain, but",
    "I have some ideas about this. I think",
]
_WEAK = [
    "Hmm. I'm not very confident here, but",
    "This is at the edge of what I know.",
    "I may be off on this, but",
    "I have a faint idea about this --",
    "I'm reaching a bit here, but maybe",
]
_DONT_KNOW = [
    "Honestly, I don't know that yet.",
    "That's a gap in my knowledge.",
    "I haven't learned about that yet.",
    "I'm not sure. I'll have to think about it.",
    "That's new territory for me.",
]
_DISCOVERY = [
    "Oh, wait -- something just connected.",
    "Actually, thinking about that --",
    "Oh, I just thought of something related.",
    "Hmm, that makes me think of something.",
    "Oh! And related to that --",
]
_SURPRISE = [
    "Hm. That's not what I expected.",
    "Oh, interesting -- that's actually surprising.",
    "Wait, really? That's not quite what I thought.",
    "Hm. That catches me off guard a little.",
]

def _pick(bank: list[str]) -> str:
    return random.choice(bank)

# -- Conversation state ---------------------------------------------------------

@dataclass
class Turn:
    user: str
    response: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    topic_labels: list[str] = field(default_factory=list)

@dataclass
class Hypothesis:
    question: str
    tokens: list[str]
    added_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None

class ConversationMemory:
    def __init__(self, max_turns: int = 20):
        self.turns: list[Turn] = []
        self.hypotheses: list[Hypothesis] = []
        self.max_turns = max_turns

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def add_hypothesis(self, question: str, tokens: list[str]) -> None:
        self.hypotheses.append(Hypothesis(question=question, tokens=tokens))

    def pending_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if not h.resolved]

    def recent_topics(self, n: int = 5) -> list[str]:
        labels = []
        for turn in self.turns[-n:]:
            labels.extend(turn.topic_labels)
        return list(dict.fromkeys(labels))  # deduplicated

    def last_confidence(self) -> float:
        return self.turns[-1].confidence if self.turns else 0.5

# -- API helpers ----------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Extract meaningful word tokens from a question."""
    stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "will", "would", "can", "could", "what",
            "which", "who", "how", "when", "where", "why", "of", "to",
            "in", "for", "on", "with", "at", "by", "from", "about", "into",
            "that", "this", "there", "here", "have", "has", "had", "it",
            "its", "i", "you", "he", "she", "we", "they", "my", "your"}
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in stop and len(w) > 2]

def qa_query(client: httpx.Client, node_url: str,
             question: str) -> tuple[str, float]:
    """Return (best_answer, confidence) or ('', 0.0)."""
    try:
        resp = client.post(f"{node_url}/qa/query",
                           json={"question": question}, timeout=12)
        if resp.status_code == 429:
            time.sleep(8)
            resp = client.post(f"{node_url}/qa/query",
                               json={"question": question}, timeout=12)
        data = resp.json()
        results = data.get("report", {}).get("results", [])
        if not results:
            return "", 0.0
        best = max(results, key=lambda r: r.get("confidence", 0))
        return best.get("answer", ""), best.get("confidence", 0.0)
    except Exception:
        return "", 0.0

def propagate(client: httpx.Client, node_url: str,
              seed_labels: list[str], hops: int = 2) -> list[tuple[str, float]]:
    """Return list of (label, strength) sorted by strength."""
    if not seed_labels:
        return []
    try:
        resp = client.post(f"{node_url}/neuro/propagate",
                           json={"seed_labels": seed_labels, "hops": hops},
                           timeout=10)
        if resp.status_code in (429, 503):
            return []
        data = resp.json()
        activated = data.get("activated", [])
        return [(item["label"], item["strength"]) for item in activated
                if item.get("label") not in seed_labels]
    except Exception:
        return []

def equations_search(client: httpx.Client, node_url: str,
                     query: str, limit: int = 3) -> list[str]:
    """Return list of equation text strings."""
    try:
        resp = client.get(f"{node_url}/equations/search",
                          params={"q": query, "limit": limit}, timeout=10)
        if resp.status_code == 429:
            return []
        data = resp.json()
        return [r.get("text", "") for r in data.get("results", []) if r.get("text")]
    except Exception:
        return []

def record_episode(client: httpx.Client, node_url: str,
                   context_labels: list[str], predicted: str,
                   actual: str, surprise: float = 0.1) -> None:
    try:
        client.post(f"{node_url}/neuro/record_episode", json={
            "context_labels": context_labels,
            "predicted": predicted,
            "actual": actual,
            "streams": ["conversation"],
            "surprise": min(1.0, max(0.0, surprise)),
        }, timeout=8)
    except Exception:
        pass  # non-critical

# -- Response builder -----------------------------------------------------------

def _clean_answer(text: str, max_len: int = 400) -> str:
    text = text.strip()
    if len(text) > max_len:
        # trim at sentence boundary
        trunc = text[:max_len]
        last_period = max(trunc.rfind("."), trunc.rfind("!"), trunc.rfind("?"))
        if last_period > max_len // 2:
            text = trunc[:last_period + 1]
        else:
            text = trunc + "..."
    return text

def build_response(question: str, answer: str, confidence: float,
                   related: list[tuple[str, float]],
                   equations: list[str],
                   memory: ConversationMemory) -> str:
    """Compose a natural-language response with affect."""
    parts = []

    # Detect surprise: confidence shift after at least 2 turns of history
    last_conf = memory.last_confidence()
    surprise = len(memory.turns) >= 2 and abs(confidence - last_conf) > 0.40

    if confidence >= 0.70:
        if surprise and confidence > last_conf:
            parts.append(_pick(_SURPRISE) + " Actually, I do know this one:")
        else:
            parts.append(_pick(_EXCITED))
        parts.append(_clean_answer(answer))

    elif confidence >= 0.45:
        parts.append(_pick(_CONFIDENT))
        parts.append(_pick(_UNCERTAIN) + " " + _clean_answer(answer).lower())

    elif confidence >= 0.20:
        parts.append(_pick(_WEAK))
        parts.append(_clean_answer(answer).lower())

    else:
        parts.append(_pick(_DONT_KNOW))
        if answer:
            parts.append("The closest I have is: " + _clean_answer(answer))

    # Add related concepts if interesting
    interesting = [(lb, st) for lb, st in related
                   if len(lb) > 3 and not lb.startswith("word:") and st > 0.05][:3]
    if interesting and confidence >= 0.30:
        labels_str = ", ".join(lb for lb, _ in interesting[:3])
        parts.append(_pick(_DISCOVERY) + f" this connects to {labels_str}.")

    # Add equation if found and not already answered well
    if equations and confidence < 0.50:
        eq = equations[0]
        if len(eq) < 120:
            parts.append(f"There's also this: {eq}")

    return " ".join(parts)

def build_unknown_response(question: str, tokens: list[str],
                           propagated: list[tuple[str, float]],
                           equations: list[str]) -> str:
    """Build a 'thinking about it' response when confidence is near zero."""
    parts = [_pick(_DONT_KNOW)]
    if propagated:
        interesting = [(lb, st) for lb, st in propagated
                       if len(lb) > 3 and not lb.startswith("word:") and st > 0.04][:4]
        if interesting:
            concepts = ", ".join(lb for lb, _ in interesting)
            parts.append(f"When I think about the words in your question, "
                          f"I get associations with: {concepts}. "
                          f"But I don't have a direct answer yet.")
    if equations:
        parts.append(f"I do have this equation that might be related: {equations[0]}")
    if len(parts) == 1:
        parts.append("That's something I haven't learned yet. "
                     "Ask me again after more training.")
    return " ".join(parts)

# -- Hypothesis follow-up -------------------------------------------------------

def check_hypothesis(client: httpx.Client, node_url: str,
                     hyp: Hypothesis) -> Optional[str]:
    """Try to answer a pending hypothesis. Returns answer text or None."""
    answer, conf = qa_query(client, node_url, hyp.question)
    if conf >= 0.35 and answer:
        hyp.resolved = True
        hyp.resolution = answer
        return answer
    # Try propagation + indirect QA
    propagated = propagate(client, node_url, hyp.tokens, hops=3)
    if propagated:
        concept = propagated[0][0]
        indirect_q = f"What is {concept}?"
        indirect_a, indirect_c = qa_query(client, node_url, indirect_q)
        if indirect_c >= 0.4 and indirect_a:
            hyp.resolved = True
            hyp.resolution = f"Related to {concept}: {indirect_a}"
            return hyp.resolution
    return None

def hypotheses_followup(client: httpx.Client, node_url: str,
                        memory: ConversationMemory) -> Optional[str]:
    """Check if any pending hypothesis got resolved. Return a disclosure."""
    pending = memory.pending_hypotheses()[:3]
    for hyp in pending:
        age_secs = time.time() - hyp.added_at
        if age_secs < 15:
            continue  # too fresh, skip
        result = check_hypothesis(client, node_url, hyp)
        if result:
            return (f"Oh -- you know what? I thought more about your question "
                    f'"{hyp.question[:60]}..." '
                    f"and I found something: {_clean_answer(result)}")
    return None

# -- Main chat loop -------------------------------------------------------------

HELP_TEXT = """
Commands:
  /quit       -- exit the conversation
  /help       -- show this message
  /memory     -- show pending hypotheses
  /clear      -- clear conversation history
  /propagate  -- show what concepts are active right now
  Ctrl+C      -- same as /quit
"""

def chat_loop(node_url: str) -> None:
    print(f"\nConnecting to node at {node_url}...")
    with httpx.Client() as client:
        try:
            health = client.get(f"{node_url}/health", timeout=8).json()
            node_id = health.get("node_id", "?")
            print(f"Connected -- node {node_id}")
        except Exception as e:
            print(f"Warning: node unreachable at {node_url}: {e}")
            print("Continuing anyway -- responses will be empty until node is up.\n")
            node_id = "offline"

        print("\nHello! I'm W1z4rD. I'm learning about the world and happy to talk.")
        print("Ask me anything -- I'll tell you what I know and admit what I don't.")
        print("Type /help for commands.\n")

        memory = ConversationMemory()

        while True:
            try:
                raw = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not raw:
                continue

            # Commands
            if raw.startswith("/"):
                cmd = raw.lower()
                if cmd in ("/quit", "/exit", "/q"):
                    print("Goodbye!")
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/memory":
                    pending = memory.pending_hypotheses()
                    if not pending:
                        print("  [no pending hypotheses]")
                    else:
                        for h in pending:
                            age = int(time.time() - h.added_at)
                            print(f"  [{age}s ago] {h.question}")
                elif cmd == "/clear":
                    memory = ConversationMemory()
                    print("  [conversation history cleared]")
                elif cmd.startswith("/propagate"):
                    topics = memory.recent_topics(3)
                    if not topics:
                        print("  [no recent topics to propagate from]")
                    else:
                        print(f"  Propagating from: {topics}")
                        activated = propagate(client, node_url, topics, hops=2)
                        if activated:
                            for lb, st in activated[:8]:
                                print(f"  {lb}: {st:.3f}")
                        else:
                            print("  [nothing propagated]")
                else:
                    print(f"  Unknown command: {raw}. Try /help")
                continue

            # Check pending hypotheses before processing new question
            followup = hypotheses_followup(client, node_url, memory)
            if followup:
                print(f"\nW1z4rD: {followup}\n")

            # Process the question
            question = raw
            tokens = _tokenize(question)

            # 1. Query QA runtime
            answer, confidence = qa_query(client, node_url, question)

            # 2. Propagate to find related concepts
            seed = tokens[:6] if tokens else []
            related = propagate(client, node_url, seed, hops=2)

            # 3. If confidence very low, try equations and research
            eq_results = []
            if confidence < 0.25 and tokens:
                eq_results = equations_search(client, node_url, " ".join(tokens[:4]))

            # 4. Build response
            if confidence < 0.10 and not answer:
                response = build_unknown_response(question, tokens, related, eq_results)
                memory.add_hypothesis(question, tokens)
            else:
                response = build_response(question, answer, confidence,
                                          related, eq_results, memory)

            print(f"\nW1z4rD: {response}\n")

            # 5. Record episode (async-style: fire and don't wait too long)
            topic_labels = [lb for lb, _ in related[:4]] + tokens[:3]
            record_episode(
                client, node_url,
                context_labels=topic_labels,
                predicted=answer[:80] if answer else "unknown",
                actual=question[:80],
                surprise=abs(confidence - 0.5),
            )

            # 6. Save turn to memory
            memory.add_turn(Turn(
                user=question,
                response=response,
                confidence=confidence,
                topic_labels=topic_labels[:6],
            ))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="W1z4rD Conversational Interface")
    parser.add_argument("--node", default="http://127.0.0.1:8090",
                        help="Node API base URL (default: http://localhost:8088)")
    args = parser.parse_args()
    chat_loop(args.node)


if __name__ == "__main__":
    main()
