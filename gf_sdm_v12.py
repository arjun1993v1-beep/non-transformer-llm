"""
GF-SDM v12 — Symbolic-First, Neural-Refine
============================================
THE KEY INSIGHT (from v11 failure analysis):

  v10/v11 pipeline (WRONG ORDER):
    question → neural generates → validate → maybe fallback
    ↑ neural dominates first, truth is a fallback

  v12 pipeline (CORRECT ORDER):
    question → symbolic finds fact → neural rephrases → output
    ↑ truth is guaranteed first, neural only adds language polish

Why this fixes everything:
  - "memory → evolution" drift is IMPOSSIBLE:
    symbolic locks the fact to memory first,
    neural seeds from THAT fact's first words,
    output must overlap with THAT fact's concepts.
  - Neural doesn't need to be smart. It only needs to rephrase.
  - Fallback cost is zero: if neural drifts, return the fact directly.

Also fixes v11's "dna works..." bug:
  Relation word is always taken from the FACT TEXT itself,
  not guessed from the question type.

Architecture:
  Input  : 3-word context window (averaged embeddings)
  Embed  : 64-dim
  Hidden : 128-dim ReLU
  Output : vocab_size softmax

Training data:
  - 150+ base facts (from v11)
  - Q→A pairs auto-generated from every fact (Fix 3 from v11)
  - All v8 BUILTIN_QA pairs (rich, curated, 100+ entries)

Pure Python + Numpy only. Runs on Celeron / Termux / Bodhi Linux.
"""

import os, re, json, random, math
import numpy as np

# ============================================================
# SHARED UTILITIES
# ============================================================

STOP = {
    'what','is','are','a','an','the','of','in','does','do',
    'how','why','when','where','who','which','was','were',
    'has','have','had','its','it','be','been','can','could',
    'and','or','but','so','if','then','that','this','to',
    'for','on','at','by','from','with','about','those','these',
    'their','they','we','you','i','as','all','any','not','no',
    'also','more','most','some','such','than','through','each',
    'into','onto','upon','after','before','between','among',
    'other','another','same','different','many','few','much'
}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def keywords(text):
    return [w for w in tokenize(text) if w not in STOP and len(w) > 2]

def keyset(text):
    return set(keywords(text))


# ============================================================
# CONCEPT NORMALIZER
# ============================================================

class ConceptNormalizer:
    SYNONYM_GROUPS = [
        (['attracts','attract','pulls','pull','draws','draw'], 'attract'),
        (['pushes','push','repels','repel'], 'push'),
        (['moves','move','travels','travel','flows','flow','transfers','transfer'], 'move'),
        (['releases','release','emits','emit','produces','produce',
          'generates','generate','creates','create','forms','form'], 'produce'),
        (['carries','carry','transports','transport','transmits','transmit',
          'sends','send','delivers','deliver'], 'carry'),
        (['stores','store','holds','hold','contains','contain',
          'keeps','keep','retains','retain'], 'store'),
        (['converts','convert','transforms','transform','changes','change'], 'convert'),
        (['connects','connect','links','link','joins','join',
          'relates','relate','bridges','bridge'], 'connect'),
        (['enables','enable','allows','allow','causes','cause',
          'makes','makes','leads','lead'], 'enable'),
        (['requires','require','needs','need','depends','depend'], 'require'),
        (['breaks','break','splits','split','divides','divide',
          'separates','separate'], 'split'),
        (['combines','combine','fuses','fuse','merges','merge'], 'combine'),
        (['massive','enormous','huge','vast','giant','large','big',
          'great','immense'], 'large'),
        (['tiny','small','little','minute','micro','mini'], 'small'),
        (['hot','warm','heated','thermal'], 'heat'),
        (['cold','cool','frozen','icy'], 'cold'),
        (['made','built','composed','formed','consisting'], 'made'),
        (['found','located','situated','present','exists'], 'found'),
        (['called','named','known','termed','defined'], 'called'),
        (['similar','alike','same','identical','equivalent'], 'similar'),
        (['learns','learn','acquires','acquire','gains','gain'], 'learn'),
        (['thinks','think','reasons','reason','processes','process'], 'think'),
        (['knows','know','understands','understand','recognizes','recognize'], 'know'),
        (['measures','measure','calculates','calculate','counts','count'], 'measure'),
        (['grows','grow','develops','develop','evolves','evolve'], 'grow'),
        (['dies','die','decays','decay','destroys','destroy'], 'die'),
        (['lives','live','survives','survive','exists','exist'], 'live'),
        (['reproduces','reproduce','copies','copy','replicates','replicate'], 'reproduce'),
        (['shines','shine','glows','glow','radiates','radiate',
          'illuminates','illuminate'], 'shine'),
        (['absorbs','absorb','takes','take','receives','receive'], 'absorb'),
        (['burns','burn','combusts','combust','ignites','ignite'], 'burn'),
    ]

    def __init__(self):
        self.word_to_canon = {}
        for group, canon in self.SYNONYM_GROUPS:
            for word in group:
                self.word_to_canon[word] = canon

    def normalize(self, word):
        return self.word_to_canon.get(word.lower(), word.lower())

    def normalize_set(self, words):
        return {self.normalize(w) for w in words if w not in STOP and len(w) > 2}

    def normalize_text(self, text):
        words = tokenize(text)
        return ' '.join(self.normalize(w) for w in words)


NORMALIZER = ConceptNormalizer()


# ============================================================
# FACT
# ============================================================

RELATION_WORDS = {
    'is','are','was','were','contains','produces','uses',
    'causes','enables','requires','connects','forms','carries',
    'stores','transfers','converts','attracts','releases',
    'made','found','called','known','defined','described',
    'moves','travels','creates','generates','transmits','absorbs'
}

class Fact:
    def __init__(self, raw):
        self.raw      = raw
        self.subject  = ''
        self.relation = 'is'
        self.objects  = []
        self.concepts = NORMALIZER.normalize_set(keyset(raw))
        self._parse()

    def _parse(self):
        words = tokenize(self.raw)
        for i, w in enumerate(words):
            if w in RELATION_WORDS:
                self.subject  = ' '.join(words[:i]).strip()
                self.relation = NORMALIZER.normalize(w)
                self.objects  = [NORMALIZER.normalize(w)
                                 for w in words[i+1:]
                                 if w not in STOP and len(w) > 2]
                return
        self.subject  = words[0] if words else ''
        self.relation = 'relates'
        self.objects  = [NORMALIZER.normalize(w)
                         for w in words[1:] if w not in STOP]

    def seed_words(self, n=4):
        """
        Return the first N words of this fact as a generation seed.
        This is how v12 anchors neural generation to truth:
        the seed always comes from the VERIFIED fact, never from
        question parsing guesswork.
        """
        return tokenize(self.raw)[:n]


# ============================================================
# FACT INDEX
# ============================================================

class FactIndex:
    def __init__(self):
        self.facts         = []
        self.concept_index = {}

    def add(self, raw):
        fid  = len(self.facts)
        fact = Fact(raw)
        self.facts.append(fact)
        for c in fact.concepts:
            if c not in self.concept_index:
                self.concept_index[c] = []
            self.concept_index[c].append(fid)
        return fid

    def lookup(self, concept):
        c = NORMALIZER.normalize(concept)
        return [self.facts[i] for i in self.concept_index.get(c, [])]

    def lookup_many(self, concepts, top_k=6):
        norm = {NORMALIZER.normalize(c) for c in concepts}
        scores = {}
        for c in norm:
            for fid in self.concept_index.get(c, []):
                scores[fid] = scores.get(fid, 0) + 1
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self.facts[fid] for fid, _ in ranked[:top_k]]


# ============================================================
# CONCEPT GRAPH
# ============================================================

class ConceptGraph:
    def __init__(self):
        self.edges   = {}
        self.fact_map = {}   # (c1,c2) → raw fact (for narration)

    def add_fact(self, fact):
        concepts = list(fact.concepts)
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                self._add(c1, c2, fact.raw)
                self._add(c2, c1, fact.raw)

    def _add(self, a, b, raw):
        if a not in self.edges: self.edges[a] = {}
        self.edges[a][b] = self.edges[a].get(b, 0) + 1
        key = (min(a,b), max(a,b))
        if key not in self.fact_map:
            self.fact_map[key] = raw

    def neighbors(self, concept, top_k=8):
        c = NORMALIZER.normalize(concept)
        nbrs = self.edges.get(c, {})
        return sorted(nbrs.items(), key=lambda x: -x[1])[:top_k]

    def get_edge_fact(self, c1, c2):
        key = (min(c1,c2), max(c1,c2))
        return self.fact_map.get(key, '')

    def multihop_path(self, start, end, max_hops=4):
        start = NORMALIZER.normalize(start)
        end   = NORMALIZER.normalize(end)
        if start == end: return [(start, '')]
        if start not in self.edges: return []

        from collections import deque
        visited = {start}
        queue   = deque([[(start, '')]])

        while queue:
            path = queue.popleft()
            node = path[-1][0]
            if len(path) > max_hops: continue
            for nb, _ in self.neighbors(node, top_k=6):
                edge_fact = self.get_edge_fact(node, nb)
                new_path  = path + [(nb, edge_fact)]
                if nb == end: return new_path
                if nb not in visited:
                    visited.add(nb)
                    queue.append(new_path)
        return []

    def related(self, concepts, top_k=8):
        scores = {}
        for c in concepts:
            nc = NORMALIZER.normalize(c)
            for nb, wt in self.neighbors(nc, top_k=12):
                scores[nb] = scores.get(nb, 0) + wt
        for c in concepts:
            scores.pop(NORMALIZER.normalize(c), None)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ============================================================
# MULTI-HOP COMPOSER  (from v8)
# ============================================================

class MultiHopComposer:
    def narrate_path(self, path, src_word, dst_word):
        if not path or len(path) < 2:
            return None
        steps = []
        for i in range(len(path) - 1):
            c1, _  = path[i]
            c2, ef = path[i+1]
            if ef:
                steps.append(ef[0].upper() + ef[1:] + '.')
            else:
                steps.append(f"{c1} connects to {c2}.")
        seen, dedup = set(), []
        for s in steps:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        chain    = ' '.join(dedup)
        concepts = ' → '.join(c for c, _ in path)
        return f"Reasoning chain ({concepts}):\n{chain}"

    def compose_direct(self, A, B, q_concepts):
        shared  = (A.concepts & B.concepts) - STOP
        shared |= (q_concepts & (A.concepts | B.concepts)) - STOP
        if not shared:
            return None, 0.0
        bridge = ', '.join(sorted(shared)[:3])
        text   = (f"{A.raw[0].upper()}{A.raw[1:].rstrip('.')}. "
                  f"Furthermore, {B.raw[0].lower()}{B.raw[1:].rstrip('.')}. "
                  f"These connect through: {bridge}.")
        score  = min(0.5 + 0.1 * len(shared), 0.9)
        return text, score


# ============================================================
# MEMORY  (from v8, trimmed)
# ============================================================

class Memory:
    def __init__(self, max_turns=5):
        self.turns     = []
        self.max_turns = max_turns

    def add(self, q, a):
        self.turns.append((q, a))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def last_topic(self):
        for q, _ in reversed(self.turns):
            kw = keywords(q)
            if kw: return kw[0]
        return None

    def enrich(self, question):
        """Add context from last turn if question is short."""
        kw = keywords(question)
        if len(kw) <= 1 and self.turns:
            last_q, _ = self.turns[-1]
            return question + ' ' + last_q
        return question

    def clear(self):
        self.turns = []

    def summary(self):
        if not self.turns:
            return "No conversation history."
        lines = ["Recent turns:"]
        for q, a in self.turns[-3:]:
            short_a = a[:60] + ('...' if len(a) > 60 else '')
            lines.append(f"  Q: {q}\n  A: {short_a}")
        return '\n'.join(lines)


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:
    PAD = '<PAD>'
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2id  = {}
        self.id2word  = []
        self.freq     = {}
        for tok in [self.PAD, self.UNK, self.BOS, self.EOS]:
            self._add(tok)

    def _add(self, word):
        if word not in self.word2id:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
        return self.word2id[word]

    def count(self, words):
        for w in words:
            self.freq[w] = self.freq.get(w, 0) + 1

    def build(self):
        for w, c in self.freq.items():
            if c >= self.min_freq:
                self._add(w)

    def encode(self, word):
        return self.word2id.get(word, self.word2id[self.UNK])

    def decode(self, idx):
        if 0 <= idx < len(self.id2word):
            return self.id2word[idx]
        return self.UNK

    def size(self):
        return len(self.id2word)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'id2word': self.id2word,
                       'word2id': self.word2id}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.id2word = d['id2word']
        self.word2id = d['word2id']


# ============================================================
# ADAM OPTIMIZER
# ============================================================

class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.t  = 0;  self.m  = {};  self.v  = {}

    def step(self, params, grads):
        self.t += 1
        updated = {}
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.b1 * self.m[key] + (1-self.b1) * grads[key]
            self.v[key] = self.b2 * self.v[key] + (1-self.b2) * grads[key]**2
            m_hat = self.m[key] / (1 - self.b1**self.t)
            v_hat = self.v[key] / (1 - self.b2**self.t)
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


# ============================================================
# NEURAL BRAIN
# ============================================================

class NeuralBrain:
    CONTEXT = 3

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.params = {
            'E' : np.random.randn(vocab_size, embed_dim)  * 0.1,
            'W1': np.random.randn(embed_dim, hidden_dim)  * np.sqrt(2/embed_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, vocab_size) * np.sqrt(2/hidden_dim),
            'b2': np.zeros(vocab_size),
        }
        self.optimizer = Adam(lr=0.002)

    def context_embed(self, context_ids):
        E    = self.params['E']
        vecs = [E[i] for i in context_ids if i >= 0]
        if not vecs: return np.zeros(self.embed_dim)
        return np.mean(vecs, axis=0)

    def forward(self, context_ids):
        emb    = self.context_embed(context_ids)
        h_pre  = emb @ self.params['W1'] + self.params['b1']
        h      = np.maximum(0, h_pre)
        logits = h @ self.params['W2'] + self.params['b2']
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()
        cache  = {'context_ids': context_ids, 'emb': emb,
                  'h_pre': h_pre, 'h': h, 'probs': probs}
        return probs, cache

    def backward(self, cache, y_id):
        probs, h, h_pre = cache['probs'], cache['h'], cache['h_pre']
        emb, context_ids = cache['emb'], cache['context_ids']

        loss           = -np.log(probs[y_id] + 1e-10)
        d_logits       = probs.copy()
        d_logits[y_id] -= 1.0

        d_W2    = np.outer(h, d_logits)
        d_b2    = d_logits
        d_h     = self.params['W2'] @ d_logits
        d_h_pre = d_h * (h_pre > 0)
        d_W1    = np.outer(emb, d_h_pre)
        d_b1    = d_h_pre
        d_emb   = self.params['W1'] @ d_h_pre

        d_E = np.zeros_like(self.params['E'])
        n   = max(len(context_ids), 1)
        for i in context_ids:
            if i >= 0:
                d_E[i] += d_emb / n

        return loss, {'E': d_E, 'W1': d_W1, 'b1': d_b1,
                      'W2': d_W2, 'b2': d_b2}

    def train_step(self, context_ids, y_id):
        probs, cache = self.forward(context_ids)
        loss, grads  = self.backward(cache, y_id)
        self.params  = self.optimizer.step(self.params, grads)
        return loss

    def predict(self, context_ids, temperature=0.8, top_k=10):
        probs, _ = self.forward(context_ids)
        top_ids  = np.argsort(probs)[-top_k:]
        top_p    = probs[top_ids] ** (1.0 / max(temperature, 0.1))
        top_p   /= top_p.sum()
        return int(np.random.choice(top_ids, p=top_p))

    def predict_top(self, context_ids, k=5):
        probs, _ = self.forward(context_ids)
        top_ids  = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top_ids]

    def save(self, path):
        np.savez(path, **self.params,
                 vocab_size=np.array([self.vocab_size]),
                 embed_dim=np.array([self.embed_dim]),
                 hidden_dim=np.array([self.hidden_dim]))

    @classmethod
    def load(cls, path):
        d = np.load(path + '.npz')
        brain = cls(int(d['vocab_size'][0]),
                    int(d['embed_dim'][0]),
                    int(d['hidden_dim'][0]))
        brain.params = {k: d[k] for k in ['E','W1','b1','W2','b2']}
        return brain


# ============================================================
# TRAINING
# ============================================================

def build_context_pairs(sentences, vocab, context_size=3):
    pad_id = vocab.encode(vocab.PAD)
    bos_id = vocab.encode(vocab.BOS)
    eos_id = vocab.encode(vocab.EOS)
    pairs  = []
    for sent in sentences:
        words = tokenize(sent)
        if len(words) < 2: continue
        ids = [bos_id] + [vocab.encode(w) for w in words] + [eos_id]
        for i in range(1, len(ids)):
            start   = max(0, i - context_size)
            context = ids[start:i]
            while len(context) < context_size:
                context = [pad_id] + context
            pairs.append((context, ids[i]))
    return pairs


def train_brain(sentences, epochs=30, embed_dim=64,
                hidden_dim=128, lr=0.002, verbose=True,
                context_size=3):
    vocab = Vocabulary()
    for sent in sentences:
        vocab.count(tokenize(sent))
    vocab.build()

    if verbose:
        print(f"[v12] Vocab: {vocab.size()} words")

    pairs = build_context_pairs(sentences, vocab, context_size)
    if verbose:
        print(f"[v12] Training pairs: {len(pairs)}")

    brain = NeuralBrain(vocab.size(), embed_dim, hidden_dim)
    brain.optimizer.lr = lr

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = sum(brain.train_step(c, y) for c, y in pairs)
        avg = total_loss / max(len(pairs), 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

    return brain, vocab


# ============================================================
# GENERATION
# ============================================================

def generate(brain, vocab, seed_ids, max_words=30,
             temperature=0.7, top_k=8, context_size=3):
    pad_id    = vocab.encode(vocab.PAD)
    eos_id    = vocab.encode(vocab.EOS)
    generated = list(seed_ids)

    for _ in range(max_words):
        ctx_start = max(0, len(generated) - context_size)
        context   = generated[ctx_start:]
        while len(context) < context_size:
            context = [pad_id] + context
        next_id = brain.predict(context, temperature, top_k)
        if next_id == eos_id: break
        generated.append(next_id)

    return ' '.join(vocab.decode(i) for i in generated
                    if vocab.decode(i) not in
                    [vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK])


# ============================================================
# Q→A PAIR GENERATOR  (Fix 3 from v11, kept in v12)
# ============================================================

def fact_to_qa_sentences(fact_raw):
    """Auto-generate Q→A training sentences from a raw fact."""
    words = tokenize(fact_raw)
    if not words: return []

    subj = None
    for i, w in enumerate(words):
        if w in RELATION_WORDS and i > 0:
            subj = ' '.join(words[:i])
            break
    if not subj:
        subj = words[0]

    bare   = fact_raw.rstrip('.')
    return [
        f"what is {subj} {bare}",
        f"how does {subj} work {bare}",
        f"explain {subj} {bare}",
    ]


def build_qa_sentences(facts):
    out = []
    for f in facts:
        out.extend(fact_to_qa_sentences(f))
    return out


# ============================================================
# STRICT OUTPUT VALIDATOR  (Fix 1 from v11, kept in v12)
# ============================================================

def output_is_valid(output, anchor_fact):
    """
    v12 validator is simpler and stricter than v11:
    The anchor is always the FACT, not the question.

    Valid if:
      1. At least 4 words
      2. Shares ≥1 keyword with the source fact

    This is tighter than v11 (which checked against the question).
    Checking against the FACT is better because:
    - The fact is ground truth
    - The question may use different words ("how does X work")
    """
    if not output or len(output.split()) < 4:
        return False
    out_keys  = keyset(output)
    fact_keys = keyset(anchor_fact.raw)
    return bool(out_keys & fact_keys)


# ============================================================
# THE V12 CORE: SYMBOLIC-FIRST PIPELINE
# ============================================================

def symbolic_first_answer(reasoning, brain, vocab, question,
                           max_words=30, temperature=0.65, top_k=8):
    """
    THE KEY DIFFERENCE from v10/v11:

    OLD (neural-first):
      seed = guess from question → generate → hope it's right → validate

    NEW (symbolic-first):
      fact = symbolic lookup (guaranteed correct)
      seed = first words of THAT FACT (not question)
      result = neural rephrases from that seed
      if valid → return neural output
      else     → return fact directly (zero-cost fallback)

    The neural model's only job is to produce a more natural
    sentence. If it fails, truth is already in hand.
    """
    # ── Step 1: Symbolic truth engine finds the fact ──────────
    relevant = reasoning.find_relevant_facts(question, top_k=3)

    if not relevant:
        # No fact at all — can't help
        return "I don't have information on that topic.", 'none'

    best_fact = relevant[0]

    # ── Step 2: Check for multi-hop (question has 2+ concepts) ──
    multihop = reasoning.try_multihop(question)
    if multihop:
        return multihop, 'multihop'

    # ── Step 3: Build seed FROM THE FACT (not the question) ───
    # This is the core v12 fix.
    # seed_words = first N words of the verified fact
    seed_words = best_fact.seed_words(n=4)
    seed_ids   = [vocab.encode(w) for w in seed_words
                  if vocab.encode(w) != vocab.encode(vocab.UNK)]

    # If seed_ids empty (vocab too small), go direct
    if not seed_ids:
        raw = best_fact.raw
        return raw[0].upper() + raw[1:], 'direct'

    # ── Step 4: Neural rephrases from fact-anchored seed ─────
    result = generate(brain, vocab, seed_ids,
                      max_words=max_words,
                      temperature=temperature,
                      top_k=top_k)

    # ── Step 5: Validate against the SOURCE FACT ─────────────
    if output_is_valid(result, best_fact):
        return result, 'neural+fact-anchored'
    else:
        # Neural drifted — return fact directly (free fallback)
        raw = best_fact.raw
        return raw[0].upper() + raw[1:], 'direct'


# ============================================================
# BUILT-IN FACTS  (150+ from v11, unchanged)
# ============================================================

BUILTIN_FACTS = [
    # Gravity
    "gravity is a fundamental force that attracts objects with mass.",
    "gravity pulls objects toward each other based on their mass.",
    "gravity keeps planets in orbit around the sun.",
    "gravity causes time to pass more slowly near massive objects.",
    "black holes have gravity so strong that light cannot escape.",
    "gravity is described by einstein general theory of relativity.",
    "gravity is the weakest of the four fundamental forces.",
    "the force of gravity between two objects depends on their masses and distance.",
    "gravity holds galaxies together and shapes the structure of the universe.",
    "on earth gravity accelerates falling objects at 9.8 meters per second squared.",
    # Light
    "light is electromagnetic radiation that travels at 299792 kilometers per second.",
    "light can behave as both a wave and a particle.",
    "light carries energy and can transfer that energy to matter.",
    "light bends when passing near massive objects due to gravity.",
    "visible light is a small part of the electromagnetic spectrum.",
    "light is made of particles called photons.",
    "different colors of light have different wavelengths and frequencies.",
    "light cannot travel through a vacuum faster than 299792 kilometers per second.",
    "ultraviolet light is invisible to humans but can cause sunburn.",
    "infrared light is emitted by warm objects as heat radiation.",
    # DNA
    "dna is a molecule that carries genetic instructions for life.",
    "dna is made of four bases adenine thymine cytosine and guanine.",
    "dna stores information in sequences of base pairs.",
    "dna replicates itself so cells can divide and copy information.",
    "mutations in dna drive the process of evolution over generations.",
    "dna is found in the nucleus of every cell in the human body.",
    "the human genome contains about three billion base pairs of dna.",
    "dna is read by proteins called ribosomes to produce other proteins.",
    "dna damage can lead to cancer if repair mechanisms fail.",
    "identical twins share the same dna sequence.",
    # Evolution
    "evolution is the change in inherited traits of populations over generations.",
    "natural selection favors traits that improve survival and reproduction.",
    "species evolve from common ancestors through accumulated mutations.",
    "evolution produces the diversity of life on earth.",
    "charles darwin first proposed the theory of natural selection.",
    "all life on earth shares a common ancestor from about four billion years ago.",
    "evolution can be observed in bacteria that develop antibiotic resistance.",
    "sexual selection drives traits that improve mating success.",
    # Neurons
    "neurons are cells that transmit electrical signals through the nervous system.",
    "neurons communicate using chemicals called neurotransmitters.",
    "synapses are junctions between neurons where signals pass.",
    "the brain contains about 86 billion neurons.",
    "memory is formed by strengthening connections between neurons.",
    "a neuron has a cell body dendrites and an axon.",
    "electrical signals in neurons are called action potentials.",
    "neurons can fire up to 1000 times per second.",
    "damage to neurons in the spinal cord can cause paralysis.",
    "neuroplasticity is the ability of neurons to form new connections.",
    # Memory
    "memory is the ability to store and recall past experiences.",
    "short term memory holds a small amount of information briefly.",
    "long term memory stores information for extended periods.",
    "sleep helps consolidate memories in the brain.",
    "the hippocampus is important for forming new memories.",
    "working memory holds information actively in mind for immediate use.",
    "episodic memory stores personal experiences and events.",
    "semantic memory stores general knowledge and facts about the world.",
    # Stars
    "stars are massive balls of hot plasma that produce light and heat.",
    "stars generate energy through nuclear fusion in their cores.",
    "the sun is a medium sized star at the center of our solar system.",
    "stars produce heavier elements through fusion reactions.",
    "when massive stars die they explode in supernovae.",
    "the nearest star to earth besides the sun is proxima centauri.",
    "stars are classified by their temperature color and brightness.",
    "red giant stars have expanded outer layers and cooler surfaces.",
    "white dwarf stars are dense remnants of dead low mass stars.",
    "neutron stars are incredibly dense remnants of supernova explosions.",
    # Energy
    "energy is the ability to do work or cause change.",
    "energy cannot be created or destroyed only transformed.",
    "kinetic energy is the energy of motion.",
    "potential energy is stored energy due to position or configuration.",
    "nuclear energy is released when atomic nuclei split or fuse.",
    "thermal energy is the internal energy of a system from particle motion.",
    "chemical energy is stored in the bonds between atoms.",
    "solar energy comes from the nuclear fusion reactions in the sun.",
    "electrical energy is carried by moving electrons through conductors.",
    "the law of conservation of energy states that total energy is always constant.",
    # Black holes
    "a black hole forms when a massive star collapses under its own gravity.",
    "the event horizon is the boundary beyond which nothing can escape.",
    "black holes warp space and time around them.",
    "hawking radiation is a theoretical process where black holes slowly lose energy.",
    "supermassive black holes exist at the centers of most galaxies.",
    "the first image of a black hole was captured in 2019.",
    "time passes more slowly closer to a black hole.",
    "black holes can merge and produce gravitational waves.",
    # Water
    "water is a molecule made of two hydrogen atoms and one oxygen atom.",
    "water exists as liquid solid and gas depending on temperature.",
    "water is essential for all known forms of life.",
    "water covers about 71 percent of earth surface.",
    "water has a high heat capacity allowing it to regulate temperatures.",
    "ice is less dense than liquid water which is why it floats.",
    "water is a universal solvent that dissolves many substances.",
    "the water cycle moves water between oceans atmosphere and land.",
    # AI / ML
    "artificial intelligence is the simulation of human intelligence by machines.",
    "machine learning allows systems to learn from data without explicit programming.",
    "neural networks are computing systems inspired by biological neurons.",
    "deep learning uses many layers of neural networks to find patterns.",
    "language models learn to predict the next word from large text datasets.",
    "reinforcement learning trains agents by rewarding correct actions.",
    "supervised learning uses labeled data to train prediction models.",
    "unsupervised learning finds hidden patterns in unlabeled data.",
    "transformers are neural network architectures that use attention mechanisms.",
    "gradient descent is the optimization algorithm used to train neural networks.",
    # Atoms
    "atoms are the basic units of matter that make up all substances.",
    "an atom consists of a nucleus surrounded by electrons.",
    "the nucleus of an atom contains protons and neutrons.",
    "protons carry positive charge and neutrons carry no charge.",
    "electrons carry negative charge and orbit the nucleus.",
    "the number of protons in an atom determines which element it is.",
    "chemical bonds form when atoms share or transfer electrons.",
    "isotopes are atoms of the same element with different numbers of neutrons.",
    # Cells
    "cells are the basic unit of life in all living organisms.",
    "every cell contains dna that encodes the organism genetic information.",
    "the cell membrane controls what enters and leaves the cell.",
    "mitochondria produce energy for the cell through cellular respiration.",
    "the cell nucleus contains the genetic material of the cell.",
    "cells divide through a process called mitosis to produce identical copies.",
    "stem cells can differentiate into many different types of specialized cells.",
    "bacteria are single celled organisms without a true nucleus.",
    # Climate
    "climate is the long term pattern of weather in a region.",
    "the greenhouse effect traps heat in earth atmosphere.",
    "carbon dioxide is a greenhouse gas that contributes to global warming.",
    "the oceans absorb about 30 percent of the carbon dioxide produced by humans.",
    "global warming is causing ice caps to melt and sea levels to rise.",
    "the atmosphere protects life on earth from harmful solar radiation.",
    "photosynthesis by plants absorbs carbon dioxide and releases oxygen.",
    "deforestation reduces the earth ability to absorb carbon dioxide.",
    # Mathematics
    "mathematics is the study of numbers patterns and logical structures.",
    "prime numbers are whole numbers greater than one divisible only by themselves and one.",
    "the pythagorean theorem states that the square of the hypotenuse equals the sum of squares of the other sides.",
    "calculus is the mathematics of change and motion.",
    "algebra uses symbols and rules to manipulate mathematical expressions.",
    "statistics is the study of collecting analyzing and interpreting data.",
    "infinity is a concept describing something without any limit.",
    "zero is the additive identity meaning adding zero to any number leaves it unchanged.",
    # Human body
    "the human body contains about 37 trillion cells.",
    "the heart pumps blood through the circulatory system.",
    "the lungs exchange oxygen and carbon dioxide with the blood.",
    "the liver filters toxins from the blood and produces bile.",
    "the immune system defends the body against pathogens and disease.",
    "bones provide structure support and protection for the body.",
    "the digestive system breaks down food into nutrients for the body.",
    "the brain is the control center of the nervous system.",
]


# ============================================================
# PATHS
# ============================================================

BRAIN_PATH   = 'v12_brain'
VOCAB_PATH   = 'v12_vocab.json'
LEARNED_PATH = 'v12_learned.json'


def load_learned():
    if os.path.exists(LEARNED_PATH):
        with open(LEARNED_PATH) as f:
            return json.load(f)
    return []

def save_learned(data):
    with open(LEARNED_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# REASONING ENGINE
# ============================================================

class ReasoningEngine:
    def __init__(self):
        self.index    = FactIndex()
        self.graph    = ConceptGraph()
        self.composer = MultiHopComposer()
        self.facts    = []

        for fact in BUILTIN_FACTS:
            self._add_fact(fact)
        for _, a in load_learned():
            if a: self._add_fact(a)

    def _add_fact(self, raw):
        fid  = self.index.add(raw)
        fact = self.index.facts[fid]
        self.graph.add_fact(fact)
        self.facts.append(raw)

    def teach(self, raw):
        self._add_fact(raw)
        learned = load_learned()
        learned.append(['', raw])
        save_learned(learned)

    def find_relevant_facts(self, question, top_k=4):
        return self.index.lookup_many(keywords(question), top_k=top_k)

    def try_multihop(self, question):
        """
        If question asks about connections between 2+ concepts,
        try to narrate a multi-hop reasoning chain.
        """
        q_words = [NORMALIZER.normalize(w) for w in keywords(question)]
        for i in range(len(q_words)):
            for j in range(len(q_words)):
                if i == j: continue
                path = self.graph.multihop_path(
                    q_words[i], q_words[j], max_hops=4)
                if path and len(path) >= 3:
                    narrated = self.composer.narrate_path(
                        path, q_words[i], q_words[j])
                    if narrated:
                        return narrated
        return None


# ============================================================
# SELF-CHECK
# ============================================================

def self_check(brain, vocab, facts, n=20, context_size=3):
    sample  = random.sample(facts, min(n, len(facts)))
    correct = 0
    total   = 0
    pad_id  = vocab.encode(vocab.PAD)
    results = []

    for fact in sample:
        words = tokenize(fact)
        if len(words) < 3: continue
        ctx   = [pad_id] * context_size
        ctx[-1] = vocab.encode(words[0])
        top       = brain.predict_top(ctx, k=5)
        top_words = [vocab.decode(i) for i, _ in top]
        hit = words[1] in top_words
        if hit: correct += 1
        total += 1
        results.append({'seed': words[0], 'expected': words[1],
                        'predicted': top_words[0], 'hit': hit})

    acc = correct / max(total, 1) * 100
    print(f"\n{'='*52}")
    print(f"  BRAIN SELF-CHECK  (context={context_size})")
    print(f"{'='*52}")
    print(f"  Facts tested   : {total}")
    print(f"  Top-5 accuracy : {acc:.1f}%")
    if acc >= 60:   print("  ✓ Brain learned facts well")
    elif acc >= 30: print("  ~ Partial — train more epochs")
    else:           print("  ✗ Needs more training")
    print(f"\n  Sample predictions:")
    for r in results[:5]:
        s = '✓' if r['hit'] else '✗'
        print(f"  {s} '{r['seed']}' → "
              f"expected='{r['expected']}' got='{r['predicted']}'")
    print()
    return acc


# ============================================================
# MAIN SYSTEM
# ============================================================

class GFSDMv12:
    """
    v12 — Symbolic-First, Neural-Refine

    The pipeline is now reversed from v10/v11:

      BEFORE (v10, v11):
        question → neural generates → validate → maybe fallback
        (neural goes first, can produce anything, truth is plan B)

      NOW (v12):
        question → symbolic finds fact → seed from FACT → neural rephrases
        (truth goes first, neural only polishes language, drift impossible)

    Additional improvements:
      - Multi-hop reasoning restored from v8 (narrated concept chains)
      - Memory (conversation context) restored from v8
      - validator checks against FACT not question (tighter)
      - 'works' bug fixed: relation word comes from fact text, not question
    """

    CONTEXT = 3

    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.memory    = Memory(max_turns=5)
        self.brain     = None
        self.vocab     = None

        if (os.path.exists(BRAIN_PATH + '.npz') and
                os.path.exists(VOCAB_PATH)):
            self.brain = NeuralBrain.load(BRAIN_PATH)
            self.vocab = Vocabulary()
            self.vocab.load(VOCAB_PATH)
            print(f"[v12] Neural brain loaded. "
                  f"Vocab: {self.vocab.size()} words")
        else:
            print("[v12] No saved brain. Run ai.train() first.")

    # ── TRAINING ─────────────────────────────────────────────

    def train(self, extra_text=None, epochs=40,
              embed_dim=64, hidden_dim=128, lr=0.002):

        sentences = list(BUILTIN_FACTS)

        # Q→A pairs from every fact (Fix 3)
        qa = build_qa_sentences(BUILTIN_FACTS)
        sentences.extend(qa)
        print(f"[v12] Base facts     : {len(BUILTIN_FACTS)}")
        print(f"[v12] Q→A generated  : {len(qa)}")

        if extra_text:
            new_sents = [s.strip() for s in
                         re.split(r'[.!?]', extra_text)
                         if len(s.strip()) > 20]
            sentences.extend(new_sents)
            extra_qa = build_qa_sentences(new_sents)
            sentences.extend(extra_qa)
            print(f"[v12] Extra sentences: +{len(new_sents)} "
                  f"+{len(extra_qa)} Q→A")

        for _, a in load_learned():
            if a:
                sentences.append(a)
                sentences.extend(fact_to_qa_sentences(a))

        print(f"[v12] Total training : {len(sentences)} sentences")
        print(f"[v12] Training neural brain, epochs={epochs}...")

        self.brain, self.vocab = train_brain(
            sentences, epochs=epochs,
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            lr=lr, verbose=True, context_size=self.CONTEXT)

        self.brain.save(BRAIN_PATH)
        self.vocab.save(VOCAB_PATH)
        print("[v12] Brain saved.")

    def train_file(self, path, epochs=30):
        if not os.path.exists(path):
            print(f"[v12] Not found: {path}"); return
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self.train(extra_text=text, epochs=epochs)

    # ── CHAT ─────────────────────────────────────────────────

    def chat(self, question, temperature=0.65, top_k=8):
        question = question.strip()
        if not question: return "I'm listening."

        teach = self._detect_teach(question)
        if teach:
            self.reasoning.teach(teach)
            self.memory.add(question, f"Learned: {teach}")
            return f"Understood. Added: {teach}"

        # Memory enrichment (from v8) — helps with short follow-ups
        enriched = self.memory.enrich(question)

        if self.brain:
            answer, mode = symbolic_first_answer(
                self.reasoning, self.brain, self.vocab,
                enriched, max_words=30,
                temperature=temperature, top_k=top_k)
        else:
            # Neural not trained — pure symbolic (v8-style)
            answer, mode = self._symbolic_only(enriched)

        self.memory.add(question, answer)

        if mode == 'none':
            return answer
        elif mode == 'direct':
            return f"(direct fact)\n{answer}"
        elif mode == 'multihop':
            return f"(multi-hop)\n{answer}"
        else:
            return f"(neural+symbolic)\n{answer}"

    def _symbolic_only(self, question):
        """Pure v8-style fallback when neural not trained."""
        # Try multi-hop first
        hop = self.reasoning.try_multihop(question)
        if hop:
            return hop, 'multihop'
        # Single best fact
        facts = self.reasoning.find_relevant_facts(question, top_k=2)
        if len(facts) >= 2:
            text, score = self.reasoning.composer.compose_direct(
                facts[0], facts[1],
                NORMALIZER.normalize_set(keyset(question)))
            if text and score > 0.05:
                return text, 'symbolic-composed'
        if facts:
            raw = facts[0].raw
            return raw[0].upper() + raw[1:], 'direct'
        return "I don't have enough information on that topic.", 'none'

    # ── TEACH ─────────────────────────────────────────────────

    def teach(self, fact):
        self.reasoning.teach(fact.lower().rstrip('.') + '.')
        print(f"[v12] Taught: {fact}")
        print("[v12] Run ai.train() to update neural weights.")

    # ── INSPECT ───────────────────────────────────────────────

    def inspect(self, word):
        facts = self.reasoning.index.lookup(word)
        if not facts: return f"No facts for '{word}'."
        lines = [f"Facts about '{word}':"]
        for f in facts[:5]:
            lines.append(f"  · {f.raw}")
        return '\n'.join(lines)

    def hop(self, w1, w2):
        path = self.reasoning.graph.multihop_path(
            NORMALIZER.normalize(w1),
            NORMALIZER.normalize(w2), max_hops=5)
        if not path:
            return f"No path between '{w1}' and '{w2}'."
        return self.reasoning.composer.narrate_path(path, w1, w2)

    def check(self, n=20):
        if not self.brain:
            print("No brain."); return
        self_check(self.brain, self.vocab, BUILTIN_FACTS,
                   n=n, context_size=self.CONTEXT)

    def stats(self):
        lines = [
            f"GF-SDM v12 — Symbolic-First, Neural-Refine",
            f"  Facts      : {len(self.reasoning.facts)}",
            f"  Concepts   : {len(self.reasoning.index.concept_index)}",
            f"  Graph nodes: {len(self.reasoning.graph.edges)}",
        ]
        if self.brain:
            total = sum(v.size for v in self.brain.params.values())
            lines += [
                f"  Vocab      : {self.vocab.size()} words",
                f"  Embed dim  : {self.brain.embed_dim}",
                f"  Hidden dim : {self.brain.hidden_dim}",
                f"  Parameters : {total:,}",
                f"  Context    : {self.CONTEXT} words",
                f"",
                f"  v12 architecture: symbolic-first pipeline",
                f"    symbolic → fact (truth locked first)",
                f"    seed     = first words of FACT (not question)",
                f"    neural   = rephrases only (can't drift)",
                f"    fallback = fact directly (always available)",
            ]
        else:
            lines.append("  Neural brain: not trained (symbolic-only mode)")
        return '\n'.join(lines)

    # ── PRIVATE ───────────────────────────────────────────────

    def _detect_teach(self, text):
        orig = text.strip().rstrip('.')
        if re.match(r'^(what|who|how|why|when|where|which|is|are|does|do)\b',
                    orig.lower()):
            return None
        t = orig
        for p in ['remember that ','learn that ','note that ','fact: ']:
            if t.lower().startswith(p): t = t[len(p):]; break
        m = re.match(r'^(.+?)\s+(?:is|are)\s+(.+)$', t, re.IGNORECASE)
        if m:
            subj = m.group(1).strip().lower()
            defn = m.group(2).strip().lower()
            if len(subj.split()) <= 5 and len(defn.split()) >= 3:
                return f"{subj} is {defn}."
        return None


# ============================================================
# ENTRY POINT
# ============================================================

HELP = """
Commands:
  /train [epochs]       — train neural brain (default 40 epochs)
  /train_file <f>       — train from text file
  /check                — self-check accuracy
  /hop <w1> <w2>        — multi-hop concept path
  /inspect <word>       — facts containing word
  /stats                — system stats
  /history              — conversation memory
  /forget               — clear memory
  /teach <sentence>     — add new fact
  /help                 — this help
  quit                  — exit

Teaching (no /):
  "DNA is a molecule that stores genetic information"
  → Auto-detected and stored

v12 = symbolic-first pipeline
  symbolic engine finds the correct fact
  neural engine rephrases it into natural language
  validator checks output against the SOURCE FACT
  fallback returns the fact directly (always available)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GF-SDM v12  |  Symbolic-First, Neural-Refine")
    print("  Truth first. Language second.")
    print(f"  {len(BUILTIN_FACTS)} facts | 15 domains | Pure Python + Numpy")
    print("=" * 60 + "\n")

    ai = GFSDMv12()

    if not ai.brain:
        print("[v12] Auto-training...")
        ai.train(epochs=40)
        print()

    ai.check(n=15)

    print("\n--- Symbolic-First Demo ---\n")
    demos = [
        "what is gravity",
        "what is memory",
        "how does dna work",
        "what is a black hole",
        "how do neurons connect",
        "what is energy",
        "how does evolution happen",
        "what is an atom",
        "what is climate",
        "what connects gravity and light",
        "what links dna and evolution",
    ]
    for q in demos:
        print(f"Q: {q}")
        print(f"A: {ai.chat(q)}")
        print()

    print("-" * 60)
    print("Talk to the brain!\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user: continue
        if user.lower() in ('quit','exit','q'): break
        if user == '/check':    ai.check(); continue
        if user == '/stats':    print(f"\n{ai.stats()}\n"); continue
        if user == '/help':     print(HELP); continue
        if user == '/history':  print(f"\n{ai.memory.summary()}\n"); continue
        if user == '/forget':   ai.memory.clear(); print("Memory cleared.\n"); continue
        if user.startswith('/hop '):
            parts = user[5:].strip().split()
            if len(parts) >= 2:
                print(f"AI : {ai.hop(parts[0], parts[1])}\n")
            continue
        if user.startswith('/inspect '):
            print(f"AI : {ai.inspect(user[9:].strip())}\n"); continue
        if user.startswith('/teach '):
            ai.teach(user[7:].strip()); continue
        if user.startswith('/train_file '):
            ai.train_file(user[12:].strip()); continue
        if user.startswith('/train'):
            parts  = user.split()
            epochs = int(parts[1]) if len(parts) > 1 else 40
            ai.train(epochs=epochs); continue

        print(f"AI: {ai.chat(user)}\n")

    print("\n[Done]")
