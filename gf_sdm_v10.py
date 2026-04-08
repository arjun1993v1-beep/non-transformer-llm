"""
GF-SDM v10 — Hybrid Brain
==========================
v8 (symbolic reasoning) + v9 (neural generator) combined.

How it works:
  1. Question comes in
  2. v8 brain thinks → finds concept chain + relevant facts
  3. Concept chain becomes the seed for v9 generator
  4. v9 turns that seed into natural language output

Also fixes v9's single-word context problem:
  → v10 uses last 3 words averaged as context embedding

Architecture:
  Input  : 3-word context window (averaged embeddings)
  Embed  : 64-dim dense embedding
  Hidden : 128-dim ReLU
  Output : vocab_size softmax (next word prediction)

Pure Python + Numpy only. No transformer. No library.
Runs on Celeron / Termux / Bodhi Linux.
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

STOP_GEN = {'a','an','the','and','or','but','so','if','then',
            'that','this','to','for','on','at','by','of','in'}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def keywords(text):
    return [w for w in tokenize(text) if w not in STOP and len(w) > 2]

def keyset(text):
    return set(keywords(text))


# ============================================================
# CONCEPT NORMALIZER  (from v8)
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
# FACT  (from v8)
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


# ============================================================
# FACT INDEX  (from v8)
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
# CONCEPT GRAPH  (from v8, simplified)
# ============================================================

class ConceptGraph:
    def __init__(self):
        self.edges = {}

    def add_fact(self, fact):
        concepts = list(fact.concepts)
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i == j: continue
                a, b = concepts[i], concepts[j]
                if a not in self.edges:
                    self.edges[a] = {}
                self.edges[a][b] = self.edges[a].get(b, 0) + 1

    def neighbors(self, concept, top_k=5):
        c = NORMALIZER.normalize(concept)
        nbrs = self.edges.get(c, {})
        return sorted(nbrs.items(), key=lambda x: -x[1])[:top_k]

    def multihop_path(self, start, end, max_hops=4):
        start = NORMALIZER.normalize(start)
        end   = NORMALIZER.normalize(end)
        if start not in self.edges: return None

        from collections import deque
        queue   = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            if len(path) > max_hops + 1: break
            node = path[-1]
            if node == end: return path
            for nbr in self.edges.get(node, {}):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(path + [nbr])
        return None


# ============================================================
# VOCABULARY  (from v9)
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
# ADAM OPTIMIZER  (from v9)
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
# NEURAL BRAIN  — v10 upgrade: multi-word context
# ============================================================

class NeuralBrain:
    """
    3-layer neural network with CONTEXT WINDOW (v10 upgrade).

    Instead of: one word → next word
    Now:        average of last 3 words → next word

    This is the fix for v9's "gravity is a molecule" problem.
    More context = better predictions.
    """

    CONTEXT = 3   # how many previous words to use

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

    # ── CONTEXT EMBEDDING ────────────────────────────────────

    def context_embed(self, context_ids):
        """
        Average embeddings of last N words.
        This is the key v10 upgrade over v9.

        context_ids: list of word indices (last 1-3 words)
        returns: averaged embedding vector (embed_dim,)
        """
        E = self.params['E']
        vecs = [E[i] for i in context_ids if i >= 0]
        if not vecs:
            return np.zeros(self.embed_dim)
        return np.mean(vecs, axis=0)   # average = context understanding

    # ── FORWARD ──────────────────────────────────────────────

    def forward(self, context_ids):
        """
        context_ids: list of recent word indices
        Returns: (probs, cache)
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        emb    = self.context_embed(context_ids)   # averaged context
        h_pre  = emb @ W1 + b1
        h      = np.maximum(0, h_pre)              # ReLU
        logits = h @ W2 + b2
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()

        cache = {'context_ids': context_ids, 'emb': emb,
                 'h_pre': h_pre, 'h': h, 'probs': probs}
        return probs, cache

    # ── BACKWARD ─────────────────────────────────────────────

    def backward(self, cache, y_id):
        probs      = cache['probs']
        h          = cache['h']
        h_pre      = cache['h_pre']
        emb        = cache['emb']
        context_ids = cache['context_ids']

        loss           = -np.log(probs[y_id] + 1e-10)
        d_logits       = probs.copy()
        d_logits[y_id] -= 1.0

        d_W2  = np.outer(h, d_logits)
        d_b2  = d_logits
        d_h   = self.params['W2'] @ d_logits
        d_h_pre = d_h * (h_pre > 0)
        d_W1  = np.outer(emb, d_h_pre)
        d_b1  = d_h_pre
        d_emb = self.params['W1'] @ d_h_pre

        # Distribute gradient back to all context words equally
        d_E = np.zeros_like(self.params['E'])
        n   = len(context_ids)
        for i in context_ids:
            if i >= 0:
                d_E[i] += d_emb / n   # divided equally

        grads = {'E': d_E, 'W1': d_W1, 'b1': d_b1,
                 'W2': d_W2, 'b2': d_b2}
        return loss, grads

    # ── TRAIN STEP ───────────────────────────────────────────

    def train_step(self, context_ids, y_id):
        probs, cache = self.forward(context_ids)
        loss, grads  = self.backward(cache, y_id)
        self.params  = self.optimizer.step(self.params, grads)
        return loss

    # ── PREDICT ──────────────────────────────────────────────

    def predict(self, context_ids, temperature=0.8, top_k=10):
        """Predict next word given context window."""
        probs, _ = self.forward(context_ids)
        top_ids  = np.argsort(probs)[-top_k:]
        top_p    = probs[top_ids]
        top_p    = top_p ** (1.0 / max(temperature, 0.1))
        top_p    = top_p / top_p.sum()
        return int(np.random.choice(top_ids, p=top_p))

    def predict_top(self, context_ids, k=5):
        probs, _ = self.forward(context_ids)
        top_ids  = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top_ids]

    # ── SAVE / LOAD ───────────────────────────────────────────

    def save(self, path):
        np.savez(path, **self.params,
                 vocab_size=np.array([self.vocab_size]),
                 embed_dim=np.array([self.embed_dim]),
                 hidden_dim=np.array([self.hidden_dim]))

    @classmethod
    def load(cls, path):
        d    = np.load(path + '.npz')
        vs   = int(d['vocab_size'][0])
        ed   = int(d['embed_dim'][0])
        hd   = int(d['hidden_dim'][0])
        brain = cls(vs, ed, hd)
        brain.params = {k: d[k] for k in ['E','W1','b1','W2','b2']}
        return brain


# ============================================================
# TRAINING  — context-window pairs
# ============================================================

def build_context_pairs(sentences, vocab, context_size=3):
    """
    Build training pairs using a sliding context window.

    For "gravity is a force that attracts mass":
      ([BOS], is)
      ([BOS, gravity], is)            ← partial context at start
      ([gravity, is], a)
      ([is, a], force)
      ([a, force], that)
      ([force, that], attracts)
      ...

    This gives the model REAL context, not just single words.
    """
    pad_id = vocab.encode(vocab.PAD)
    bos_id = vocab.encode(vocab.BOS)
    eos_id = vocab.encode(vocab.EOS)
    pairs  = []

    for sent in sentences:
        words = tokenize(sent)
        if len(words) < 2: continue
        ids = [bos_id] + [vocab.encode(w) for w in words] + [eos_id]

        for i in range(1, len(ids)):
            # Take last `context_size` words as context
            start      = max(0, i - context_size)
            context    = ids[start:i]
            # Pad with PAD_id if context is shorter
            while len(context) < context_size:
                context = [pad_id] + context
            target = ids[i]
            pairs.append((context, target))

    return pairs


def train_brain(sentences, epochs=30, embed_dim=64,
                hidden_dim=128, lr=0.002, verbose=True,
                context_size=3):
    # Build vocabulary
    vocab = Vocabulary()
    for sent in sentences:
        vocab.count(tokenize(sent))
    vocab.build()

    if verbose:
        print(f"[v10] Vocab: {vocab.size()} words")

    # Build context pairs
    pairs = build_context_pairs(sentences, vocab, context_size)
    if verbose:
        print(f"[v10] Training pairs: {len(pairs)} "
              f"(context window = {context_size})")

    brain = NeuralBrain(vocab.size(), embed_dim, hidden_dim)
    brain.optimizer.lr = lr

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        for context_ids, y_id in pairs:
            total_loss += brain.train_step(context_ids, y_id)
        avg = total_loss / max(len(pairs), 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

    return brain, vocab


# ============================================================
# GENERATION  — context-aware
# ============================================================

def generate(brain, vocab, seed_ids, max_words=30,
             temperature=0.7, top_k=8, context_size=3):
    """Generate text using sliding context window."""
    pad_id = vocab.encode(vocab.PAD)
    eos_id = vocab.encode(vocab.EOS)

    generated = list(seed_ids)

    for _ in range(max_words):
        # Build context window from last N generated words
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
# BUILT-IN FACTS  (expanded from v8+v9)
# ============================================================

BUILTIN_FACTS = [
    # Gravity
    "gravity is a fundamental force that attracts objects with mass.",
    "gravity pulls objects toward each other based on their mass.",
    "gravity keeps planets in orbit around the sun.",
    "gravity causes time to pass more slowly near massive objects.",
    "black holes have gravity so strong that light cannot escape.",
    "gravity is described by einstein general theory of relativity.",

    # Light
    "light is electromagnetic radiation that travels at 299792 kilometers per second.",
    "light can behave as both a wave and a particle.",
    "light carries energy and can transfer that energy to matter.",
    "light bends when passing near massive objects due to gravity.",
    "visible light is a small part of the electromagnetic spectrum.",

    # DNA
    "dna is a molecule that carries genetic instructions for life.",
    "dna is made of four bases adenine thymine cytosine and guanine.",
    "dna stores information in sequences of base pairs.",
    "dna replicates itself so cells can divide and copy information.",
    "mutations in dna drive the process of evolution over generations.",

    # Evolution
    "evolution is the change in inherited traits of populations over generations.",
    "natural selection favors traits that improve survival and reproduction.",
    "species evolve from common ancestors through accumulated mutations.",
    "evolution produces the diversity of life on earth.",

    # Neurons
    "neurons are cells that transmit electrical signals through the nervous system.",
    "neurons communicate using chemicals called neurotransmitters.",
    "synapses are junctions between neurons where signals pass.",
    "the brain contains about 86 billion neurons.",
    "memory is formed by strengthening connections between neurons.",

    # Memory
    "memory is the ability to store and recall past experiences.",
    "short term memory holds a small amount of information briefly.",
    "long term memory stores information for extended periods.",
    "sleep helps consolidate memories in the brain.",
    "the hippocampus is important for forming new memories.",

    # Stars
    "stars are massive balls of hot plasma that produce light and heat.",
    "stars generate energy through nuclear fusion in their cores.",
    "the sun is a medium sized star at the center of our solar system.",
    "stars produce heavier elements through fusion reactions.",
    "when massive stars die they explode in supernovae.",

    # Energy
    "energy is the ability to do work or cause change.",
    "energy cannot be created or destroyed only transformed.",
    "kinetic energy is the energy of motion.",
    "potential energy is stored energy due to position or configuration.",
    "nuclear energy is released when atomic nuclei split or fuse.",

    # Black holes
    "a black hole forms when a massive star collapses under its own gravity.",
    "the event horizon is the boundary beyond which nothing can escape.",
    "black holes warp space and time around them.",
    "hawking radiation is a theoretical process where black holes slowly lose energy.",

    # Water
    "water is a molecule made of two hydrogen atoms and one oxygen atom.",
    "water exists as liquid solid and gas depending on temperature.",
    "water is essential for all known forms of life.",
    "water covers about 71 percent of earth surface.",

    # AI
    "artificial intelligence is the simulation of human intelligence by machines.",
    "machine learning allows systems to learn from data without explicit programming.",
    "neural networks are computing systems inspired by biological neurons.",
    "deep learning uses many layers of neural networks to find patterns.",
    "language models learn to predict the next word from large text datasets.",
]

BRAIN_PATH = 'v10_brain'
VOCAB_PATH  = 'v10_vocab.json'
LEARNED_PATH = 'v10_learned.json'


def load_learned():
    if os.path.exists(LEARNED_PATH):
        with open(LEARNED_PATH) as f:
            return json.load(f)
    return []

def save_learned(data):
    with open(LEARNED_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# V8-STYLE REASONING ENGINE
# ============================================================

class ReasoningEngine:
    """
    The 'brain' from v8 — symbolic reasoning with concept graph.
    Used by v10 to understand what the question is about
    before handing off to the neural generator.
    """

    def __init__(self):
        self.index  = FactIndex()
        self.graph  = ConceptGraph()
        self.facts  = []

        for fact in BUILTIN_FACTS:
            self._add_fact(fact)

        for q, a in load_learned():
            self._add_fact(a)

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
        """Return most relevant facts for a question."""
        q_keys = keywords(question)
        return self.index.lookup_many(q_keys, top_k=top_k)

    def find_concept_chain(self, question):
        """
        Try to find a concept path relevant to the question.
        Returns list of concepts if found, else None.
        """
        q_words = [NORMALIZER.normalize(w) for w in keywords(question)]
        # Try pairs of question words
        for i in range(len(q_words)):
            for j in range(len(q_words)):
                if i == j: continue
                path = self.graph.multihop_path(
                    q_words[i], q_words[j], max_hops=3)
                if path and len(path) >= 2:
                    return path
        return None

    def best_fact(self, question):
        """Return the single most relevant fact text."""
        facts = self.find_relevant_facts(question, top_k=1)
        return facts[0].raw if facts else None


# ============================================================
# HYBRID ANSWER  — v8 thinks, v9 speaks
# ============================================================

def hybrid_answer(reasoning, brain, vocab, question,
                  max_words=30, temperature=0.65, top_k=8):
    """
    The core of v10:
      1. v8 reasoning engine finds relevant facts + concept chain
      2. Those concepts become the seed for v9 neural generator
      3. Neural generator produces natural language from the seed
    """
    # Step 1: Find relevant facts
    relevant_facts = reasoning.find_relevant_facts(question, top_k=3)
    concept_chain  = reasoning.find_concept_chain(question)

    # Step 2: Build a smart seed
    # Priority: concept chain > first fact words > question keywords
    seed_words = []

    if concept_chain:
        # Use concept chain as seed context
        seed_words = concept_chain[:4]
    elif relevant_facts:
        # Use first few words of best fact
        seed_words = tokenize(relevant_facts[0].raw)[:4]
    else:
        # Fall back to question keywords
        seed_words = keywords(question)[:3]

    if not seed_words:
        seed_words = keywords(question)[:2]

    # Step 3: Convert seed to vocab IDs
    seed_ids = [vocab.encode(w) for w in seed_words]

    # Step 4: Neural generation from seed
    result = generate(brain, vocab, seed_ids,
                      max_words=max_words,
                      temperature=temperature,
                      top_k=top_k)

    # Step 5: If generation is poor, fall back to direct fact
    if (not result or len(result.split()) < 3) and relevant_facts:
        raw = relevant_facts[0].raw
        return raw[0].upper() + raw[1:], 'direct'

    return result, 'neural'


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

        # Give first word(s) as context, predict next
        ctx = [pad_id] * context_size
        ctx[-1] = vocab.encode(words[0])
        top      = brain.predict_top(ctx, k=5)
        top_words = [vocab.decode(i) for i, _ in top]
        hit = words[1] in top_words
        if hit: correct += 1
        total += 1
        results.append({
            'seed': words[0], 'expected': words[1],
            'predicted': top_words[0], 'hit': hit,
        })

    acc = correct / max(total, 1) * 100
    print(f"\n{'='*52}")
    print(f"  BRAIN SELF-CHECK  (context={context_size})")
    print(f"{'='*52}")
    print(f"  Facts tested   : {total}")
    print(f"  Top-5 accuracy : {acc:.1f}%")
    if acc >= 60: print("  ✓ Brain learned facts well")
    elif acc >= 30: print("  ~ Partial — train more epochs")
    else:           print("  ✗ Needs more training")
    print(f"\n  Sample predictions:")
    for r in results[:5]:
        s = '✓' if r['hit'] else '✗'
        print(f"  {s} '{r['seed']}' → "
              f"expected='{r['expected']}' "
              f"got='{r['predicted']}'")
    print()
    return acc


# ============================================================
# MAIN SYSTEM
# ============================================================

class GFSDMv10:
    """
    v10 — Hybrid Brain
    
    v8 symbolic reasoning  → understands the question
    v9 neural generation   → speaks the answer
    
    Together: better than either alone.
    """

    CONTEXT = 3   # words of context for generation

    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.brain     = None
        self.vocab     = None

        if (os.path.exists(BRAIN_PATH + '.npz') and
                os.path.exists(VOCAB_PATH)):
            self.brain = NeuralBrain.load(BRAIN_PATH)
            self.vocab = Vocabulary()
            self.vocab.load(VOCAB_PATH)
            print(f"[v10] Neural brain loaded. "
                  f"Vocab: {self.vocab.size()} words")
        else:
            print("[v10] No saved brain. Run ai.train() first.")

    # ── TRAINING ─────────────────────────────────────────────

    def train(self, extra_text=None, epochs=40,
              embed_dim=64, hidden_dim=128, lr=0.002):
        sentences = list(BUILTIN_FACTS)

        if extra_text:
            new_sents = [s.strip() for s in
                         re.split(r'[.!?]', extra_text) if len(s.strip()) > 20]
            sentences.extend(new_sents)
            print(f"[v10] Extra: +{len(new_sents)} sentences")

        # Also add learned facts
        for _, a in load_learned():
            if a: sentences.append(a)

        print(f"[v10] Training on {len(sentences)} sentences, "
              f"context={self.CONTEXT}, epochs={epochs}...")

        self.brain, self.vocab = train_brain(
            sentences, epochs=epochs,
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            lr=lr, verbose=True, context_size=self.CONTEXT)

        self.brain.save(BRAIN_PATH)
        self.vocab.save(VOCAB_PATH)
        print("[v10] Brain saved.")

    def train_file(self, path, epochs=30):
        if not os.path.exists(path):
            print(f"[v10] Not found: {path}")
            return
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self.train(extra_text=text, epochs=epochs)

    # ── CHAT  — hybrid mode ───────────────────────────────────

    def chat(self, question, temperature=0.65, top_k=8):
        question = question.strip()
        if not question: return "I'm listening."

        # Check for teaching input
        teach = self._detect_teach(question)
        if teach:
            self.reasoning.teach(teach)
            return f"Understood. Added: {teach}"

        # Hybrid: v8 reasons, v9 speaks
        if self.brain:
            answer, mode = hybrid_answer(
                self.reasoning, self.brain, self.vocab,
                question, max_words=30,
                temperature=temperature, top_k=top_k)

            # Annotate mode
            if mode == 'direct':
                return f"(direct fact)\n{answer}"
            else:
                # Also show which facts informed the generation
                facts = self.reasoning.find_relevant_facts(question, 1)
                hint  = f" [from: {facts[0].raw[:50]}...]" if facts else ""
                return f"(neural+symbolic{hint})\n{answer}"
        else:
            # v8-only fallback if no neural brain
            facts = self.reasoning.find_relevant_facts(question, 1)
            if facts:
                raw = facts[0].raw
                return "(symbolic only)\n" + raw[0].upper() + raw[1:]
            return "I don't have enough information. Try teaching me or /train."

    # ── TEACH ─────────────────────────────────────────────────

    def teach(self, fact):
        """Teach a new fact. Retrain neural brain to absorb it."""
        self.reasoning.teach(fact.lower().rstrip('.') + '.')
        print(f"[v10] Taught: {fact}")
        print("[v10] Run ai.train() to update neural weights.")

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
        return ' → '.join(path)

    def check(self, n=20):
        if not self.brain:
            print("No brain."); return
        self_check(self.brain, self.vocab, BUILTIN_FACTS,
                   n=n, context_size=self.CONTEXT)

    def stats(self):
        lines = [
            f"GF-SDM v10 — Hybrid Brain",
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
            ]
        else:
            lines.append("  Neural brain: not trained")
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
  /hop <w1> <w2>        — concept path between two words
  /inspect <word>       — facts containing word
  /stats                — system stats
  /teach <sentence>     — add new fact
  /help                 — this help
  quit                  — exit

Teaching (no /):
  "DNA is a molecule that stores genetic information"
  → Auto-detected and stored

v10 = v8 (reason) + v9 (speak)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GF-SDM v10  |  Hybrid Brain")
    print("  v8 symbolic reasoning  +  v9 neural generation")
    print("  Context window: 3 words  |  Pure Python + Numpy")
    print("=" * 60 + "\n")

    ai = GFSDMv10()

    if not ai.brain:
        print("[v10] Auto-training...")
        ai.train(epochs=40)
        print()

    ai.check(n=15)

    print("\n--- Hybrid Generation Demo ---\n")
    demos = [
        "what is gravity",
        "what is memory",
        "how does dna work",
        "what is a black hole",
        "how do neurons connect",
        "what is energy",
        "how does evolution happen",
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
        if user == '/check':     ai.check(); continue
        if user == '/stats':     print(f"\n{ai.stats()}\n"); continue
        if user == '/help':      print(HELP); continue
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
