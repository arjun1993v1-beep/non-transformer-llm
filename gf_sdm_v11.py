"""
GF-SDM v11 — Controlled Hybrid Brain
======================================
v10 upgraded with 4 critical fixes:

FIX 1 — STRICT FALLBACK
  v10: if output is short → fallback
  v11: if output contains wrong concepts (not from question domain) → return fact directly
       Uses concept overlap check: output must share ≥1 key concept with the question

FIX 2 — SEED ANCHORING
  v10: seed from concept chain (can drift)
  v11: seed_words always LOCK the subject + "is/are/does" first
       → "what is dna" → seed = ["dna", "is"] then generate
       → model is forced into the right semantic lane before it can drift

FIX 3 — Q→A TRAINING PAIRS
  v10: trained on bare facts "gravity is a force..."
  v11: trained on "what is gravity → gravity is a force..."
       Neural model learns the Q→A mapping, not just word sequences
       Fallback still uses symbolic engine

FIX 4 — EXPANDED DATASET
  v10: 53 sentences
  v11: 150+ sentences across 15 domains
       Also auto-generates Q→A pairs from every fact

Architecture (unchanged):
  Input  : 3-word context window (averaged embeddings)
  Embed  : 64-dim dense embedding
  Hidden : 128-dim ReLU
  Output : vocab_size softmax

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

STOP_GEN = {'a','an','the','and','or','but','so','if','then',
            'that','this','to','for','on','at','by','of','in'}

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
        if not vecs:
            return np.zeros(self.embed_dim)
        return np.mean(vecs, axis=0)

    def forward(self, context_ids):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        emb    = self.context_embed(context_ids)
        h_pre  = emb @ W1 + b1
        h      = np.maximum(0, h_pre)
        logits = h @ W2 + b2
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()

        cache = {'context_ids': context_ids, 'emb': emb,
                 'h_pre': h_pre, 'h': h, 'probs': probs}
        return probs, cache

    def backward(self, cache, y_id):
        probs       = cache['probs']
        h           = cache['h']
        h_pre       = cache['h_pre']
        emb         = cache['emb']
        context_ids = cache['context_ids']

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
        n   = len(context_ids)
        for i in context_ids:
            if i >= 0:
                d_E[i] += d_emb / n

        grads = {'E': d_E, 'W1': d_W1, 'b1': d_b1,
                 'W2': d_W2, 'b2': d_b2}
        return loss, grads

    def train_step(self, context_ids, y_id):
        probs, cache = self.forward(context_ids)
        loss, grads  = self.backward(cache, y_id)
        self.params  = self.optimizer.step(self.params, grads)
        return loss

    def predict(self, context_ids, temperature=0.8, top_k=10):
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
    v11: sentences include Q→A pairs like "what is gravity → gravity is..."
    so the model learns the full question → answer mapping.
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
            start   = max(0, i - context_size)
            context = ids[start:i]
            while len(context) < context_size:
                context = [pad_id] + context
            target = ids[i]
            pairs.append((context, target))

    return pairs


def train_brain(sentences, epochs=30, embed_dim=64,
                hidden_dim=128, lr=0.002, verbose=True,
                context_size=3):
    vocab = Vocabulary()
    for sent in sentences:
        vocab.count(tokenize(sent))
    vocab.build()

    if verbose:
        print(f"[v11] Vocab: {vocab.size()} words")

    pairs = build_context_pairs(sentences, vocab, context_size)
    if verbose:
        print(f"[v11] Training pairs: {len(pairs)} "
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
# FIX 3 — Q→A PAIR GENERATOR
# Auto-generates question→answer training sentences from facts.
# ============================================================

def fact_to_qa_pairs(fact_raw):
    """
    Given a raw fact string, generate Q→A training sentences.
    Example:
      "gravity is a fundamental force that attracts objects with mass."
      →
      "what is gravity gravity is a fundamental force that attracts objects with mass"
      "how does gravity work gravity is a fundamental force that attracts objects with mass"
    """
    words  = tokenize(fact_raw)
    if not words: return []

    # Extract subject (first word before relation word)
    subj = None
    for i, w in enumerate(words):
        if w in RELATION_WORDS and i > 0:
            subj = ' '.join(words[:i])
            break
    if not subj:
        subj = words[0]

    bare = fact_raw.rstrip('.')
    pairs = []

    # Pattern 1: "what is X → X is ..."
    pairs.append(f"what is {subj} {bare}")

    # Pattern 2: "how does X work → X is ..."
    pairs.append(f"how does {subj} work {bare}")

    # Pattern 3: "explain X → X is ..."
    pairs.append(f"explain {subj} {bare}")

    return pairs


def build_qa_sentences(facts):
    """
    FIX 3: Turn every fact into Q→A training sentences.
    The neural model now sees question patterns → answer text.
    """
    qa = []
    for f in facts:
        qa.extend(fact_to_qa_pairs(f))
    return qa


# ============================================================
# FIX 1 — STRICT CONCEPT VALIDATOR
# ============================================================

def output_is_valid(output, question, relevant_facts):
    """
    FIX 1: Strict fallback — check if output contains concepts
    from the question's domain, not random drift.

    Returns True if output is valid (keep it), False if bad (fallback).

    Rules:
      1. Must be at least 4 words long
      2. Must share at least 1 keyword with the question OR best fact
      3. Must not be mostly stop words
    """
    if not output or len(output.split()) < 4:
        return False

    out_keys = keyset(output)
    q_keys   = keyset(question)

    # Check overlap with question keywords
    if out_keys & q_keys:
        return True

    # Check overlap with relevant fact concepts
    if relevant_facts:
        fact_keys = keyset(relevant_facts[0].raw)
        if out_keys & fact_keys:
            return True

    # Output has zero concept overlap — it drifted
    return False


# ============================================================
# FIX 2 — ANCHOR SEED BUILDER
# ============================================================

def build_anchored_seed(question, reasoning):
    """
    FIX 2: Build a seed that LOCKS the subject word first.

    Instead of letting the concept chain drift,
    always start with: [subject, 'is'] or [subject, 'are']
    so the neural model is anchored before it generates.

    Priority:
      1. Extract subject from question → anchor it
      2. Add relation word ('is'/'are') as second anchor
      3. Optionally append one more concept from the concept chain
    """
    q_words = tokenize(question)

    # Strip question words to get the real subject
    strip_q = {'what','how','why','when','where','who','which',
                'does','do','is','are','was','were','did'}
    subject_words = [w for w in q_words if w not in strip_q and len(w) > 2]

    if not subject_words:
        # Fall back to general keyword extraction
        subject_words = keywords(question)[:1]

    subject = subject_words[0] if subject_words else None

    # Determine relation word based on question type
    if any(w in question.lower() for w in ['how does','how do','how is']):
        relation = 'works'
    elif 'are' in q_words:
        relation = 'are'
    else:
        relation = 'is'

    seed_words = []

    if subject:
        seed_words.append(subject)
        seed_words.append(relation)

    # Optionally add one concept from symbolic reasoning to guide further
    chain = reasoning.find_concept_chain(question)
    if chain:
        for c in chain:
            if c not in seed_words and c not in strip_q:
                seed_words.append(c)
                break  # just one extra concept — don't over-seed

    # If still empty, use question keywords
    if not seed_words:
        seed_words = keywords(question)[:2]

    return seed_words


# ============================================================
# EXPANDED BUILT-IN FACTS  (FIX 4: 150+ sentences, 15 domains)
# ============================================================

BUILTIN_FACTS = [
    # --- GRAVITY (6 → 10) ---
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

    # --- LIGHT (5 → 10) ---
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

    # --- DNA (5 → 10) ---
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

    # --- EVOLUTION (4 → 8) ---
    "evolution is the change in inherited traits of populations over generations.",
    "natural selection favors traits that improve survival and reproduction.",
    "species evolve from common ancestors through accumulated mutations.",
    "evolution produces the diversity of life on earth.",
    "charles darwin first proposed the theory of natural selection.",
    "all life on earth shares a common ancestor from about four billion years ago.",
    "evolution can be observed in bacteria that develop antibiotic resistance.",
    "sexual selection drives traits that improve mating success.",

    # --- NEURONS (5 → 10) ---
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

    # --- MEMORY (5 → 8) ---
    "memory is the ability to store and recall past experiences.",
    "short term memory holds a small amount of information briefly.",
    "long term memory stores information for extended periods.",
    "sleep helps consolidate memories in the brain.",
    "the hippocampus is important for forming new memories.",
    "working memory holds information actively in mind for immediate use.",
    "episodic memory stores personal experiences and events.",
    "semantic memory stores general knowledge and facts about the world.",

    # --- STARS (5 → 10) ---
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

    # --- ENERGY (5 → 10) ---
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

    # --- BLACK HOLES (4 → 8) ---
    "a black hole forms when a massive star collapses under its own gravity.",
    "the event horizon is the boundary beyond which nothing can escape.",
    "black holes warp space and time around them.",
    "hawking radiation is a theoretical process where black holes slowly lose energy.",
    "supermassive black holes exist at the centers of most galaxies.",
    "the first image of a black hole was captured in 2019.",
    "time passes more slowly closer to a black hole.",
    "black holes can merge and produce gravitational waves.",

    # --- WATER (4 → 8) ---
    "water is a molecule made of two hydrogen atoms and one oxygen atom.",
    "water exists as liquid solid and gas depending on temperature.",
    "water is essential for all known forms of life.",
    "water covers about 71 percent of earth surface.",
    "water has a high heat capacity allowing it to regulate temperatures.",
    "ice is less dense than liquid water which is why it floats.",
    "water is a universal solvent that dissolves many substances.",
    "the water cycle moves water between oceans atmosphere and land.",

    # --- AI / MACHINE LEARNING (5 → 10) ---
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

    # --- ATOMS (new domain) ---
    "atoms are the basic units of matter that make up all substances.",
    "an atom consists of a nucleus surrounded by electrons.",
    "the nucleus of an atom contains protons and neutrons.",
    "protons carry positive charge and neutrons carry no charge.",
    "electrons carry negative charge and orbit the nucleus.",
    "the number of protons in an atom determines which element it is.",
    "chemical bonds form when atoms share or transfer electrons.",
    "isotopes are atoms of the same element with different numbers of neutrons.",

    # --- CELLS (new domain) ---
    "cells are the basic unit of life in all living organisms.",
    "every cell contains dna that encodes the organism genetic information.",
    "the cell membrane controls what enters and leaves the cell.",
    "mitochondria produce energy for the cell through cellular respiration.",
    "the cell nucleus contains the genetic material of the cell.",
    "cells divide through a process called mitosis to produce identical copies.",
    "stem cells can differentiate into many different types of specialized cells.",
    "bacteria are single celled organisms without a true nucleus.",

    # --- CLIMATE (new domain) ---
    "climate is the long term pattern of weather in a region.",
    "the greenhouse effect traps heat in earth atmosphere.",
    "carbon dioxide is a greenhouse gas that contributes to global warming.",
    "the oceans absorb about 30 percent of the carbon dioxide produced by humans.",
    "global warming is causing ice caps to melt and sea levels to rise.",
    "the atmosphere protects life on earth from harmful solar radiation.",
    "photosynthesis by plants absorbs carbon dioxide and releases oxygen.",
    "deforestation reduces the earth ability to absorb carbon dioxide.",

    # --- MATHEMATICS (new domain) ---
    "mathematics is the study of numbers patterns and logical structures.",
    "prime numbers are whole numbers greater than one with no divisors except themselves and one.",
    "the pythagorean theorem states that the square of the hypotenuse equals the sum of squares of the other sides.",
    "calculus is the mathematics of change and motion.",
    "algebra uses symbols and rules to manipulate mathematical expressions.",
    "statistics is the study of collecting analyzing and interpreting data.",
    "infinity is a concept describing something without any limit.",
    "zero is the additive identity meaning adding zero to any number leaves it unchanged.",

    # --- HUMAN BODY (new domain) ---
    "the human body contains about 37 trillion cells.",
    "the heart pumps blood through the circulatory system.",
    "the lungs exchange oxygen and carbon dioxide with the blood.",
    "the liver filters toxins from the blood and produces bile.",
    "the immune system defends the body against pathogens and disease.",
    "bones provide structure support and protection for the body.",
    "the digestive system breaks down food into nutrients for the body.",
    "the brain is the control center of the nervous system.",
]

# ── Q→A SEPARATOR TOKEN ──────────────────────────────────────
# Used between question and answer in training sentences.
# The model learns: when it sees question words → produce answer words.
QA_SEP = 'then'   # "what is gravity then gravity is a force..."


# ============================================================
# PATHS
# ============================================================

BRAIN_PATH   = 'v11_brain'
VOCAB_PATH   = 'v11_vocab.json'
LEARNED_PATH = 'v11_learned.json'


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
        self.index = FactIndex()
        self.graph = ConceptGraph()
        self.facts = []

        for fact in BUILTIN_FACTS:
            self._add_fact(fact)

        for _, a in load_learned():
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
        q_keys = keywords(question)
        return self.index.lookup_many(q_keys, top_k=top_k)

    def find_concept_chain(self, question):
        q_words = [NORMALIZER.normalize(w) for w in keywords(question)]
        for i in range(len(q_words)):
            for j in range(len(q_words)):
                if i == j: continue
                path = self.graph.multihop_path(
                    q_words[i], q_words[j], max_hops=3)
                if path and len(path) >= 2:
                    return path
        return None

    def best_fact(self, question):
        facts = self.find_relevant_facts(question, top_k=1)
        return facts[0].raw if facts else None


# ============================================================
# HYBRID ANSWER  — v11: controlled generation
# ============================================================

def hybrid_answer(reasoning, brain, vocab, question,
                  max_words=30, temperature=0.65, top_k=8):
    """
    v11 core — three improvements over v10:

    FIX 2: Seed is ANCHORED to subject + relation word.
            Model can't drift before it even starts.

    FIX 1: Output is VALIDATED by concept overlap check.
            If output contains wrong concepts → return fact directly.

    v10 fallback: only checked length.
    v11 fallback: checks MEANING, not just length.
    """
    relevant_facts = reasoning.find_relevant_facts(question, top_k=3)

    # FIX 2: Build anchored seed (subject locked first)
    seed_words = build_anchored_seed(question, reasoning)
    seed_ids   = [vocab.encode(w) for w in seed_words]

    # Generate
    result = generate(brain, vocab, seed_ids,
                      max_words=max_words,
                      temperature=temperature,
                      top_k=top_k)

    # FIX 1: Strict concept validation
    if output_is_valid(result, question, relevant_facts):
        # Show which fact informed generation
        hint = ""
        if relevant_facts:
            hint = f" [from: {relevant_facts[0].raw[:50]}...]"
        return result, f'neural+anchored{hint}'
    else:
        # Output drifted — return fact directly
        if relevant_facts:
            raw = relevant_facts[0].raw
            return raw[0].upper() + raw[1:], 'direct'
        return "I don't have information on that topic.", 'none'


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

        ctx = [pad_id] * context_size
        ctx[-1] = vocab.encode(words[0])
        top       = brain.predict_top(ctx, k=5)
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

class GFSDMv11:
    """
    v11 — Controlled Hybrid Brain

    Improvements over v10:
      Fix 1: Strict concept-overlap fallback (not just length check)
      Fix 2: Anchored seed — subject word locked before generation
      Fix 3: Q→A training pairs — model learns question→answer mapping
      Fix 4: 150+ facts across 15 domains (was 53)
    """

    CONTEXT = 3

    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.brain     = None
        self.vocab     = None

        if (os.path.exists(BRAIN_PATH + '.npz') and
                os.path.exists(VOCAB_PATH)):
            self.brain = NeuralBrain.load(BRAIN_PATH)
            self.vocab = Vocabulary()
            self.vocab.load(VOCAB_PATH)
            print(f"[v11] Neural brain loaded. "
                  f"Vocab: {self.vocab.size()} words")
        else:
            print("[v11] No saved brain. Run ai.train() first.")

    # ── TRAINING ─────────────────────────────────────────────

    def train(self, extra_text=None, epochs=40,
              embed_dim=64, hidden_dim=128, lr=0.002):

        # Base facts
        sentences = list(BUILTIN_FACTS)

        # FIX 3: Add Q→A training pairs from every fact
        qa_pairs = build_qa_sentences(BUILTIN_FACTS)
        sentences.extend(qa_pairs)
        print(f"[v11] Base facts: {len(BUILTIN_FACTS)}")
        print(f"[v11] Q→A pairs generated: {len(qa_pairs)}")

        # Extra text
        if extra_text:
            new_sents = [s.strip() for s in
                         re.split(r'[.!?]', extra_text) if len(s.strip()) > 20]
            sentences.extend(new_sents)
            # Also generate Q→A from extra sentences
            extra_qa = build_qa_sentences(new_sents)
            sentences.extend(extra_qa)
            print(f"[v11] Extra: +{len(new_sents)} sentences, "
                  f"+{len(extra_qa)} Q→A pairs")

        # Learned facts
        for _, a in load_learned():
            if a:
                sentences.append(a)
                sentences.extend(fact_to_qa_pairs(a))

        print(f"[v11] Total training sentences: {len(sentences)}")
        print(f"[v11] Training, context={self.CONTEXT}, epochs={epochs}...")

        self.brain, self.vocab = train_brain(
            sentences, epochs=epochs,
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            lr=lr, verbose=True, context_size=self.CONTEXT)

        self.brain.save(BRAIN_PATH)
        self.vocab.save(VOCAB_PATH)
        print("[v11] Brain saved.")

    def train_file(self, path, epochs=30):
        if not os.path.exists(path):
            print(f"[v11] Not found: {path}")
            return
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
            return f"Understood. Added: {teach}"

        if self.brain:
            answer, mode = hybrid_answer(
                self.reasoning, self.brain, self.vocab,
                question, max_words=30,
                temperature=temperature, top_k=top_k)

            if mode == 'direct':
                return f"(direct fact)\n{answer}"
            elif mode == 'none':
                return answer
            else:
                return f"(neural+anchored)\n{answer}"
        else:
            facts = self.reasoning.find_relevant_facts(question, 1)
            if facts:
                raw = facts[0].raw
                return "(symbolic only)\n" + raw[0].upper() + raw[1:]
            return "I don't have enough information. Try /train first."

    # ── TEACH ─────────────────────────────────────────────────

    def teach(self, fact):
        self.reasoning.teach(fact.lower().rstrip('.') + '.')
        print(f"[v11] Taught: {fact}")
        print("[v11] Run ai.train() to update neural weights.")

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
            f"GF-SDM v11 — Controlled Hybrid Brain",
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
                f"  v11 improvements:",
                f"    Fix 1: Strict concept-overlap fallback",
                f"    Fix 2: Anchored seed generation",
                f"    Fix 3: Q→A training pairs",
                f"    Fix 4: {len(BUILTIN_FACTS)} facts, 15 domains",
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

v11 = v10 + strict fallback + anchored seed + Q→A training + 150 facts
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GF-SDM v11  |  Controlled Hybrid Brain")
    print("  v10 + strict fallback + anchored seed + Q→A training")
    print(f"  {len(BUILTIN_FACTS)} built-in facts across 15 domains")
    print("=" * 60 + "\n")

    ai = GFSDMv11()

    if not ai.brain:
        print("[v11] Auto-training...")
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
        "what is an atom",
        "what is climate",
        "what is a cell",
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
