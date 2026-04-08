"""
GF-SDM v9 — Numpy Neural Brain
================================
Train a real neural network (pure numpy) on our fact dataset.
The weights ARE the brain — like an LLM but tiny and local.

Architecture:
  Input  : one-hot word (vocab_size)
  Embed  : 64-dim dense embedding
  Hidden : 128-dim ReLU
  Output : vocab_size softmax (next word prediction)

Training:
  - Extract facts from dataset (v8 style)
  - Train on word sequences: given word[i] → predict word[i+1]
  - Loss: cross-entropy
  - Optimizer: Adam (pure numpy)
  - Save weights to brain.npz

Inference:
  - Load brain.npz
  - Give seed words → generate answer word by word
  - Like a tiny LLM — but no transformer, no library

Pure Python + Numpy only. i3-2100 / Termux ready.
"""

import os, re, json, random, math
import numpy as np

# ============================================================
# TOKENIZER + VOCAB
# ============================================================

STOP_GEN = {'a','an','the','and','or','but','so','if','then',
            'that','this','to','for','on','at','by','of','in'}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

class Vocabulary:
    PAD = '<PAD>'
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    def __init__(self, min_freq=1):
        self.min_freq  = min_freq
        self.word2id   = {}
        self.id2word   = []
        self.freq      = {}
        # Reserve special tokens
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
# ADAM OPTIMIZER (pure numpy)
# ============================================================

class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.t   = 0
        self.m   = {}
        self.v   = {}

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
# NEURAL BRAIN (pure numpy)
# ============================================================

class NeuralBrain:
    """
    3-layer neural network:
      Embedding (vocab → embed_dim)
      Hidden    (embed_dim → hidden_dim) + ReLU
      Output    (hidden_dim → vocab) + Softmax

    Trained on next-word prediction from fact sequences.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim

        # Xavier initialization
        self.params = {
            'E' : np.random.randn(vocab_size, embed_dim)  * 0.1,
            'W1': np.random.randn(embed_dim, hidden_dim)  * np.sqrt(2/embed_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, vocab_size) * np.sqrt(2/hidden_dim),
            'b2': np.zeros(vocab_size),
        }
        self.optimizer = Adam(lr=0.002)

    # ── FORWARD ──────────────────────────────────────────────

    def forward(self, x_id):
        """
        x_id: integer word index
        Returns: (probs, cache)
        """
        E  = self.params['E']
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        # Embedding lookup
        emb = E[x_id]                        # (embed_dim,)

        # Hidden layer + ReLU
        h_pre = emb @ W1 + b1               # (hidden_dim,)
        h     = np.maximum(0, h_pre)         # ReLU

        # Output + softmax
        logits = h @ W2 + b2                 # (vocab_size,)
        logits -= logits.max()               # numerical stability
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()         # softmax

        cache = {'x_id': x_id, 'emb': emb,
                 'h_pre': h_pre, 'h': h,
                 'probs': probs}
        return probs, cache

    # ── BACKWARD ─────────────────────────────────────────────

    def backward(self, cache, y_id):
        """
        cache: from forward()
        y_id : correct next word index
        Returns: (loss, grads)
        """
        probs  = cache['probs']
        h      = cache['h']
        h_pre  = cache['h_pre']
        emb    = cache['emb']
        x_id   = cache['x_id']

        # Cross-entropy loss
        loss = -np.log(probs[y_id] + 1e-10)

        # Output layer grad
        d_logits        = probs.copy()
        d_logits[y_id] -= 1.0               # dL/d_logits

        d_W2 = np.outer(h, d_logits)        # (hidden, vocab)
        d_b2 = d_logits                     # (vocab,)
        d_h  = self.params['W2'] @ d_logits # (hidden,)

        # ReLU grad
        d_h_pre = d_h * (h_pre > 0)        # ReLU derivative

        # Hidden layer grad
        d_W1  = np.outer(emb, d_h_pre)     # (embed, hidden)
        d_b1  = d_h_pre                    # (hidden,)
        d_emb = self.params['W1'] @ d_h_pre# (embed,)

        # Embedding grad
        d_E        = np.zeros_like(self.params['E'])
        d_E[x_id] += d_emb

        grads = {'E': d_E, 'W1': d_W1, 'b1': d_b1,
                 'W2': d_W2, 'b2': d_b2}
        return loss, grads

    # ── TRAIN STEP ───────────────────────────────────────────

    def train_step(self, x_id, y_id):
        probs, cache = self.forward(x_id)
        loss, grads  = self.backward(cache, y_id)
        self.params  = self.optimizer.step(self.params, grads)
        return loss

    # ── PREDICT ──────────────────────────────────────────────

    def predict(self, x_id, temperature=0.8, top_k=10):
        """Predict next word given current word."""
        probs, _ = self.forward(x_id)

        # Top-k sampling
        top_ids  = np.argsort(probs)[-top_k:]
        top_p    = probs[top_ids]

        # Temperature
        top_p = top_p ** (1.0 / max(temperature, 0.1))
        top_p = top_p / top_p.sum()

        return int(np.random.choice(top_ids, p=top_p))

    def predict_top(self, x_id, k=5):
        """Return top-k most likely next words."""
        probs, _ = self.forward(x_id)
        top_ids  = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top_ids]

    # ── SAVE / LOAD ───────────────────────────────────────────

    def save(self, path):
        np.savez(path,
                 E =self.params['E'],
                 W1=self.params['W1'],
                 b1=self.params['b1'],
                 W2=self.params['W2'],
                 b2=self.params['b2'],
                 meta=np.array([self.vocab_size,
                                self.embed_dim,
                                self.hidden_dim]))
        print(f"[Brain] Saved → {path}.npz")

    @classmethod
    def load(cls, path):
        d    = np.load(path + '.npz')
        meta = d['meta'].tolist()
        brain = cls(int(meta[0]), int(meta[1]), int(meta[2]))
        brain.params = {
            'E' : d['E'],
            'W1': d['W1'],
            'b1': d['b1'],
            'W2': d['W2'],
            'b2': d['b2'],
        }
        print(f"[Brain] Loaded ← {path}.npz")
        return brain


# ============================================================
# FACT EXTRACTOR (from v8)
# ============================================================

RELATION_WORDS = {
    'is','are','was','were','contains','produces','uses',
    'causes','enables','requires','connects','forms','carries',
    'stores','transfers','converts','attracts','releases',
    'made','found','called','known','defined','described',
    'moves','travels','creates','generates','transmits','absorbs'
}

def extract_sentences(text, min_w=5, max_w=50):
    """Extract clean sentences from plain text."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    for s in raw:
        s = s.strip().rstrip('.!?')
        words = re.findall(r"[a-zA-Z']+", s)
        if min_w <= len(words) <= max_w:
            alpha = sum(c.isalpha() for c in s)
            if len(s) > 0 and alpha/len(s) > 0.6:
                out.append(s.lower())
    return out


# ============================================================
# TRAINING PIPELINE
# ============================================================

BRAIN_PATH = "gf_sdm_v9_brain"
VOCAB_PATH = "gf_sdm_v9_vocab.json"

BUILTIN_FACTS = [
    "gravity is a force that attracts objects with mass toward each other",
    "energy is the ability to do work and comes in many forms like heat light and motion",
    "an atom is the smallest unit of an element made of protons neutrons and electrons",
    "an electron is a negatively charged particle that orbits the nucleus of an atom",
    "matter is anything that has mass and takes up space",
    "mass is the amount of matter in an object",
    "force is a push or pull that changes the motion of an object",
    "light is electromagnetic radiation that is visible to the human eye",
    "electricity is the flow of electric charge through a conductor",
    "a wave is a disturbance that transfers energy through space or matter",
    "heat is thermal energy transferred from a hotter object to a cooler one",
    "sound is a wave of pressure that travels through air and other materials",
    "nuclear fusion is the process where two atomic nuclei combine releasing enormous energy",
    "radiation is energy emitted as electromagnetic waves from unstable atoms",
    "a molecule is two or more atoms bonded together",
    "oxygen is a gas needed for breathing and combustion in living things",
    "carbon is an element that forms the basis of all known life on earth",
    "water is a molecule made of two hydrogen atoms and one oxygen atom",
    "photosynthesis is the process by which plants convert sunlight into sugar and oxygen",
    "combustion is a reaction where a substance reacts with oxygen to produce heat and light",
    "dna is a molecule that carries genetic instructions for all living organisms",
    "a cell is the basic structural unit of all living organisms on earth",
    "evolution is the process by which species change over time through natural selection",
    "a gene is a segment of dna that carries instructions for making a protein",
    "a protein is a large molecule made of amino acids that performs body functions",
    "a neuron is a nerve cell that transmits electrical signals through the nervous system",
    "a synapse is the junction between neurons where signals are transmitted",
    "the brain is the organ that controls body functions and is the center of thought",
    "the nervous system carries signals through the body using brain spinal cord and nerves",
    "metabolism is all chemical reactions that occur in a living organism to maintain life",
    "consciousness is the state of being aware of one's thoughts feelings and surroundings",
    "memory is the ability of the brain to store retain and recall information",
    "intelligence is the ability to learn reason solve problems and adapt to situations",
    "learning is the process of acquiring knowledge skills or behaviors through experience",
    "reasoning is the process of thinking logically to form conclusions from evidence",
    "language is a system of communication using sounds symbols or gestures to convey meaning",
    "thought is a mental process involving reasoning imagination and problem solving",
    "knowledge is justified true belief acquired through experience or education",
    "an algorithm is a step by step procedure for solving a problem or completing a task",
    "mathematics is the study of numbers shapes patterns and logical relationships",
    "logic is the study of valid reasoning and principles of correct inference",
    "artificial intelligence is the simulation of human intelligence by computer systems",
    "machine learning is where systems learn from data to improve their performance",
    "a neural network is a computing system inspired by the human brain made of connected nodes",
    "a computer is an electronic device that processes and stores information",
    "the internet is a global network of computers connected together to share information",
    "the universe is all of space time matter and energy that exists",
    "the big bang is the theory that the universe began from an extremely hot dense state",
    "a star is a massive ball of gas that produces energy through nuclear fusion",
    "the sun is the star at the center of our solar system providing light and heat to earth",
    "a black hole is a region of space where gravity is so strong that nothing can escape it",
    "a galaxy is a large system of stars gas and dark matter held together by gravity",
    "dark matter is an invisible substance making up most of the universe mass",
    "a supernova is the explosive death of a massive star that releases enormous energy",
    "orbit is the curved path an object takes around another object due to gravity",
    "a planet is a large body that orbits a star and has cleared its orbital path",
    "earth is the third planet from the sun and the only known planet to support life",
    "space is the vast region beyond earth atmosphere where stars and galaxies exist",
    "time is the continuous progression of existence from past through present to future",
    "the atmosphere is the layer of gases surrounding earth that protects life and regulates climate",
    "climate is the average weather conditions of a region over a long period of time",
    "philosophy is the study of fundamental questions about existence knowledge and reality",
    "truth is the property of statements that accurately correspond to reality",
    "ethics is the branch of philosophy concerned with moral principles and right conduct",
    "science is the systematic study of the natural world through observation and experiment",
    "technology is the application of scientific knowledge to solve practical problems",
    "natural selection is the process where organisms with favorable traits survive and reproduce",
    "biodiversity is the variety of life in a habitat or on earth as a whole",
    "the hippocampus is a brain region crucial for forming and storing new memories",
    "synaptic plasticity is the ability of synapses to strengthen based on activity enabling memory",
    "neurons form synaptic connections that store memories through repeated electrical signals",
    "dna carries genetic mutations that drive evolution through natural selection over generations",
    "stars produce energy through nuclear fusion converting hydrogen into helium and releasing light",
    "gravity becomes so strong in black holes that nothing including light can escape their pull",
    "the hippocampus plays a crucial role in forming new memories and spatial navigation",
    "deep learning models use multiple layers of neural networks to learn patterns from data",
    "black holes form when massive stars collapse under their own gravity after a supernova",
    "evolution occurs through natural selection where advantageous traits spread through populations",
    "plants use chlorophyll to absorb sunlight during photosynthesis and produce oxygen",
    "the speed of light is approximately three hundred thousand kilometers per second in vacuum",
    "an ecosystem is a community of living organisms interacting with their physical environment",
    "a mutation is a change in dna sequence that can be inherited and drives evolution",
    "heredity is the passing of genetic traits through dna from parents to offspring",
    "probability is the measure of how likely an event is to occur between zero and one",
    "calculus is the branch of mathematics that studies rates of change and accumulation",
    "a vaccine trains the immune system to recognize and fight specific pathogens",
    "the immune system defends the body against pathogens using white blood cells and antibodies",
    "antibodies are proteins produced by the immune system to neutralize foreign substances",
    "temperature is a measure of the average kinetic energy of particles in a substance",
    "pressure is the force applied per unit area on a surface",
    "momentum is the product of an object mass and its velocity",
    "acceleration is the rate at which the velocity of an object changes over time",
    "frequency is the number of waves that pass a fixed point per second",
    "a transistor is a semiconductor device used to amplify or switch electronic signals",
    "binary is a number system using only zero and one to represent all digital data",
    "encryption is the process of converting information into a code to prevent unauthorized access",
    "a database is an organized collection of structured information stored and accessed electronically",
    "software is a set of instructions written in a programming language that a computer executes",
]


def build_training_sequences(sentences, vocab):
    """
    Convert sentences into (input_word_id, output_word_id) pairs
    for next-word prediction training.
    """
    pairs = []
    bos_id = vocab.encode(Vocabulary.BOS)
    eos_id = vocab.encode(Vocabulary.EOS)

    for sent in sentences:
        words = tokenize(sent)
        ids   = [bos_id] + [vocab.encode(w) for w in words] + [eos_id]
        for i in range(len(ids) - 1):
            pairs.append((ids[i], ids[i+1]))
    return pairs


def train_brain(sentences, epochs=30, embed_dim=64,
                hidden_dim=128, lr=0.002, verbose=True):
    """Full training pipeline — returns trained brain + vocab."""

    # Build vocabulary
    vocab = Vocabulary(min_freq=1)
    for sent in sentences:
        vocab.count(tokenize(sent))
    vocab.build()

    if verbose:
        print(f"[Train] Vocab size : {vocab.size()}")

    # Build training pairs
    pairs = build_training_sequences(sentences, vocab)
    if verbose:
        print(f"[Train] Training pairs : {len(pairs)}")

    # Init brain
    brain = NeuralBrain(vocab.size(), embed_dim, hidden_dim)
    brain.optimizer.lr = lr

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0

        for x_id, y_id in pairs:
            loss = brain.train_step(x_id, y_id)
            total_loss += loss

        avg_loss = total_loss / max(len(pairs), 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"loss={avg_loss:.4f} | "
                  f"best={best_loss:.4f}")

    if verbose:
        print(f"[Train] Done. Best loss: {best_loss:.4f}")

    return brain, vocab


# ============================================================
# GENERATOR  — LLM-style generation from neural brain
# ============================================================

def generate(brain, vocab, seed_words, max_words=30,
             temperature=0.8, top_k=10):
    """
    Generate text from seed words using neural brain.
    Like an LLM — word by word from neural weights.
    """
    words  = tokenize(' '.join(seed_words))
    ids    = [vocab.encode(w) for w in words]

    if not ids:
        ids = [vocab.encode(Vocabulary.BOS)]

    output = list(words)
    eos_id = vocab.encode(Vocabulary.EOS)

    # Use last word as context (bigram-style)
    context_id = ids[-1]

    for _ in range(max_words):
        next_id = brain.predict(context_id, temperature, top_k)

        if next_id == eos_id:
            break

        word = vocab.decode(next_id)
        if word in (Vocabulary.PAD, Vocabulary.BOS, Vocabulary.UNK):
            context_id = next_id
            continue

        output.append(word)
        context_id = next_id

    # Format output
    result = ' '.join(output)
    result = result[0].upper() + result[1:] if result else ''
    return result


def answer_question(brain, vocab, question, max_words=25,
                    temperature=0.5, top_k=8):
    """
    Answer a question using neural brain generation.
    Seeds with question keywords, generates the answer.
    """
    q_words = tokenize(question)
    # Use content words as seed
    seed = [w for w in q_words if w not in STOP_GEN][-3:]
    if not seed:
        seed = q_words[-2:]

    return generate(brain, vocab, seed, max_words, temperature, top_k)


# ============================================================
# SELF-CHECK  — how well does the brain know the facts?
# ============================================================

def self_check(brain, vocab, facts, n=20, verbose=True):
    """
    Test brain on n random facts.
    For each fact, give first word, check if brain predicts correctly.
    """
    sample = random.sample(facts, min(n, len(facts)))
    correct = 0
    total   = 0

    results = []
    for fact in sample:
        words = tokenize(fact)
        if len(words) < 3: continue

        # Give first word, predict second
        x_id = vocab.encode(words[0])
        top  = brain.predict_top(x_id, k=5)
        top_words = [vocab.decode(i) for i, _ in top]

        hit = words[1] in top_words
        if hit: correct += 1
        total += 1

        results.append({
            'fact'      : fact[:50],
            'seed'      : words[0],
            'expected'  : words[1],
            'predicted' : top_words[0],
            'hit'       : hit,
        })

    acc = correct / max(total, 1) * 100

    if verbose:
        print(f"\n{'='*52}")
        print(f"  BRAIN SELF-CHECK")
        print(f"{'='*52}")
        print(f"  Facts tested   : {total}")
        print(f"  Top-5 accuracy : {acc:.1f}%")
        print(f"{'='*52}")
        if acc >= 60:
            print("  ✓ Brain learned facts well")
        elif acc >= 30:
            print("  ~ Brain partially learned — train more epochs")
        else:
            print("  ✗ Brain needs more training")

        print(f"\n  Sample predictions:")
        for r in results[:5]:
            status = '✓' if r['hit'] else '✗'
            print(f"  {status} '{r['seed']}' → "
                  f"expected='{r['expected']}' "
                  f"got='{r['predicted']}'")
        print()

    return acc


# ============================================================
# MAIN SYSTEM
# ============================================================

class GFSDMv9:
    """
    v9 — Numpy Neural Brain
    Trains a real neural network on facts.
    Generates answers LLM-style from neural weights.
    """

    def __init__(self):
        self.brain  = None
        self.vocab  = None
        self.facts  = list(BUILTIN_FACTS)

        # Try loading saved brain
        if os.path.exists(BRAIN_PATH + '.npz') and \
           os.path.exists(VOCAB_PATH):
            self.brain = NeuralBrain.load(BRAIN_PATH)
            self.vocab = Vocabulary()
            self.vocab.load(VOCAB_PATH)
            print(f"[v9] Vocab: {self.vocab.size()} words")
        else:
            print("[v9] No saved brain. Run train() first.")

    def train(self, extra_text=None, epochs=30,
              embed_dim=64, hidden_dim=128, lr=0.002):
        """Train neural brain on built-in facts + optional extra text."""
        sentences = list(self.facts)

        # Extract sentences from extra text
        if extra_text:
            new_sents = extract_sentences(extra_text)
            sentences.extend(new_sents)
            print(f"[v9] Extra text: +{len(new_sents)} sentences")

        print(f"[v9] Training on {len(sentences)} sentences...")
        print(f"[v9] embed={embed_dim} hidden={hidden_dim} "
              f"epochs={epochs} lr={lr}")

        self.brain, self.vocab = train_brain(
            sentences, epochs=epochs,
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            lr=lr, verbose=True)

        # Save
        self.brain.save(BRAIN_PATH)
        self.vocab.save(VOCAB_PATH)
        print("[v9] Brain + vocab saved.")

    def train_file(self, path, epochs=20):
        """Train from a text file."""
        if not os.path.exists(path):
            print(f"[v9] File not found: {path}")
            return
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        print(f"[v9] Training from {path} ({len(text)} chars)...")
        self.train(extra_text=text, epochs=epochs)

    def chat(self, question, temperature=0.6, top_k=8):
        """Answer a question using neural brain."""
        if not self.brain:
            return "Brain not trained yet. Run: ai.train()"
        return answer_question(self.brain, self.vocab, question,
                               max_words=25,
                               temperature=temperature,
                               top_k=top_k)

    def generate(self, seed, max_words=40, temperature=0.8):
        """Free generation from seed."""
        if not self.brain:
            return "Brain not trained yet."
        return generate(self.brain, self.vocab,
                        tokenize(seed), max_words, temperature)

    def self_check(self, n=20):
        """Test how well brain learned the facts."""
        if not self.brain:
            print("Brain not trained yet.")
            return
        self_check(self.brain, self.vocab, self.facts, n=n)

    def teach(self, fact_sentence):
        """Add a new fact and retrain (quick fine-tune)."""
        self.facts.append(fact_sentence.lower().rstrip('.') + '.')
        print(f"[v9] Fact added. Retrain with ai.train() to update brain.")

    def stats(self):
        if not self.brain:
            return "No brain loaded."
        p = self.brain.params
        total_params = sum(v.size for v in p.values())
        return (f"Vocab     : {self.vocab.size()} words\n"
                f"Facts     : {len(self.facts)}\n"
                f"Embed dim : {self.brain.embed_dim}\n"
                f"Hidden dim: {self.brain.hidden_dim}\n"
                f"Parameters: {total_params:,} weights\n"
                f"Brain file: {BRAIN_PATH}.npz")


# ============================================================
# ENTRY POINT
# ============================================================

HELP = """
Commands:
  /train [epochs]   — train brain on built-in facts
  /train_file <f>   — train from text file
  /check            — self-check brain accuracy
  /gen <seed>       — free generation from seed
  /stats            — brain stats
  /teach <sentence> — add new fact (then retrain)
  /help             — this help
  quit              — exit

After training:
  Ask any question → brain generates answer from weights
  Like a tiny LLM!

Tips:
  More epochs = better learning (try 50-100)
  More facts = richer brain
  /train_file book.txt → brain grows from your data
"""

if __name__ == "__main__":
    print("=" * 58)
    print("  GF-SDM v9  |  Numpy Neural Brain")
    print("  Real weights. Real training. No Transformer.")
    print("=" * 58 + "\n")

    ai = GFSDMv9()

    # Auto-train if no brain
    if not ai.brain:
        print("[v9] Auto-training on built-in facts...")
        ai.train(epochs=40)
        print()

    # Self-check
    ai.self_check(n=20)

    # Demo generation
    print("--- Neural Generation Demo ---\n")
    demos = [
        "what is gravity",
        "what is memory",
        "how does dna work",
        "what connects neurons",
        "what is a black hole",
        "how does evolution happen",
    ]
    for q in demos:
        print(f"Q: {q}")
        print(f"A: {ai.chat(q)}")
        print()

    print("-" * 58)
    print("Talk to the brain!\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user: continue
        if user.lower() in ('quit','exit','q'): break
        if user == '/check':     ai.self_check(); continue
        if user == '/stats':     print(f"\n{ai.stats()}\n"); continue
        if user == '/help':      print(HELP); continue
        if user.startswith('/train_file '):
            ai.train_file(user[12:].strip()); continue
        if user.startswith('/gen '):
            print(f"\n{ai.generate(user[5:].strip())}\n"); continue
        if user.startswith('/teach '):
            ai.teach(user[7:].strip()); continue
        if user.startswith('/train'):
            parts  = user.split()
            epochs = int(parts[1]) if len(parts) > 1 else 40
            ai.train(epochs=epochs); continue

        ans = ai.chat(user)
        print(f"AI: {ans}\n")

    print("\n[Done]")
