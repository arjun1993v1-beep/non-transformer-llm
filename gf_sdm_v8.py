"""
GF-SDM v8 — Self-Building Brain
=================================
1. Concept Normalization  — synonyms collapse to one concept
2. Multi-hop Composition  — A→C→B reasoning chains
3. Auto Dataset Expansion — plain text in, brain grows out

Pure Python. Zero dependencies. i3-2100 / Termux ready.
"""

import os, re, json, random, math

# ============================================================
# STOPWORDS
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
# 1. CONCEPT NORMALIZER
# ============================================================

class ConceptNormalizer:
    """
    Maps synonyms and variants to a single canonical concept.

    attracts / pulls / draws       → attract
    massive / enormous / huge      → large
    produces / generates / creates → produce
    carries / transports / moves   → carry
    ...

    This fixes reasoning:
    "gravity attracts mass" + "stars have massive bodies"
    → both map to [attract, large, mass]
    → system sees they share concepts correctly
    """

    SYNONYM_GROUPS = [
        # Motion / force
        (['attracts','attract','pulls','pull','draws','draw','draws'], 'attract'),
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
        # Size / scale
        (['massive','enormous','huge','vast','giant','large','big',
          'great','immense'], 'large'),
        (['tiny','small','little','minute','micro','mini'], 'small'),
        (['hot','warm','heated','thermal'], 'heat'),
        (['cold','cool','frozen','icy'], 'cold'),
        # Structure
        (['made','built','composed','formed','consisting'], 'made'),
        (['found','located','situated','present','exists'], 'found'),
        (['called','named','known','termed','defined'], 'called'),
        (['similar','alike','same','identical','equivalent'], 'similar'),
        # Knowledge
        (['learns','learn','acquires','acquire','gains','gain'], 'learn'),
        (['thinks','think','reasons','reason','processes','process'], 'think'),
        (['knows','know','understands','understand','recognizes','recognize'], 'know'),
        (['measures','measure','calculates','calculate','counts','count'], 'measure'),
        # Biology
        (['grows','grow','develops','develop','evolves','evolve'], 'grow'),
        (['dies','die','decays','decay','destroys','destroy'], 'die'),
        (['lives','live','survives','survive','exists','exist'], 'live'),
        (['reproduces','reproduce','copies','copy','replicates','replicate'], 'reproduce'),
        # Energy / light
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
# FACT  — normalized
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
        # Normalized concept set — key for reasoning
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
# FACT INDEX  — normalized concept lookup
# ============================================================

class FactIndex:
    def __init__(self):
        self.facts         = []
        self.concept_index = {}   # normalized_concept → [fact_ids]

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
# 2. CONCEPT GRAPH  — multi-hop reasoning
# ============================================================

class ConceptGraph:
    """
    Weighted directed graph of normalized concepts.
    Edge A→B means: A and B co-occur in a fact.
    Weight = number of facts they co-occur in.

    Multi-hop: find path A→C→D→B
    → each hop = one shared fact
    → chain = reasoning steps
    """

    def __init__(self):
        self.edges = {}     # {concept: {neighbor: weight}}
        self.fact_map = {}  # {(c1,c2): fact_raw} for path narration

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
        if c not in self.edges: return []
        return sorted(self.edges[c].items(), key=lambda x: -x[1])[:top_k]

    def get_edge_fact(self, c1, c2):
        key = (min(c1,c2), max(c1,c2))
        return self.fact_map.get(key, '')

    def multihop_path(self, src, dst, max_hops=4):
        """
        BFS — find shortest concept path src→...→dst.
        Returns list of (concept, edge_fact) pairs.
        """
        src = NORMALIZER.normalize(src)
        dst = NORMALIZER.normalize(dst)
        if src == dst: return [(src, '')]
        if src not in self.edges: return []

        visited = {src}
        # queue items: [(concept, fact_used)]
        queue = [[(src, '')]]

        while queue:
            path = queue.pop(0)
            node = path[-1][0]
            if len(path) > max_hops: continue
            for nb, _ in self.neighbors(node, top_k=6):
                edge_fact = self.get_edge_fact(node, nb)
                new_path  = path + [(nb, edge_fact)]
                if nb == dst:
                    return new_path
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
# 3. AUTO DATASET EXPANDER — plain text → facts
# ============================================================

class DatasetExpander:
    """
    Reads plain text and auto-extracts Q&A facts.

    Strategy:
    - Split into sentences
    - Each sentence with a relation word → potential fact
    - Extract subject + relation + objects
    - Generate question: "what is/does <subject>?"
    - Store as Q&A pair
    """

    MIN_WORDS   = 5
    MAX_WORDS   = 50

    def extract_facts(self, text, verbose=True):
        sentences = self._split_sentences(text)
        facts     = []
        skipped   = 0

        for sent in sentences:
            words = tokenize(sent)
            if len(words) < self.MIN_WORDS or len(words) > self.MAX_WORDS:
                skipped += 1
                continue

            # Must have at least one relation word OR enough content
            has_rel = any(w in RELATION_WORDS for w in words)
            content = [w for w in words if w not in STOP and len(w) > 2]
            if not has_rel and len(content) < 5:
                skipped += 1
                continue

            # Clean sentence
            clean = sent.strip().rstrip('.!?') + '.'
            clean = clean[0].lower() + clean[1:]

            # Generate question from subject
            f = Fact(clean)
            if f.subject and len(f.subject.split()) <= 5:
                q = f"what is {f.subject}"
                facts.append((q, clean))

        if verbose:
            print(f"[Expander] {len(sentences)} sentences → "
                  f"{len(facts)} facts extracted "
                  f"({skipped} skipped)")

        return facts

    def _split_sentences(self, text):
        # Split on '.', '!', '?' then clean
        raw = re.split(r'(?<=[.!?])\s+', text)
        out = []
        for s in raw:
            s = s.strip()
            # Remove lines with too many special chars
            alpha = sum(c.isalpha() for c in s)
            if len(s) > 0 and alpha / len(s) > 0.6:
                out.append(s)
        return out

    def from_file(self, path, verbose=True):
        if not os.path.exists(path):
            print(f"[Expander] File not found: {path}")
            return []
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        if verbose:
            print(f"[Expander] Reading {path} ({len(text)} chars)...")
        return self.extract_facts(text, verbose=verbose)

    def from_string(self, text, verbose=False):
        return self.extract_facts(text, verbose=verbose)


# ============================================================
# MULTI-HOP COMPOSER
# ============================================================

class MultiHopComposer:
    """
    Given a path A→C→D→B, narrates the reasoning chain
    as a natural paragraph.
    """

    def narrate_path(self, path, src_word, dst_word):
        """
        path = [(concept, edge_fact), ...]
        Returns a natural language explanation of the chain.
        """
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

        # Deduplicate
        seen  = set()
        dedup = []
        for s in steps:
            if s not in seen:
                seen.add(s)
                dedup.append(s)

        chain = ' '.join(dedup)
        concepts = ' → '.join(c for c, _ in path)
        intro = f"Reasoning chain ({concepts}):\n"
        return intro + chain

    def compose_direct(self, A, B, q_concepts):
        """Direct two-fact composition via shared normalized concepts."""
        shared = (A.concepts & B.concepts) - STOP
        shared |= (q_concepts & (A.concepts | B.concepts)) - STOP

        if not shared:
            return None, 0.0

        total = len(A.concepts | B.concepts)
        score = len(shared) / max(total, 1)
        bridge = sorted(shared, key=lambda w: -len(w))[0]

        templates = [
            f"{A.raw[0].upper()+A.raw[1:]}. {B.raw[0].upper()+B.raw[1:]}. Both involve {bridge}.",
            f"Regarding {bridge}: {A.raw}. This connects to {B.subject}: {B.raw}.",
            f"{A.raw[0].upper()+A.raw[1:]}. Similarly, {B.raw}. The link is {bridge}.",
        ]
        return random.choice(templates), score


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class Memory:
    def __init__(self, max_turns=5):
        self.history = []
        self.topics  = []
        self.max     = max_turns

    def add(self, user, bot):
        self.history.append((user, bot))
        if len(self.history) > self.max:
            self.history = self.history[-self.max:]
        self.topics = (self.topics + keywords(user))[-20:]

    def enrich(self, q):
        words = q.strip().split()
        if len(words) == 1 and self.topics:
            return q + ' ' + self.topics[-1]
        return q

    def last_topic(self):
        return self.topics[-1] if self.topics else None

    def summary(self):
        if not self.history: return "No conversation yet."
        lines = []
        for u, b in self.history[-3:]:
            lines.append(f"  You : {u}")
            lines.append(f"  AI  : {b[:70]}...")
        return '\n'.join(lines)

    def clear(self):
        self.history = []
        self.topics  = []


# ============================================================
# BUILT-IN Q&A BASE
# ============================================================

BUILTIN_QA = [
    ("what is gravity","gravity is a force that attracts objects with mass toward each other"),
    ("what is energy","energy is the ability to do work and comes in many forms like heat light and motion"),
    ("what is an atom","an atom is the smallest unit of an element made of protons neutrons and electrons"),
    ("what is an electron","an electron is a negatively charged particle that orbits the nucleus of an atom"),
    ("what is matter","matter is anything that has mass and takes up space"),
    ("what is mass","mass is the amount of matter in an object"),
    ("what is force","force is a push or pull that changes the motion of an object"),
    ("what is light","light is electromagnetic radiation that is visible to the human eye"),
    ("what is electricity","electricity is the flow of electric charge through a conductor"),
    ("what is a wave","a wave is a disturbance that transfers energy through space or matter"),
    ("what is heat","heat is thermal energy transferred from a hotter object to a cooler one"),
    ("what is sound","sound is a wave of pressure that travels through air and other materials"),
    ("what is nuclear fusion","nuclear fusion is the process where two atomic nuclei combine releasing enormous energy"),
    ("what is radiation","radiation is energy emitted as electromagnetic waves from unstable atoms"),
    ("what is a molecule","a molecule is two or more atoms bonded together"),
    ("what is oxygen","oxygen is a gas needed for breathing and combustion"),
    ("what is carbon","carbon is an element that forms the basis of all known life on earth"),
    ("what is water","water is a molecule made of two hydrogen atoms and one oxygen atom"),
    ("what is photosynthesis","photosynthesis is the process by which plants convert sunlight into sugar and oxygen"),
    ("what is combustion","combustion is a reaction where a substance reacts with oxygen to produce heat and light"),
    ("what is DNA","DNA is a molecule that carries genetic instructions for living organisms"),
    ("what is a cell","a cell is the basic structural unit of all living organisms"),
    ("what is evolution","evolution is the process by which species change over time through natural selection"),
    ("what is a gene","a gene is a segment of DNA that carries instructions for making a protein"),
    ("what is a protein","a protein is a large molecule made of amino acids that performs body functions"),
    ("what is a neuron","a neuron is a nerve cell that transmits electrical signals through the nervous system"),
    ("what is a synapse","a synapse is the junction between neurons where signals are transmitted"),
    ("what is the brain","the brain is the organ that controls body functions and is the center of thought"),
    ("what is the nervous system","the nervous system carries signals through the body using brain spinal cord and nerves"),
    ("what is metabolism","metabolism is all chemical reactions that occur in a living organism to maintain life"),
    ("what is consciousness","consciousness is the state of being aware of one's thoughts feelings and surroundings"),
    ("what is memory","memory is the ability of the brain to store retain and recall information"),
    ("what is intelligence","intelligence is the ability to learn reason solve problems and adapt to situations"),
    ("what is learning","learning is the process of acquiring knowledge skills or behaviors through experience"),
    ("what is reasoning","reasoning is the process of thinking logically to form conclusions from evidence"),
    ("what is language","language is a system of communication using sounds symbols or gestures"),
    ("what is thought","thought is a mental process involving reasoning imagination and problem solving"),
    ("what is knowledge","knowledge is justified true belief acquired through experience or education"),
    ("what is an algorithm","an algorithm is a step by step procedure for solving a problem"),
    ("what is mathematics","mathematics is the study of numbers shapes patterns and logical relationships"),
    ("what is logic","logic is the study of valid reasoning and principles of correct inference"),
    ("what is artificial intelligence","artificial intelligence is the simulation of human intelligence by computer systems"),
    ("what is machine learning","machine learning is where systems learn from data to improve their performance"),
    ("what is a neural network","a neural network is a computing system inspired by the human brain"),
    ("what is a computer","a computer is an electronic device that processes and stores information"),
    ("what is the internet","the internet is a global network of computers sharing information"),
    ("what is the universe","the universe is all of space time matter and energy that exists"),
    ("what is the big bang","the big bang is the theory that the universe began from an extremely hot dense state"),
    ("what is a star","a star is a massive ball of gas that produces energy through nuclear fusion"),
    ("what is the sun","the sun is the star at the center of our solar system providing light and heat"),
    ("what is a black hole","a black hole is a region of space where gravity is so strong nothing can escape"),
    ("what is a galaxy","a galaxy is a large system of stars gas and dark matter held together by gravity"),
    ("what is dark matter","dark matter is an invisible substance making up most of the universe's mass"),
    ("what is a supernova","a supernova is the explosive death of a massive star releasing enormous energy"),
    ("what is orbit","orbit is the curved path an object takes around another due to gravity"),
    ("what is a planet","a planet is a large body that orbits a star and has cleared its orbital path"),
    ("what is earth","earth is the third planet from the sun and the only known planet to support life"),
    ("what is space","space is the vast region beyond earth's atmosphere where stars and galaxies exist"),
    ("what is time","time is the continuous progression of existence from past through present to future"),
    ("what is the atmosphere","the atmosphere is the layer of gases surrounding earth that protects life"),
    ("what is climate","climate is the average weather conditions of a region over a long period"),
    ("what is philosophy","philosophy is the study of fundamental questions about existence knowledge and reality"),
    ("what is truth","truth is the property of statements that accurately correspond to reality"),
    ("what is ethics","ethics is the branch of philosophy concerned with moral principles and right conduct"),
    ("what is science","science is the systematic study of the natural world through observation and experiment"),
    ("what is technology","technology is the application of scientific knowledge to solve practical problems"),
    ("what is natural selection","natural selection is the process where organisms with favorable traits survive and reproduce"),
    ("what is biodiversity","biodiversity is the variety of life in a habitat or on earth as a whole"),
    ("what is an ecosystem","an ecosystem is a community of organisms interacting with their environment"),
    ("what connects neurons and memory","neurons form synaptic connections that store memories through repeated electrical signals"),
    ("what connects dna and evolution","dna carries genetic mutations that drive evolution through natural selection over generations"),
    ("what connects stars and energy","stars produce energy through nuclear fusion converting hydrogen into helium and releasing light"),
    ("what is a mutation","a mutation is a change in dna sequence that can be inherited and drives evolution"),
    ("what is heredity","heredity is the passing of genetic traits through dna from parents to offspring"),
    ("what is synaptic plasticity","synaptic plasticity is the ability of synapses to strengthen or weaken based on activity enabling memory"),
]


# ============================================================
# PERSISTENCE
# ============================================================

BRAIN_FILE   = "gf_sdm_v8_brain.json"
LEARNED_FILE = "gf_sdm_v8_learned.json"

def load_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE) as f: return json.load(f)
        except: pass
    return []

def save_brain(qa_pairs):
    with open(BRAIN_FILE, 'w') as f:
        json.dump([[q,a] for q,a in qa_pairs], f, indent=2)
    print(f"[v8] Brain saved → {BRAIN_FILE} ({len(qa_pairs)} facts)")

def load_learned():
    if os.path.exists(LEARNED_FILE):
        try:
            with open(LEARNED_FILE) as f: return json.load(f)
        except: pass
    return []

def save_learned(pairs):
    with open(LEARNED_FILE,'w') as f: json.dump(pairs, f, indent=2)


# ============================================================
# MAIN AI — v8
# ============================================================

class GFSDMv8:
    def __init__(self):
        self.qa_pairs  = list(BUILTIN_QA)
        self.index     = FactIndex()
        self.graph     = ConceptGraph()
        self.composer  = MultiHopComposer()
        self.expander  = DatasetExpander()
        self.memory    = Memory(max_turns=5)
        self.verbose   = False

        # Load brain (previously expanded facts)
        brain = load_brain()
        if brain:
            extra = [(q,a) for q,a in brain
                     if (q,a) not in self.qa_pairs]
            self.qa_pairs.extend(extra)
            print(f"[v8] Brain loaded: {len(brain)} facts.")

        # Load user-taught facts
        learned = load_learned()
        if learned:
            extra = [(q,a) for q,a in learned
                     if (q,a) not in self.qa_pairs]
            self.qa_pairs.extend(extra)
            print(f"[v8] Learned: {len(learned)} user facts.")

        self._build_index()

    def _build_index(self):
        self.index = FactIndex()
        self.graph = ConceptGraph()
        for _, answer in self.qa_pairs:
            fid  = self.index.add(answer)
            fact = self.index.facts[fid]
            self.graph.add_fact(fact)
        print(f"[v8] Index: {len(self.index.concept_index)} concepts | "
              f"Graph: {len(self.graph.edges)} nodes")

    # ── TRAIN on plain text / file ────────────────────────────

    def train_text(self, text, verbose=True):
        """Extract facts from plain text and add to brain."""
        new_facts = self.expander.from_string(text, verbose=verbose)
        added = 0
        existing = {a for _,a in self.qa_pairs}
        for q, a in new_facts:
            if a not in existing:
                self.qa_pairs.append((q, a))
                existing.add(a)
                added += 1
        if added:
            self._build_index()
            save_brain(self.qa_pairs)
        if verbose:
            print(f"[v8] Added {added} new facts. "
                  f"Total: {len(self.qa_pairs)}")
        return added

    def train_file(self, path):
        """Train from a text file."""
        new_facts = self.expander.from_file(path, verbose=True)
        added = 0
        existing = {a for _,a in self.qa_pairs}
        for q, a in new_facts:
            if a not in existing:
                self.qa_pairs.append((q, a))
                existing.add(a)
                added += 1
        if added:
            self._build_index()
            save_brain(self.qa_pairs)
        print(f"[v8] Trained from {path}: +{added} facts. "
              f"Total: {len(self.qa_pairs)}")
        return added

    # ── THINK ─────────────────────────────────────────────────

    def think(self, question):
        q_norm     = NORMALIZER.normalize_set(keyset(question))
        q_words    = list(q_norm)

        if self.verbose:
            print(f"\n  [think] normalized: {', '.join(sorted(q_norm))}")

        # Retrieve top facts by normalized concept overlap
        facts = self.index.lookup_many(q_norm, top_k=6)

        if self.verbose:
            print(f"  [think] {len(facts)} facts retrieved")

        # Try multi-hop if question has 2+ distinct concepts
        if len(q_words) >= 2:
            hop_result = self._try_multihop(q_words)
            if hop_result:
                if self.verbose:
                    print(f"  [think] multi-hop path found!")
                return hop_result, 0.75

        # Direct pairwise composition
        if len(facts) >= 2:
            best_s, best_score = None, 0.0
            for i in range(len(facts)):
                for j in range(i+1, len(facts)):
                    s, sc = self.composer.compose_direct(
                        facts[i], facts[j], q_norm)
                    if sc > best_score:
                        best_score = sc
                        best_s     = s
            if best_s and best_score > 0.05:
                if self.verbose:
                    print(f"  [think] composed score: {best_score:.2f}")
                return best_s, best_score

        # Single fact expand
        if facts:
            f    = facts[0]
            extra = sorted((q_norm - f.concepts), key=lambda w: -len(w))[:3]
            sent  = f.raw[0].upper() + f.raw[1:] + '.'
            if extra:
                sent += f" This relates to {', '.join(extra)}."
            return sent, 0.55

        return None, 0.0

    def _try_multihop(self, q_words):
        """Try to find a multi-hop path between question concepts."""
        for i in range(len(q_words)):
            for j in range(len(q_words)):
                if i == j: continue
                path = self.graph.multihop_path(
                    q_words[i], q_words[j], max_hops=4)
                if path and len(path) >= 3:
                    return self.composer.narrate_path(
                        path, q_words[i], q_words[j])
        return None

    # ── CHAT ─────────────────────────────────────────────────

    def chat(self, user_input):
        user_input = user_input.strip()
        if not user_input: return "I'm listening."

        # Detect teaching
        teach = self._detect_teach(user_input)
        if teach:
            q, a = teach
            existing = [eq for eq,_ in self.qa_pairs]
            if q not in existing:
                self.qa_pairs.append((q, a))
                fid  = self.index.add(a)
                fact = self.index.facts[fid]
                self.graph.add_fact(fact)
                learned = load_learned()
                learned.append([q, a])
                save_learned(learned)
                response = f"Understood. I'll remember: {a}."
            else:
                response = "I already know that."
            self.memory.add(user_input, response)
            return response

        enriched = self.memory.enrich(user_input)
        answer, score = self.think(enriched)

        if not answer or score < 0.04:
            last = self.memory.last_topic()
            response = (f"I don't have enough to answer that. "
                       + (f"We were on '{last}' — want to continue?" if last
                          else "Try teaching me or train on a text file."))
        else:
            pct   = int(min(score, 0.99) * 100)
            label = self._label(score)
            response = f"({pct}% — {label})\n{answer}"

        self.memory.add(user_input, response)
        return response

    def _label(self, s):
        if s >= 0.75: return "multi-hop reasoning"
        if s >= 0.6:  return "confident"
        if s >= 0.4:  return "fairly sure"
        if s >= 0.2:  return "inferring"
        return "weak inference"

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
                return f"what is {subj}", f"{subj} is {defn}"
        return None

    def multihop(self, w1, w2):
        """Manual multi-hop between two concepts."""
        path = self.graph.multihop_path(
            NORMALIZER.normalize(w1),
            NORMALIZER.normalize(w2), max_hops=5)
        if not path:
            return f"No path found between '{w1}' and '{w2}'."
        return self.composer.narrate_path(path, w1, w2)

    def inspect(self, word):
        facts = self.index.lookup(word)
        if not facts:
            return f"No facts for '{word}'."
        lines = [f"Facts about '{word}' (normalized):"]
        for f in facts[:5]:
            lines.append(f"  · {f.raw}")
        return '\n'.join(lines)

    def toggle_verbose(self):
        self.verbose = not self.verbose
        return f"Thinking steps {'ON' if self.verbose else 'OFF'}."

    def stats(self):
        base    = len(BUILTIN_QA)
        total   = len(self.qa_pairs)
        learned = total - base
        return (f"Facts: {total} ({base} built-in + {learned} added)\n"
                f"  Concepts : {len(self.index.concept_index)}\n"
                f"  Graph    : {len(self.graph.edges)} nodes\n"
                f"  Synonyms : {len(NORMALIZER.word_to_canon)} mapped")


# ============================================================
# ENTRY POINT
# ============================================================

HELP = """
Commands:
  /think              — toggle thinking steps
  /hop <w1> <w2>      — multi-hop path between concepts
  /inspect <word>     — show facts containing concept
  /train <file.txt>   — train brain from text file
  /stats              — knowledge stats
  /history            — recent conversation
  /forget             — clear memory
  /help               — this help
  quit                — exit

Teaching:
  "ARIA is an AI companion with emotional memory"
  "Qwen3 is a large language model by Alibaba"

Training from text:
  /train book.txt
  /train wikipedia_excerpt.txt
  → brain automatically grows
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GF-SDM v8  |  Self-Building Brain")
    print("  Normalize + Multi-hop + Auto-expand")
    print("  Pure Python | No Transformer | No Backprop")
    print("=" * 60)

    ai = GFSDMv8()
    print(f"\n  Type /help for commands.\n")

    # Demo
    print("--- Demo: Multi-hop Reasoning ---\n")
    demos = [
        "what is gravity",
        "what connects gravity and light",
        "how do neurons and memory relate",
        "what links DNA and evolution",
        "how does energy connect to stars",
    ]
    for q in demos:
        print(f"You: {q}")
        print(f"AI : {ai.chat(q)}")
        print()

    print("--- Direct Multi-hop ---")
    for w1, w2 in [("gravity","light"), ("neuron","memory"),
                   ("dna","evolution"), ("energy","star")]:
        print(f"/hop {w1} {w2}")
        print(ai.multihop(w1, w2))
        print()

    print("-" * 60)

    # Check for train.txt
    for fname in ["train.txt","book.txt","data.txt"]:
        if os.path.exists(fname):
            print(f"[v8] Found {fname} — training...")
            ai.train_file(fname)
            break

    print("Your turn!\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user: continue
        if user.lower() in ('quit','exit','q'): break
        if user == '/think':    print(f"AI : {ai.toggle_verbose()}\n"); continue
        if user == '/history':  print(f"\n{ai.memory.summary()}\n"); continue
        if user == '/forget':   ai.memory.clear(); print("AI : Memory cleared.\n"); continue
        if user == '/stats':    print(f"AI : {ai.stats()}\n"); continue
        if user == '/help':     print(HELP); continue
        if user.startswith('/hop '):
            parts = user[5:].strip().split()
            if len(parts) >= 2:
                print(f"AI : {ai.multihop(parts[0], parts[1])}\n")
            continue
        if user.startswith('/inspect '):
            print(f"AI : {ai.inspect(user[9:].strip())}\n"); continue
        if user.startswith('/train '):
            path = user[7:].strip()
            ai.train_file(path); continue
        print(f"AI : {ai.chat(user)}\n")

    print("\n[Done]")
