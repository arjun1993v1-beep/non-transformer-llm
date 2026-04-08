"""
GF-SDM v7 — Symbolic Composition Engine
=========================================
Genuinely thinks. No transformer. No backprop. No lookup.

Core idea:
  fact_A + fact_B → shared concepts → NEW sentence

Pipeline:
  1. PARSE    — decompose each fact into (subject, relation, object)
  2. INDEX    — build concept → fact map
  3. COMPOSE  — find shared concepts, generate new meaning
  4. SPEAK    — turn composed structure into natural sentence

Pure Python. Zero dependencies. i3-2100 / Termux ready.
"""

import os, re, json, random

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
    'also','more','most','some','such','than','through','each'
}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def keywords(text):
    return [w for w in tokenize(text) if w not in STOP and len(w) > 2]

def keyset(text):
    return set(keywords(text))


# ============================================================
# FACT PARSER  — (subject, relation, objects[])
# ============================================================

RELATION_WORDS = {
    'is','are','was','were','contains','produces','uses',
    'causes','enables','requires','connects','forms','carries',
    'stores','transfers','converts','attracts','releases',
    'made','found','called','known','defined','described'
}

class Fact:
    def __init__(self, raw):
        self.raw      = raw
        self.subject  = None
        self.relation = None
        self.objects  = []
        self.concepts = keyset(raw)
        self._parse()

    def _parse(self):
        words = tokenize(self.raw)
        # Find first relation word
        for i, w in enumerate(words):
            if w in RELATION_WORDS:
                self.subject  = ' '.join(words[:i])
                self.relation = w
                self.objects  = [w for w in words[i+1:]
                                 if w not in STOP and len(w) > 2]
                return
        # Fallback — all keywords as objects
        self.subject  = words[0] if words else ''
        self.relation = 'relates to'
        self.objects  = [w for w in words[1:] if w not in STOP]

    def __repr__(self):
        return f"Fact({self.subject!r} {self.relation!r} {self.objects})"


# ============================================================
# FACT INDEX  — concept → [facts]
# ============================================================

class FactIndex:
    def __init__(self):
        self.facts          = []          # list of Fact
        self.concept_index  = {}          # word → [fact_ids]
        self.subject_index  = {}          # subject → [fact_ids]

    def add(self, raw_answer):
        fid  = len(self.facts)
        fact = Fact(raw_answer)
        self.facts.append(fact)

        for concept in fact.concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(fid)

        subj = fact.subject.strip()
        if subj:
            if subj not in self.subject_index:
                self.subject_index[subj] = []
            self.subject_index[subj].append(fid)

        return fid

    def lookup(self, concept):
        """All facts containing this concept."""
        ids = self.concept_index.get(concept, [])
        return [self.facts[i] for i in ids]

    def lookup_many(self, concepts, top_k=5):
        """Facts scored by how many query concepts they contain."""
        scores = {}
        for c in concepts:
            for fid in self.concept_index.get(c, []):
                scores[fid] = scores.get(fid, 0) + 1
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self.facts[fid] for fid, _ in ranked[:top_k]]


# ============================================================
# COMPOSITION ENGINE  — the core thinking step
# ============================================================

class CompositionEngine:
    """
    Given two or more facts, finds what they share
    and composes a NEW sentence expressing that relationship.

    Example:
      fact_A: gravity is a force that attracts objects with mass
      fact_B: a star is a massive ball of gas that produces energy

      shared : mass, objects, energy
      composed: "Gravity acts on stars because stars have mass,
                 which gravity attracts, releasing energy."
    """

    # Templates for composed sentences
    def compose(self, facts, question_concepts):
        if not facts:
            return None, 0.0
        if len(facts) == 1:
            return self._expand_single(facts[0], question_concepts)

        best_sentence = None
        best_score    = 0.0

        for i in range(len(facts)):
            for j in range(i+1, len(facts)):
                sentence, score = self._compose_pair(
                    facts[i], facts[j], question_concepts)
                if score > best_score:
                    best_score    = score
                    best_sentence = sentence

        if not best_sentence:
            # Return top 2 facts as paragraph
            parts = []
            seen  = set()
            for f in facts[:3]:
                k = frozenset(list(f.concepts)[:3])
                if k not in seen:
                    seen.add(k)
                    parts.append(f.raw[0].upper() + f.raw[1:] + '.')
            return ' '.join(parts), 0.55

        return best_sentence, best_score

    def _compose_pair(self, A, B, q_concepts):
        # Objects only — not subject words
        A_obj = set(A.objects)
        B_obj = set(B.objects)
        shared_obj = (A_obj & B_obj) - STOP

        # Also include question concepts in either fact
        q_in_A = q_concepts & A.concepts - STOP
        q_in_B = q_concepts & B.concepts - STOP
        q_bridge = q_in_A & q_in_B

        shared = (shared_obj | q_bridge) - STOP

        if not shared:
            return None, 0.0

        total = len(A_obj | B_obj)
        score = len(shared) / max(total, 1)

        bridge = sorted(shared, key=lambda w: -len(w))[0]

        templates = [
            f"{A.raw[0].upper() + A.raw[1:]}. {B.raw[0].upper() + B.raw[1:]}. Both involve {bridge}.",
            f"{A.raw[0].upper() + A.raw[1:]}. This connects to {B.subject} because both share {bridge}: {B.raw}.",
            f"Regarding {bridge}: {A.raw}. Also, {B.raw}.",
        ]
        return random.choice(templates), score

    def _expand_single(self, fact, q_concepts):
        extra = sorted((q_concepts - fact.concepts) - STOP, key=lambda w: -len(w))[:3]
        sentence = fact.raw[0].upper() + fact.raw[1:] + '.'
        if extra:
            sentence += f" This relates to {', '.join(extra)}."
        return sentence, 0.6


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
# Q&A KNOWLEDGE BASE
# ============================================================

LEARN_FILE = "gf_sdm_v7_learned.json"

def load_learned():
    if os.path.exists(LEARN_FILE):
        try:
            with open(LEARN_FILE) as f: return json.load(f)
        except: pass
    return []

def save_learned(pairs):
    with open(LEARN_FILE,'w') as f: json.dump(pairs, f, indent=2)

QA_PAIRS = [
    ("what is gravity","gravity is a force that attracts objects with mass toward each other"),
    ("what is the speed of light","light travels at about three hundred thousand kilometers per second"),
    ("what is energy","energy is the ability to do work and comes in many forms like heat light and motion"),
    ("what is an atom","an atom is the smallest unit of an element made of protons neutrons and electrons"),
    ("what is an electron","an electron is a negatively charged particle that orbits the nucleus of an atom"),
    ("what is a proton","a proton is a positively charged particle found in the nucleus of an atom"),
    ("what is matter","matter is anything that has mass and takes up space"),
    ("what is mass","mass is the amount of matter in an object"),
    ("what is force","force is a push or pull that changes the motion of an object"),
    ("what is nuclear energy","nuclear energy is released when the nucleus of an atom is split or fused"),
    ("what is radiation","radiation is energy emitted as particles or electromagnetic waves from unstable atoms"),
    ("what is light","light is electromagnetic radiation that is visible to the human eye"),
    ("what is electricity","electricity is the flow of electric charge through a conductor"),
    ("what is magnetism","magnetism is a force produced by moving electric charges and magnetic materials"),
    ("what is a wave","a wave is a disturbance that transfers energy through space or matter"),
    ("what is heat","heat is thermal energy transferred from a hotter object to a cooler one"),
    ("what is temperature","temperature is a measure of the average kinetic energy of particles in a substance"),
    ("what is sound","sound is a wave of pressure that travels through air and other materials"),
    ("what is a molecule","a molecule is two or more atoms bonded together"),
    ("what is oxygen","oxygen is a gas that makes up about twenty one percent of air and is needed for breathing"),
    ("what is hydrogen","hydrogen is the lightest and most abundant element in the universe"),
    ("what is carbon","carbon is an element that forms the basis of all known life on earth"),
    ("what is water","water is a molecule made of two hydrogen atoms and one oxygen atom"),
    ("what is photosynthesis","photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into sugar and oxygen"),
    ("what is combustion","combustion is a chemical reaction where a substance reacts with oxygen to produce heat and light"),
    ("what is DNA","DNA is a molecule that carries the genetic instructions for the development and function of living organisms"),
    ("what is a cell","a cell is the basic structural and functional unit of all living organisms"),
    ("what is evolution","evolution is the process by which species change over time through natural selection"),
    ("what is a gene","a gene is a segment of DNA that carries instructions for making a specific protein"),
    ("what is a protein","a protein is a large molecule made of amino acids that performs many functions in the body"),
    ("what is an enzyme","an enzyme is a protein that speeds up chemical reactions in living organisms"),
    ("what is a neuron","a neuron is a nerve cell that transmits electrical signals through the nervous system"),
    ("what is a synapse","a synapse is the junction between two neurons where signals are transmitted"),
    ("what is the brain","the brain is the organ that controls all body functions and is the center of thought and emotion"),
    ("what is the nervous system","the nervous system sends signals throughout the body using the brain spinal cord and nerves"),
    ("what is the immune system","the immune system is the body's defense against pathogens and foreign substances"),
    ("what is metabolism","metabolism is all the chemical reactions that occur in a living organism to maintain life"),
    ("what is a hormone","a hormone is a chemical messenger produced by glands that regulates body functions"),
    ("what is consciousness","consciousness is the state of being aware of one's own thoughts feelings and surroundings"),
    ("what is memory","memory is the ability of the brain to store retain and recall information and experiences"),
    ("what is intelligence","intelligence is the ability to learn reason solve problems and adapt to new situations"),
    ("what is learning","learning is the process of acquiring new knowledge skills or behaviors through experience"),
    ("what is reasoning","reasoning is the process of thinking logically to form conclusions from evidence"),
    ("what is perception","perception is the process by which the brain interprets sensory information from the environment"),
    ("what is emotion","emotion is a mental state that arises from thoughts experiences and physiological changes"),
    ("what is imagination","imagination is the ability to form new ideas or images not present to the senses"),
    ("what is language","language is a system of communication using sounds symbols or gestures to convey meaning"),
    ("what is thought","thought is a mental process that involves reasoning imagination and problem solving"),
    ("what is knowledge","knowledge is justified true belief acquired through experience or education"),
    ("what is creativity","creativity is the ability to generate original ideas solutions or artistic works"),
    ("what is the hippocampus","the hippocampus is a brain region crucial for forming and storing new memories"),
    ("what is sleep","sleep is a natural state of rest where the body and brain recover and consolidate memories"),
    ("what is mathematics","mathematics is the study of numbers shapes patterns and logical relationships"),
    ("what is an algorithm","an algorithm is a step by step procedure for solving a problem or completing a task"),
    ("what is logic","logic is the study of valid reasoning and the principles of correct inference"),
    ("what is probability","probability is the measure of how likely an event is to occur between zero and one"),
    ("what is pi","pi is the ratio of a circle's circumference to its diameter approximately equal to three point fourteen"),
    ("what is calculus","calculus is the branch of mathematics that studies rates of change and accumulation"),
    ("what is artificial intelligence","artificial intelligence is the simulation of human intelligence processes by computer systems"),
    ("what is machine learning","machine learning is a type of AI where systems learn from data to improve their performance"),
    ("what is a neural network","a neural network is a computing system inspired by the human brain made of connected nodes"),
    ("what is deep learning","deep learning is machine learning using neural networks with many layers"),
    ("what is a language model","a language model is an AI system trained to understand and generate human language"),
    ("what is a computer","a computer is an electronic device that processes and stores information"),
    ("what is software","software is a set of instructions that tells a computer what to do"),
    ("what is the internet","the internet is a global network of computers connected together to share information"),
    ("what is binary","binary is a number system that uses only two digits zero and one to represent all data"),
    ("what is a transistor","a transistor is a semiconductor device used to amplify or switch electronic signals"),
    ("what is the universe","the universe is all of space time matter and energy that exists"),
    ("what is the big bang","the big bang is the theory that the universe began from an extremely hot dense state about fourteen billion years ago"),
    ("what is a star","a star is a massive ball of gas that produces energy through nuclear fusion"),
    ("what is the sun","the sun is the star at the center of our solar system that provides light and heat to earth"),
    ("what is a black hole","a black hole is a region of space where gravity is so strong that nothing can escape it"),
    ("what is a galaxy","a galaxy is a large system of stars gas dust and dark matter held together by gravity"),
    ("what is the milky way","the milky way is the galaxy that contains our solar system"),
    ("what is dark matter","dark matter is an invisible substance that makes up most of the universe's mass"),
    ("what is a supernova","a supernova is the explosive death of a massive star that releases enormous energy"),
    ("what is orbit","orbit is the curved path an object takes around another object due to gravity"),
    ("what is a planet","a planet is a large body that orbits a star and has cleared its orbital path"),
    ("what is earth","earth is the third planet from the sun and the only known planet to support life"),
    ("what is the moon","the moon is earth's only natural satellite that orbits our planet"),
    ("what is space","space is the vast empty region beyond earth's atmosphere where stars planets and galaxies exist"),
    ("what is time","time is the continuous progression of existence from the past through the present to the future"),
    ("what is the atmosphere","the atmosphere is the layer of gases surrounding earth that protects life and regulates climate"),
    ("what is the water cycle","the water cycle is the continuous movement of water through evaporation precipitation and collection"),
    ("what is climate","climate is the average weather conditions of a region over a long period of time"),
    ("what is an earthquake","an earthquake is a sudden shaking of the ground caused by movement of tectonic plates"),
    ("what is a volcano","a volcano is an opening in earth's crust through which molten rock gas and ash erupt"),
    ("what is philosophy","philosophy is the study of fundamental questions about existence knowledge ethics and reality"),
    ("what is consciousness in philosophy","consciousness in philosophy refers to the hard problem of why physical processes give rise to subjective experience"),
    ("what is free will","free will is the ability to make choices that are not determined by prior causes"),
    ("what is truth","truth is the property of beliefs or statements that accurately correspond to reality"),
    ("what is ethics","ethics is the branch of philosophy concerned with moral principles and right conduct"),
    ("what is science","science is the systematic study of the natural world through observation and experiment"),
    ("what is technology","technology is the application of scientific knowledge to solve practical problems"),
    ("what is education","education is the process of acquiring knowledge skills values and habits"),
    ("what is culture","culture is the shared beliefs values customs and behaviors of a group of people"),
    ("what is communication","communication is the exchange of information ideas or feelings between individuals"),
    ("what is biodiversity","biodiversity is the variety of life found in a particular habitat or on earth as a whole"),
    ("what is an ecosystem","an ecosystem is a community of living organisms interacting with their environment"),
    ("what is natural selection","natural selection is the process where organisms with favorable traits survive and reproduce more"),
    ("what is nuclear fusion","nuclear fusion is the process where two atomic nuclei combine to form a heavier nucleus releasing enormous energy"),
    ("what is a light year","a light year is the distance light travels in one year about nine point five trillion kilometers"),
]


# ============================================================
# MAIN AI  — v7
# ============================================================

class GFSDMv7:
    def __init__(self):
        self.qa_pairs  = list(QA_PAIRS)
        self.index     = FactIndex()
        self.composer  = CompositionEngine()
        self.memory    = Memory(max_turns=5)
        self.verbose   = False

        # Load learned
        learned = load_learned()
        if learned:
            self.qa_pairs.extend([(q,a) for q,a in learned])
            print(f"[v7] Loaded {len(learned)} learned facts.")

        # Build index
        print(f"[v7] Indexing {len(self.qa_pairs)} facts...")
        for _, answer in self.qa_pairs:
            self.index.add(answer)
        print(f"[v7] Index: {len(self.index.concept_index)} concepts.")

    # ── CORE: think about a question ──────────────────────────

    def think(self, question):
        """
        Full thinking pipeline:
        1. Extract concepts from question
        2. Retrieve relevant facts
        3. COMPOSE — generate new sentence from shared concepts
        4. Return answer + confidence
        """
        q_concepts = set(keywords(question))

        if self.verbose:
            print(f"\n  [think] concepts: {', '.join(sorted(q_concepts))}")

        # Retrieve top facts
        facts = self.index.lookup_many(q_concepts, top_k=5)

        if self.verbose:
            print(f"  [think] retrieved {len(facts)} facts")
            for f in facts[:3]:
                print(f"    · {f.raw[:55]}...")

        # Compose
        answer, score = self.composer.compose(facts, q_concepts)

        if self.verbose:
            print(f"  [think] score: {score:.2f}")
            print(f"  [think] composed: {answer[:60] if answer else 'none'}...\n")

        return answer, score

    # ── CHAT ─────────────────────────────────────────────────

    def chat(self, user_input):
        user_input = user_input.strip()
        if not user_input:
            return "I'm listening."

        # Detect teaching
        teach = self._detect_teach(user_input)
        if teach:
            q, a = teach
            existing = [eq for eq,_ in self.qa_pairs]
            if q not in existing:
                self.qa_pairs.append((q, a))
                self.index.add(a)
                learned = load_learned()
                learned.append([q, a])
                save_learned(learned)
                response = f"Understood. I'll remember: {a}."
            else:
                response = f"I already know that."
            self.memory.add(user_input, response)
            return response

        # Enrich with context
        enriched = self.memory.enrich(user_input)

        # Think
        answer, score = self.think(enriched)

        if not answer or score < 0.05:
            last = self.memory.last_topic()
            response = (f"I don't have enough to answer that. "
                       + (f"We were on {last} — want to continue?" if last
                          else "Try teaching me something new."))
        else:
            pct = int(min(score, 0.99) * 100)
            label = self._label(score)
            response = f"({pct}% — {label}) {answer}"

        self.memory.add(user_input, response)
        return response

    def _label(self, s):
        if s >= 0.8: return "very confident"
        if s >= 0.6: return "confident"
        if s >= 0.4: return "fairly sure"
        if s >= 0.2: return "not fully sure"
        return "inferring"

    def _detect_teach(self, text):
        orig = text.strip().rstrip('.')
        if re.match(r'^(what|who|how|why|when|where|which|is|are|does|do)\b',
                    orig.lower()):
            return None
        t = orig
        for p in ['remember that ','learn that ','note that ','learn: ','fact: ']:
            if t.lower().startswith(p): t = t[len(p):]; break
        m = re.match(r'^(.+?)\s+(?:is|are)\s+(.+)$', t, re.IGNORECASE)
        if m:
            subj = m.group(1).strip().lower()
            defn = m.group(2).strip().lower()
            if len(subj.split()) <= 5 and len(defn.split()) >= 3:
                return f"what is {subj}", f"{subj} is {defn}"
        return None

    def toggle_verbose(self):
        self.verbose = not self.verbose
        return f"Thinking steps {'ON' if self.verbose else 'OFF'}."

    def inspect(self, word):
        """Show all facts containing a concept."""
        facts = self.index.lookup(word.lower())
        if not facts:
            return f"No facts found for '{word}'."
        lines = [f"Facts about '{word}':"]
        for f in facts[:5]:
            lines.append(f"  · {f.raw}")
        return '\n'.join(lines)

    def compose_two(self, w1, w2):
        """Manually compose two concepts — shows the engine working."""
        f1 = self.index.lookup_many({w1}, top_k=1)
        f2 = self.index.lookup_many({w2}, top_k=1)
        if not f1 or not f2:
            return f"Not enough facts for '{w1}' or '{w2}'."
        sentence, score = self.composer._compose_pair(
            f1[0], f2[0], {w1, w2})
        if not sentence:
            return f"No shared concepts between '{w1}' and '{w2}'."
        return f"({int(score*100)}%) {sentence}"

    def stats(self):
        base    = len(QA_PAIRS)
        learned = len(self.qa_pairs) - base
        return (f"Facts: {len(self.qa_pairs)} ({base} built-in + {learned} learned) | "
                f"Concepts: {len(self.index.concept_index)}")


# ============================================================
# ENTRY POINT
# ============================================================

HELP = """
Commands:
  /think          — toggle thinking steps ON/OFF
  /history        — recent conversation
  /forget         — clear memory
  /stats          — knowledge stats
  /inspect <word> — show all facts containing a concept
  /compose <w1> <w2> — manually compose two concepts
  /help           — this help
  quit            — exit

Teaching:
  "ARIA is an AI companion with emotional memory built by Arjun"
  "Qwen3 is a large language model by Alibaba"
  "remember that Kerala is a coastal state in southern India"
"""

if __name__ == "__main__":
    print("=" * 58)
    print("  GF-SDM v7  |  Symbolic Composition Engine")
    print("  Think without Transformer | Pure Python | No BP")
    print("=" * 58)

    ai = GFSDMv7()
    print(f"\n  Type /help for commands.\n")

    print("--- Demo: Composed Thinking ---\n")
    demos = [
        "what is gravity",
        "what connects gravity and stars",
        "how do neurons and memory relate",
        "what is the link between energy and light",
        "how does DNA connect to life",
        "what connects black holes and gravity and mass",
    ]
    for q in demos:
        print(f"You: {q}")
        print(f"AI : {ai.chat(q)}")
        print()

    print("--- Compose Engine Direct ---")
    print(ai.compose_two("gravity", "star"))
    print(ai.compose_two("neuron", "memory"))
    print(ai.compose_two("energy", "light"))
    print()

    print("-" * 58)
    print("Your turn!\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user: continue
        if user.lower() in ('quit','exit','q'): break
        if user == '/think':   print(f"AI : {ai.toggle_verbose()}\n"); continue
        if user == '/history': print(f"\n{ai.memory.summary()}\n"); continue
        if user == '/forget':  print(f"AI : {ai.memory.clear() or 'Memory cleared.'}\n"); continue
        if user == '/stats':   print(f"AI : {ai.stats()}\n"); continue
        if user == '/help':    print(HELP); continue
        if user.startswith('/inspect '):
            print(f"AI : {ai.inspect(user[9:].strip())}\n"); continue
        if user.startswith('/compose '):
            parts = user[9:].strip().split()
            if len(parts) >= 2:
                print(f"AI : {ai.compose_two(parts[0], parts[1])}\n")
            continue
        print(f"AI : {ai.chat(user)}\n")

    print("\n[Done]")
