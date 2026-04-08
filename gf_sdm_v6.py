"""
GF-SDM v6
=========
Conversational Factual AI — Think + Learn + Talk + Reason

New in v6:
- COMBINE: forms NEW answers from existing facts for unknown questions
- ASK BACK: model asks clarifying counter-questions when unsure
- CONFIDENCE: every answer has a confidence score shown to user
- TOPIC GRAPH: maps relationships between concepts, finds inference paths
- All v5 features retained

Pure Python. Zero dependencies. i3-2100 / Termux ready.
"""

import os
import re
import json
import time

# ============================================================
# TOKENIZER
# ============================================================

STOPWORDS = {
    'what','is','are','a','an','the','of','in','does','do',
    'how','why','when','where','who','which','was','were',
    'has','have','had','its','it','be','been','can','could',
    'and','or','but','so','if','then','that','this','these',
    'those','to','for','on','at','by','from','with','about'
}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+|[0-9]+", text.lower())

def key_tokens(text):
    return set(tokenize(text)) - STOPWORDS


# ============================================================
# KNOWLEDGE BASE
# ============================================================

LEARN_FILE = "gf_sdm_v5_learned.json"

def load_learned():
    if os.path.exists(LEARN_FILE):
        try:
            with open(LEARN_FILE) as f:
                return json.load(f)
        except:
            pass
    return []

def save_learned(pairs):
    with open(LEARN_FILE, 'w') as f:
        json.dump(pairs, f, indent=2)


# ============================================================
# Q&A DATASET
# ============================================================

QA_PAIRS = [
    # Physics
    ("what is the speed of light", "light travels at about three hundred thousand kilometers per second"),
    ("what is gravity", "gravity is a force that attracts objects with mass toward each other"),
    ("what is energy", "energy is the ability to do work and comes in many forms like heat light and motion"),
    ("what is an atom", "an atom is the smallest unit of an element made of protons neutrons and electrons"),
    ("what is an electron", "an electron is a negatively charged particle that orbits the nucleus of an atom"),
    ("what is a proton", "a proton is a positively charged particle found in the nucleus of an atom"),
    ("what is matter", "matter is anything that has mass and takes up space"),
    ("what is mass", "mass is the amount of matter in an object"),
    ("what is force", "force is a push or pull that changes the motion of an object"),
    ("what is acceleration", "acceleration is the rate at which velocity changes over time"),
    ("what is momentum", "momentum is the product of an object's mass and velocity"),
    ("what is temperature", "temperature is a measure of the average kinetic energy of particles in a substance"),
    ("what is heat", "heat is thermal energy transferred from a hotter object to a cooler one"),
    ("what is sound", "sound is a wave of pressure that travels through air and other materials"),
    ("what is light", "light is electromagnetic radiation that is visible to the human eye"),
    ("what is electricity", "electricity is the flow of electric charge through a conductor"),
    ("what is magnetism", "magnetism is a force produced by moving electric charges and magnetic materials"),
    ("what is a wave", "a wave is a disturbance that transfers energy through space or matter"),
    ("what is frequency", "frequency is the number of waves that pass a point per second"),
    ("what is nuclear energy", "nuclear energy is released when the nucleus of an atom is split or fused"),
    ("what is radiation", "radiation is energy emitted as particles or electromagnetic waves from unstable atoms"),
    ("what is pressure", "pressure is the force applied per unit area on a surface"),
    ("what is velocity", "velocity is the speed of an object in a specific direction"),
    # Chemistry
    ("what is a molecule", "a molecule is two or more atoms bonded together"),
    ("what is oxygen", "oxygen is a gas that makes up about twenty one percent of air and is needed for breathing"),
    ("what is hydrogen", "hydrogen is the lightest and most abundant element in the universe"),
    ("what is carbon", "carbon is an element that forms the basis of all known life on earth"),
    ("what is water", "water is a molecule made of two hydrogen atoms and one oxygen atom"),
    ("what is an acid", "an acid is a substance that donates hydrogen ions and has a pH below seven"),
    ("what is a base", "a base is a substance that accepts hydrogen ions and has a pH above seven"),
    ("what is a chemical reaction", "a chemical reaction is a process where substances are transformed into new substances"),
    ("what is photosynthesis", "photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into sugar and oxygen"),
    ("what is combustion", "combustion is a chemical reaction where a substance reacts with oxygen to produce heat and light"),
    ("what is DNA", "DNA is a molecule that carries the genetic instructions for the development and function of living organisms"),
    ("what is carbon dioxide", "carbon dioxide is a gas produced by burning and breathing that plants use in photosynthesis"),
    # Biology
    ("what is a cell", "a cell is the basic structural and functional unit of all living organisms"),
    ("what is evolution", "evolution is the process by which species change over time through natural selection"),
    ("what is natural selection", "natural selection is the process where organisms with favorable traits survive and reproduce more"),
    ("what is a gene", "a gene is a segment of DNA that carries instructions for making a specific protein"),
    ("what is a protein", "a protein is a large molecule made of amino acids that performs many functions in the body"),
    ("what is an enzyme", "an enzyme is a protein that speeds up chemical reactions in living organisms"),
    ("what is a virus", "a virus is a tiny infectious agent that can only replicate inside a living cell"),
    ("what is the immune system", "the immune system is the body's defense against pathogens and foreign substances"),
    ("what is an ecosystem", "an ecosystem is a community of living organisms interacting with their environment"),
    ("what is biodiversity", "biodiversity is the variety of life found in a particular habitat or on earth as a whole"),
    ("what is metabolism", "metabolism is all the chemical reactions that occur in a living organism to maintain life"),
    ("what is a hormone", "a hormone is a chemical messenger produced by glands that regulates body functions"),
    ("what is the nervous system", "the nervous system sends signals throughout the body using the brain spinal cord and nerves"),
    ("what is a neuron", "a neuron is a nerve cell that transmits electrical signals through the nervous system"),
    ("what is a synapse", "a synapse is the junction between two neurons where signals are transmitted"),
    ("what is respiration", "respiration is the process by which cells break down sugar to release energy"),
    # Mathematics
    ("what is mathematics", "mathematics is the study of numbers shapes patterns and logical relationships"),
    ("what is a prime number", "a prime number is a number greater than one that can only be divided by one and itself"),
    ("what is algebra", "algebra is a branch of mathematics that uses symbols to represent unknown values"),
    ("what is geometry", "geometry is the branch of mathematics that studies shapes sizes and properties of figures"),
    ("what is pi", "pi is the ratio of a circle's circumference to its diameter approximately equal to three point fourteen"),
    ("what is calculus", "calculus is the branch of mathematics that studies rates of change and accumulation"),
    ("what is probability", "probability is the measure of how likely an event is to occur between zero and one"),
    ("what is statistics", "statistics is the science of collecting analyzing and interpreting numerical data"),
    ("what is an algorithm", "an algorithm is a step by step procedure for solving a problem or completing a task"),
    ("what is infinity", "infinity is a concept representing a quantity that is larger than any real number"),
    ("what is a logarithm", "a logarithm is the power to which a base must be raised to produce a given number"),
    ("what is a matrix", "a matrix is a rectangular array of numbers arranged in rows and columns"),
    ("what is a vector", "a vector is a quantity that has both magnitude and direction"),
    ("what is logic", "logic is the study of valid reasoning and the principles of correct inference"),
    # Computing & AI
    ("what is a computer", "a computer is an electronic device that processes and stores information"),
    ("what is software", "software is a set of instructions that tells a computer what to do"),
    ("what is hardware", "hardware is the physical components that make up a computer system"),
    ("what is the internet", "the internet is a global network of computers connected together to share information"),
    ("what is artificial intelligence", "artificial intelligence is the simulation of human intelligence processes by computer systems"),
    ("what is machine learning", "machine learning is a type of AI where systems learn from data to improve their performance"),
    ("what is a neural network", "a neural network is a computing system inspired by the human brain made of connected nodes"),
    ("what is deep learning", "deep learning is machine learning using neural networks with many layers"),
    ("what is a language model", "a language model is an AI system trained to understand and generate human language"),
    ("what is natural language processing", "natural language processing is the field of AI that deals with human language"),
    ("what is a database", "a database is an organized collection of structured information stored electronically"),
    ("what is memory in computers", "computer memory stores data and instructions that the processor needs to access quickly"),
    ("what is a CPU", "a CPU is the central processing unit that executes instructions in a computer"),
    ("what is binary", "binary is a number system that uses only two digits zero and one to represent all data"),
    ("what is encryption", "encryption is the process of converting information into a code to prevent unauthorized access"),
    ("what is a transistor", "a transistor is a semiconductor device used to amplify or switch electronic signals"),
    # Mind
    ("what is consciousness", "consciousness is the state of being aware of one's own thoughts feelings and surroundings"),
    ("what is memory", "memory is the ability of the brain to store retain and recall information and experiences"),
    ("what is learning", "learning is the process of acquiring new knowledge skills or behaviors through experience"),
    ("what is intelligence", "intelligence is the ability to learn reason solve problems and adapt to new situations"),
    ("what is emotion", "emotion is a mental state that arises from thoughts experiences and physiological changes"),
    ("what is attention", "attention is the cognitive process of selectively concentrating on one thing over others"),
    ("what is perception", "perception is the process by which the brain interprets sensory information from the environment"),
    ("what is imagination", "imagination is the ability to form new ideas or images not present to the senses"),
    ("what is reasoning", "reasoning is the process of thinking logically to form conclusions from evidence"),
    ("what is the brain", "the brain is the organ that controls all body functions and is the center of thought and emotion"),
    ("what is the hippocampus", "the hippocampus is a brain region crucial for forming and storing new memories"),
    ("what is sleep", "sleep is a natural state of rest where the body and brain recover and consolidate memories"),
    ("what is a dream", "a dream is a sequence of thoughts images and sensations that occur during sleep"),
    ("what is language", "language is a system of communication using sounds symbols or gestures to convey meaning"),
    ("what is thought", "thought is a mental process that involves reasoning imagination and problem solving"),
    ("what is knowledge", "knowledge is justified true belief acquired through experience or education"),
    ("what is creativity", "creativity is the ability to generate original ideas solutions or artistic works"),
    ("what is intuition", "intuition is the ability to understand something immediately without conscious reasoning"),
    ("what is motivation", "motivation is the internal drive that initiates and sustains goal directed behavior"),
    # Universe
    ("what is the universe", "the universe is all of space time matter and energy that exists"),
    ("what is the big bang", "the big bang is the theory that the universe began from an extremely hot dense state about fourteen billion years ago"),
    ("what is a galaxy", "a galaxy is a large system of stars gas dust and dark matter held together by gravity"),
    ("what is the milky way", "the milky way is the galaxy that contains our solar system"),
    ("what is a star", "a star is a massive ball of gas that produces energy through nuclear fusion"),
    ("what is the sun", "the sun is the star at the center of our solar system that provides light and heat to earth"),
    ("what is a planet", "a planet is a large body that orbits a star and has cleared its orbital path"),
    ("what is earth", "earth is the third planet from the sun and the only known planet to support life"),
    ("what is the moon", "the moon is earth's only natural satellite that orbits our planet"),
    ("what is a black hole", "a black hole is a region of space where gravity is so strong that nothing can escape it"),
    ("what is dark matter", "dark matter is an invisible substance that makes up most of the universe's mass"),
    ("what is a supernova", "a supernova is the explosive death of a massive star that releases enormous energy"),
    ("what is a light year", "a light year is the distance light travels in one year about nine point five trillion kilometers"),
    ("what is orbit", "orbit is the curved path an object takes around another object due to gravity"),
    ("what is the solar system", "the solar system consists of the sun and all the objects that orbit it including eight planets"),
    ("what is space", "space is the vast empty region beyond earth's atmosphere where stars planets and galaxies exist"),
    ("what is time", "time is the continuous progression of existence from the past through the present to the future"),
    # Earth
    ("what is the atmosphere", "the atmosphere is the layer of gases surrounding earth that protects life and regulates climate"),
    ("what is climate", "climate is the average weather conditions of a region over a long period of time"),
    ("what is weather", "weather is the short term state of the atmosphere including temperature wind and precipitation"),
    ("what is an earthquake", "an earthquake is a sudden shaking of the ground caused by movement of tectonic plates"),
    ("what is a volcano", "a volcano is an opening in earth's crust through which molten rock gas and ash erupt"),
    ("what is erosion", "erosion is the wearing away of rock and soil by wind water and other natural forces"),
    ("what is the water cycle", "the water cycle is the continuous movement of water through evaporation precipitation and collection"),
    ("what is soil", "soil is the top layer of earth made of minerals organic matter water and air that supports plant growth"),
    ("what is a fossil", "a fossil is the preserved remains or trace of an ancient organism found in rock"),
    ("what is an ocean", "an ocean is a vast body of salt water that covers most of earth's surface"),
    ("what is a forest", "a forest is a large area of land covered with trees and other plants"),
    # Philosophy
    ("what is philosophy", "philosophy is the study of fundamental questions about existence knowledge ethics and reality"),
    ("what is truth", "truth is the property of beliefs or statements that accurately correspond to reality"),
    ("what is reality", "reality is everything that exists whether or not it can be observed or measured"),
    ("what is ethics", "ethics is the branch of philosophy concerned with moral principles and right conduct"),
    ("what is free will", "free will is the ability to make choices that are not determined by prior causes"),
    ("what is identity", "identity is the set of characteristics that make a person or thing distinct from others"),
    ("what is meaning", "meaning is the significance or purpose that people find in their experiences and existence"),
    ("what is beauty", "beauty is the quality of things that gives pleasure or satisfaction to the mind or senses"),
    ("what is justice", "justice is the quality of being fair and reasonable in treating people and situations"),
    ("what is morality", "morality is a system of beliefs about what is right and wrong in human conduct"),
    # Society
    ("what is science", "science is the systematic study of the natural world through observation and experiment"),
    ("what is technology", "technology is the application of scientific knowledge to solve practical problems"),
    ("what is education", "education is the process of acquiring knowledge skills values and habits"),
    ("what is history", "history is the study of past events particularly in human affairs"),
    ("what is culture", "culture is the shared beliefs values customs and behaviors of a group of people"),
    ("what is democracy", "democracy is a system of government where citizens vote to elect their leaders"),
    ("what is economics", "economics is the study of how people produce distribute and consume goods and services"),
    ("what is medicine", "medicine is the science and practice of diagnosing treating and preventing disease"),
    ("what is agriculture", "agriculture is the practice of cultivating land to grow crops and raise animals for food"),
    ("what is art", "art is the expression of human creativity and imagination in visual auditory or performance forms"),
    ("what is music", "music is the art of arranging sounds in time to create harmony melody and rhythm"),
    ("what is communication", "communication is the exchange of information ideas or feelings between individuals"),
    ("what is writing", "writing is the representation of language through symbols to communicate ideas across time"),
]


# ============================================================
# THINK ENGINE
# ============================================================

def find_best_match(question, qa_pairs, min_score=0.25):
    q_clean = key_tokens(question)
    if not q_clean:
        return None, 0.0
    best_score  = 0.0
    best_answer = None
    for q, a in qa_pairs:
        t = key_tokens(q)
        if not t: continue
        overlap   = len(q_clean & t)
        precision = overlap / len(q_clean)
        recall    = overlap / len(t)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best_score:
            best_score  = f1
            best_answer = a
    return (best_answer, best_score) if best_score >= min_score else (None, 0.0)


def find_related(question, qa_pairs, main_answer, max_related=2):
    q_clean = key_tokens(question)
    # Only use question keywords, NOT answer keywords — avoids over-broad matches
    scored = []
    for q, a in qa_pairs:
        if a == main_answer: continue
        score = len(q_clean & key_tokens(q))
        if score >= 2:  # require at least 2 keyword overlaps
            scored.append((score, a))
    scored.sort(key=lambda x: -x[0])
    return [a for _, a in scored[:max_related]]


def think(question, qa_pairs, show_thinking=False):
    concepts = list(key_tokens(question))

    if show_thinking:
        print(f"\n  [thinking]")
        print(f"  concepts : {', '.join(concepts[:6])}")

    main_answer, score = find_best_match(question, qa_pairs)

    if show_thinking:
        print(f"  found    : {main_answer[:55] + '...' if main_answer else 'none'}")

    related = find_related(question, qa_pairs, main_answer, max_related=2)

    if show_thinking:
        print(f"  related  : {len(related)} facts")
        print(f"  [/thinking]\n")

    parts = []
    if main_answer:
        parts.append(main_answer[0].upper() + main_answer[1:] + '.')
    for rel in related:
        parts.append(rel[0].upper() + rel[1:] + '.')

    return ' '.join(parts) if parts else None


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history   = []
        self.max_turns = max_turns
        self.topics    = []

    def add(self, user_msg, bot_msg):
        self.history.append((user_msg, bot_msg))
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        self.topics = (self.topics + list(key_tokens(user_msg)))[-20:]

    def enrich(self, question):
        """Add context only for very short follow-ups like 'why', 'how'."""
        words = question.lower().split()
        # Only enrich single-word follow-ups
        if len(words) == 1 and self.topics:
            return question + ' ' + self.topics[-1]
        return question

    def last_topic(self):
        return self.topics[-1] if self.topics else None

    def summary(self):
        if not self.history:
            return "No conversation yet."
        lines = []
        for u, b in self.history[-3:]:
            lines.append(f"  You : {u}")
            lines.append(f"  AI  : {b[:70]}...")
        return '\n'.join(lines)

    def clear(self):
        self.history = []
        self.topics  = []



# ============================================================
# TOPIC GRAPH
# ============================================================

class TopicGraph:
    def __init__(self):
        self.edges = {}

    def build(self, qa_pairs):
        for q, a in qa_pairs:
            words = list(key_tokens(q) | key_tokens(a))
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    self._add(w1, w2)
                    self._add(w2, w1)

    def _add(self, a, b):
        if a not in self.edges: self.edges[a] = {}
        self.edges[a][b] = self.edges[a].get(b, 0) + 1

    def neighbors(self, word, top_k=5):
        if word not in self.edges: return []
        return sorted(self.edges[word].items(), key=lambda x: -x[1])[:top_k]

    def related_concepts(self, words, top_k=6):
        scores = {}
        for w in words:
            for nb, wt in self.neighbors(w, top_k=10):
                scores[nb] = scores.get(nb, 0) + wt
        for w in words: scores.pop(w, None)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    def path(self, src, dst, max_hops=3):
        if src == dst: return [src]
        visited = {src}
        queue   = [[src]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if len(path) > max_hops: continue
            for nb, _ in self.neighbors(node, top_k=8):
                if nb == dst: return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        return []


# ============================================================
# COMBINE ENGINE
# ============================================================

def combine_facts(question, qa_pairs, graph, max_facts=3):
    q_concepts = key_tokens(question)
    expanded   = set(q_concepts)
    for nb, _ in graph.related_concepts(q_concepts, top_k=6):
        expanded.add(nb)

    scored = []
    for q, a in qa_pairs:
        t     = key_tokens(q) | key_tokens(a)
        score = len(expanded & t)
        if score >= 1:
            scored.append((score, a))

    scored.sort(key=lambda x: -x[0])
    top_facts = [a for _, a in scored[:max_facts]]
    if not top_facts:
        return None, 0.0

    coverage = len(q_concepts & expanded) / max(len(q_concepts), 1)
    parts = []
    seen  = set()
    for fact in top_facts:
        key = frozenset(list(key_tokens(fact))[:4])
        if key not in seen:
            seen.add(key)
            parts.append(fact[0].upper() + fact[1:] + '.')

    return ' '.join(parts), min(coverage * 0.6, 0.55)


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_label(score):
    if score >= 0.85: return "very confident"
    if score >= 0.65: return "confident"
    if score >= 0.45: return "fairly sure"
    if score >= 0.25: return "not fully sure"
    return "guessing"


# ============================================================
# ASK BACK
# ============================================================

ASK_TEMPLATES = [
    "I found something related to {topic}. Are you asking about {topic} specifically?",
    "That touches on {topic}. Do you want to go deeper into {topic}?",
    "I know about {topic} which seems related. Is that what you mean?",
    "Are you asking about {topic}? Tell me more and I can help better.",
]

def form_counter_question(question, graph):
    import random
    q_words = key_tokens(question)
    related = graph.related_concepts(q_words, top_k=3)
    if not related:
        return "Could you be more specific? What aspect are you asking about?"
    topic = related[0][0]
    return random.choice(ASK_TEMPLATES).format(topic=topic)


# ============================================================
# MAIN AI
# ============================================================

LEARN_FILE_V6 = "gf_sdm_v6_learned.json"

def load_learned_v6():
    if os.path.exists(LEARN_FILE_V6):
        try:
            with open(LEARN_FILE_V6) as f: return json.load(f)
        except: pass
    return []

def save_learned_v6(pairs):
    with open(LEARN_FILE_V6, 'w') as f: json.dump(pairs, f, indent=2)


class GFSDMv6:
    def __init__(self):
        self.qa_pairs      = list(QA_PAIRS)
        self.memory        = ConversationMemory(max_turns=5)
        self.show_thinking = False
        self.graph         = TopicGraph()

        learned = load_learned_v6()
        if learned:
            self.qa_pairs.extend([(q, a) for q, a in learned])
            print(f"[v6] Loaded {len(learned)} learned facts.")

        print(f"[v6] Building topic graph...")
        self.graph.build(self.qa_pairs)
        print(f"[v6] Graph: {len(self.graph.edges)} concept nodes.")

    def chat(self, user_input):
        user_input = user_input.strip()
        if not user_input:
            return "I'm listening."

        # ── Detect teaching
        teach = self._detect_teach(user_input)
        if teach:
            q, a = teach
            existing = [eq for eq, _ in self.qa_pairs]
            if q not in existing:
                self.qa_pairs.append((q, a))
                # Update graph with new fact
                self.graph.build([(q, a)])
                learned = load_learned_v6()
                learned.append([q, a])
                save_learned_v6(learned)
                response = f"Got it! I'll remember that {a}."
            else:
                response = f"I already know that. {a[0].upper() + a[1:]}."
            self.memory.add(user_input, response)
            return response

        # ── Enrich with context
        enriched = self.memory.enrich(user_input)

        # ── Step 1: Try direct match
        main_answer, score = find_best_match(enriched, self.qa_pairs, min_score=0.25)

        if main_answer and score >= 0.25:
            # Build paragraph from main + related
            related  = find_related(enriched, self.qa_pairs, main_answer, max_related=2)
            parts    = [main_answer[0].upper() + main_answer[1:] + '.']
            parts   += [r[0].upper() + r[1:] + '.' for r in related]
            answer   = ' '.join(parts)
            confidence = min(score, 0.98)

        # ── Step 2: Try combining facts (inference)
        else:
            answer, confidence = combine_facts(enriched, self.qa_pairs, self.graph)

            # ── Step 3: Ask back if still unsure
            if not answer or confidence < 0.15:
                if should_ask_back(confidence, enriched):
                    response = form_counter_question(enriched, self.graph)
                    self.memory.add(user_input, response)
                    return response
                last = self.memory.last_topic()
                response = (f"I don't have enough information on that. "
                            + (f"We were discussing {last} — want to continue?" if last
                               else "Try teaching me: say 'X is Y'."))
                self.memory.add(user_input, response)
                return response

        if self.show_thinking:
            label = confidence_label(confidence)
            pct   = int(confidence * 100)
            response = f"[{pct}% — {label}]\n{answer}"
        else:
            pct   = int(confidence * 100)
            response = f"({pct}%) {answer}"

        self.memory.add(user_input, response)
        return response

    def graph_explore(self, word):
        """Show what the model knows about a concept."""
        neighbors = self.graph.neighbors(word.lower(), top_k=8)
        if not neighbors:
            return f"No graph data for '{word}'."
        related = ', '.join([f"{w}({s})" for w, s in neighbors])
        return f"Concepts related to '{word}': {related}"

    def path_between(self, w1, w2):
        """Show inference path between two concepts."""
        p = self.graph.path(w1.lower(), w2.lower(), max_hops=4)
        if not p:
            return f"No connection found between '{w1}' and '{w2}'."
        return f"Path: {' → '.join(p)}"

    def _detect_teach(self, text):
        original = text.strip().rstrip('.')
        if re.match(r'^(what|who|how|why|when|where|which|is|are|does|do)\b',
                    original.lower()):
            return None
        text = original
        for prefix in ['remember that ', 'learn that ', 'note that ',
                       'learn: ', 'fact: ', 'know that ']:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        m = re.match(r'^(.+?)\s+(?:is|are)\s+(.+)$', text, re.IGNORECASE)
        if m:
            subject    = m.group(1).strip().lower()
            definition = m.group(2).strip().lower()
            if len(subject.split()) <= 5 and len(definition.split()) >= 3:
                return f"what is {subject}", f"{subject} is {definition}"
        return None

    def toggle_thinking(self):
        self.show_thinking = not self.show_thinking
        return f"Thinking mode {'ON — showing confidence scores' if self.show_thinking else 'OFF'}."

    def forget(self):
        self.memory.clear()
        return "Conversation memory cleared."

    def stats(self):
        base    = len(QA_PAIRS)
        learned = len(self.qa_pairs) - base
        return (f"Knowledge: {len(self.qa_pairs)} facts "
                f"({base} built-in + {learned} learned) | "
                f"Graph: {len(self.graph.edges)} nodes")


def should_ask_back(score, question):
    return score < 0.15 and len(key_tokens(question)) <= 3


# ============================================================
# ENTRY POINT
# ============================================================

HELP = """
Commands:
  /think         — toggle confidence scores ON/OFF
  /history       — show recent conversation
  /forget        — clear conversation memory
  /stats         — knowledge base + graph size
  /graph <word>  — explore concept relationships
  /path <w1> <w2>— find path between two concepts
  /help          — this help
  quit           — exit

Teaching new facts:
  "Kerala is a state in southern India"
  "remember that ARIA is an AI companion built by Arjun"
  "Qwen3 is a large language model made by Alibaba"
"""

if __name__ == "__main__":
    print("=" * 58)
    print("  GF-SDM v6  |  Think + Learn + Combine + Graph")
    print("  Pure Python | No dependencies | Termux ready")
    print("=" * 58)

    ai = GFSDMv6()
    print(f"\n  {len(QA_PAIRS)} built-in facts. Type /help for commands.\n")

    print("--- Demo ---\n")
    demos = [
        "what is consciousness",
        "what is a black hole",
        "what connects gravity and stars",   # combine/inference
        "what is light and energy",          # multi-concept
    ]
    for q in demos:
        print(f"You: {q}")
        print(f"AI : {ai.chat(q)}\n")

    print("-" * 58)
    print("Your turn!\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user: continue
        if user.lower() in ('quit','exit','q'): break
        if user == '/think':    print(f"AI : {ai.toggle_thinking()}\n"); continue
        if user == '/history':  print(f"\n{ai.memory.summary()}\n"); continue
        if user == '/forget':   print(f"AI : {ai.forget()}\n"); continue
        if user == '/stats':    print(f"AI : {ai.stats()}\n"); continue
        if user == '/help':     print(HELP); continue
        if user.startswith('/graph '):
            w = user[7:].strip()
            print(f"AI : {ai.graph_explore(w)}\n"); continue
        if user.startswith('/path '):
            parts = user[6:].strip().split()
            if len(parts) >= 2:
                print(f"AI : {ai.path_between(parts[0], parts[1])}\n")
            continue
        print(f"AI : {ai.chat(user)}\n")

    print("\n[Done]")
