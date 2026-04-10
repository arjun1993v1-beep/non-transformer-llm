"""
GF-SDM v15 — Neuro-Symbolic AI with Concept-ID Neural Core
============================================================
Architecture: Truth first. Language second.

Philosophy:
  Old way: words → noisy n-gram model → hallucination
  New way: concepts → clean relational model → facts → language

Full Pipeline:
  Question
    ↓ ConceptExtractor  → concept IDs  (Layer 1)
    ↓ ConceptBrain      → predict next concept IDs  (Layer 2 — NEW)
    ↓ ConceptGraph      → validate predicted path  (Layer 3)
    ↓ FactAnchor        → find grounding fact  (Layer 4)
    ↓ LanguageDecoder   → concept IDs → natural sentence  (Layer 5)
    → Final answer

What changed from v14.3:
  ✦ Neural model now operates on CONCEPT IDs, not word IDs
    → Vocab shrinks from 600+ words to ~200 concepts
    → Model learns relationships, not sentence surface patterns
    → Training is faster and cleaner
  ✦ Language Decoder is separate from neural learning
    → Decode step: [gravity, attract, mass] → "gravity attracts objects with mass"
    → Neural never has to learn words — just concepts
  ✦ Hard domain filtering kept from v14.3
  ✦ Question routing kept from v14.3 (what-is → direct, how/why → reasoning)
  ✦ Multi-hop still disabled until concept brain is stable

Pure Python + Numpy. Runs on i3-2100 / Termux / Bodhi Linux.
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

WEAK_WORDS = {
    'system','process','thing','part','type','form',
    'way','work','use','used','using','known','called',
    'based','related','including','example','result',
    'number','amount','level','state','structure',
    'material','object','body','time','place','area',
    'region','surface','layer','unit','point',
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
        (['falls','fall','drops','drop','descends','descend'], 'fall'),
        (['rises','rise','ascends','ascend','lifts','lift'], 'rise'),
    ]

    def __init__(self):
        self.word_to_canon = {}
        for group, canon in self.SYNONYM_GROUPS:
            for word in group:
                self.word_to_canon[word] = canon

    def normalize(self, word):
        return self.word_to_canon.get(word.lower(), word.lower())

    def normalize_set(self, words):
        return {
            self.normalize(w)
            for w in words
            if w not in STOP
            and len(w) > 2
            and self.normalize(w) not in WEAK_WORDS
        }

    def normalize_text(self, text):
        words = tokenize(text)
        return ' '.join(self.normalize(w) for w in words)


NORMALIZER = ConceptNormalizer()


# ============================================================
# DOMAIN TAGGING
# ============================================================

DOMAIN_TAGS = {
    'physics':     {'gravity','force','mass','energy','light','photon',
                    'wave','radiation','electromagnetic','quantum','particle',
                    'relativity','spacetime','motion','velocity','momentum',
                    'thermodynamics','entropy','nuclear','fusion','fission',
                    'electric','magnetic','charge','current','voltage'},
    'astronomy':   {'star','sun','planet','galaxy','universe','orbit',
                    'blackhole','supernova','neutron','comet','asteroid',
                    'cosmos','solar','lunar','telescope','nebula','quasar',
                    'gravitational','redshift','darkmatter','bigbang',
                    'hole','event','horizon','hawking','collapse','stellar'},
    'biology':     {'dna','cell','gene','evolution','neuron','brain',
                    'protein','organism','species','mutation','chromosome',
                    'ecosystem','biodiversity','photosynthesis','bacteria',
                    'virus','immune','membrane','mitosis','ribosomes',
                    'memory','neurons','synapse','axon','dendrite',
                    'hippocampus','neuroplasticity','neurotransmitter',
                    'memories','connections','signals','nervous','spinal',
                    'cognitive','recall','store','episodic','semantic',
                    'sleep','consolidate','working','short','long'},
    'chemistry':   {'atom','molecule','electron','proton','neutron',
                    'element','compound','bond','reaction','acid',
                    'base','isotope','ion','catalyst','oxidation',
                    'periodic','carbon','hydrogen','oxygen','nitrogen',
                    'atoms','molecules','electrons','protons','nucleus',
                    'chemical','bonds','charges','ions'},
    'climate':     {'climate','atmosphere','greenhouse','carbon',
                    'temperature','weather','ocean','glacier','ozone',
                    'precipitation','humidity','warming','deforestation',
                    'renewable','fossil','emission','dioxide',
                    'global','sea','level','ice','cap','arctic','polar',
                    'drought','flood','storm','wind','rainfall'},
    'ai':          {'intelligence','learning','neural','algorithm',
                    'model','training','prediction','classification',
                    'transformer','attention','gradient','backprop',
                    'reinforcement','supervised','unsupervised',
                    'machine','data','network','deep','weights','layers'},
    'mathematics': {'number','prime','geometry','calculus','algebra',
                    'statistics','probability','theorem','proof','function',
                    'infinity','zero','integer','rational','vector','matrix',
                    'equation','formula','logic','set','graph','sequence'},
    'medicine':    {'heart','lung','liver','kidney','blood','immune',
                    'muscle','bone','nerve','hormone','vaccine','antibiotic',
                    'diagnosis','surgery','therapy','cancer','diabetes',
                    'disease','patient','treatment','organ','tissue',
                    'body','health','medical','symptom'},
}

def detect_domain(fact_raw):
    words = set(tokenize(fact_raw))
    best_domain, best_score = 'general', 0
    for domain, tags in DOMAIN_TAGS.items():
        score = len(words & tags)
        if score > best_score:
            best_score  = score
            best_domain = domain
    return best_domain


# ============================================================
# LAYER 1: CONCEPT MAP
# The heart of v15. Maps normalized concepts → integer IDs.
# Vocab: ~200 concepts instead of 600+ words.
# ============================================================

class ConceptMap:
    """
    Bidirectional concept ↔ ID registry.
    Built from all facts at startup.
    Unknown concepts get ID 0 (UNK).
    """
    UNK_ID = 0

    def __init__(self):
        self.concept_to_id = {'<UNK>': 0}
        self.id_to_concept = ['<UNK>']

    def add(self, concept):
        c = NORMALIZER.normalize(concept)
        if c in WEAK_WORDS or c in STOP or len(c) < 2:
            return self.UNK_ID
        if c not in self.concept_to_id:
            self.concept_to_id[c] = len(self.id_to_concept)
            self.id_to_concept.append(c)
        return self.concept_to_id[c]

    def encode(self, concept):
        c = NORMALIZER.normalize(concept)
        return self.concept_to_id.get(c, self.UNK_ID)

    def decode(self, cid):
        if 0 <= cid < len(self.id_to_concept):
            return self.id_to_concept[cid]
        return '<UNK>'

    def size(self):
        return len(self.id_to_concept)

    def build_from_facts(self, facts_raw):
        for raw in facts_raw:
            for word in keywords(raw):
                self.add(word)
        return self

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'id_to_concept': self.id_to_concept,
                       'concept_to_id': self.concept_to_id}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.id_to_concept = d['id_to_concept']
        self.concept_to_id = d['concept_to_id']
        return self


# ============================================================
# LAYER 2: FACT → CONCEPT SEQUENCE
# Converts raw facts into sequences of concept IDs.
# This is the training signal for the ConceptBrain.
# ============================================================

def fact_to_concept_sequence(raw, concept_map):
    """
    "gravity is a fundamental force that attracts objects with mass"
    → normalize → filter stop/weak
    → [gravity, attract, mass, object, force, fundamental]
    → [1, 3, 2, 5, 4, 7]
    """
    words = keywords(raw)
    ids = []
    for w in words:
        cid = concept_map.encode(w)
        if cid != ConceptMap.UNK_ID:
            ids.append(cid)
    # deduplicate while preserving order
    seen, unique = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def facts_to_training_pairs(facts_raw, concept_map, context_size=3):
    """
    Build (context_ids, target_id) pairs from concept sequences.
    This is what the ConceptBrain trains on.

    Example:
      sequence = [1, 3, 2, 5]
      pairs:
        ([1],       3)
        ([1, 3],    2)
        ([1, 3, 2], 5)
    """
    pad_id = ConceptMap.UNK_ID
    pairs  = []
    for raw in facts_raw:
        seq = fact_to_concept_sequence(raw, concept_map)
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            ctx = seq[max(0, i - context_size):i]
            while len(ctx) < context_size:
                ctx = [pad_id] + ctx
            pairs.append((ctx, seq[i]))
    return pairs


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
# LAYER 2: CONCEPT BRAIN
# Same architecture as v14 NeuralBrain but operates on
# concept IDs instead of word IDs.
#
# Input : context of concept IDs   e.g. [1, 3]
# Output: predicted next concept ID e.g. 2
#
# Because concepts are normalized and deduplicated:
#   - Vocab: ~150-200 IDs  (vs 600+ words)
#   - Cleaner signal: "attract" always = 3 (not also "attracts","attracted")
#   - Faster convergence, fewer parameters needed
# ============================================================

class ConceptBrain:
    def __init__(self, num_concepts, embed_dim=48, hidden_dim=96):
        self.num_concepts = num_concepts
        self.embed_dim    = embed_dim
        self.hidden_dim   = hidden_dim
        scale_e  = 0.1
        scale_w1 = np.sqrt(2 / embed_dim)
        scale_w2 = np.sqrt(2 / hidden_dim)
        self.params = {
            'E' : np.random.randn(num_concepts, embed_dim)  * scale_e,
            'W1': np.random.randn(embed_dim, hidden_dim)    * scale_w1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, num_concepts) * scale_w2,
            'b2': np.zeros(num_concepts),
        }
        self.optimizer = Adam(lr=0.003)

    def forward(self, context_ids):
        E      = self.params['E']
        valid  = [i for i in context_ids if i > 0]
        emb    = np.mean(E[valid], axis=0) if valid else np.zeros(self.embed_dim)
        h_pre  = emb @ self.params['W1'] + self.params['b1']
        h      = np.maximum(0, h_pre)
        logits = h @ self.params['W2'] + self.params['b2']
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()
        return probs, {'emb': emb, 'h_pre': h_pre, 'h': h,
                       'probs': probs, 'context_ids': context_ids}

    def backward(self, cache, y_id):
        probs, h, h_pre  = cache['probs'], cache['h'], cache['h_pre']
        emb, ctx         = cache['emb'], cache['context_ids']
        loss             = -np.log(probs[y_id] + 1e-10)
        d_logits         = probs.copy()
        d_logits[y_id]  -= 1.0
        d_W2  = np.outer(h, d_logits)
        d_b2  = d_logits
        d_h   = self.params['W2'] @ d_logits
        d_pre = d_h * (h_pre > 0)
        d_W1  = np.outer(emb, d_pre)
        d_b1  = d_pre
        d_emb = self.params['W1'] @ d_pre
        d_E   = np.zeros_like(self.params['E'])
        valid = [i for i in ctx if i > 0]
        n     = max(len(valid), 1)
        for i in valid:
            d_E[i] += d_emb / n
        return loss, {'E': d_E, 'W1': d_W1, 'b1': d_b1,
                      'W2': d_W2, 'b2': d_b2}

    def train_step(self, context_ids, y_id):
        probs, cache = self.forward(context_ids)
        loss, grads  = self.backward(cache, y_id)
        self.params  = self.optimizer.step(self.params, grads)
        return loss

    def predict(self, context_ids, temperature=0.7, top_k=8):
        probs, _ = self.forward(context_ids)
        probs[0] = 0.0  # never predict UNK
        top_ids  = np.argsort(probs)[-top_k:]
        top_p    = probs[top_ids] ** (1.0 / max(temperature, 0.1))
        top_p   /= top_p.sum()
        return int(np.random.choice(top_ids, p=top_p))

    def predict_top(self, context_ids, k=5):
        probs, _ = self.forward(context_ids)
        probs[0] = 0.0
        top_ids  = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top_ids]

    def save(self, path):
        np.savez(path, **self.params,
                 num_concepts=np.array([self.num_concepts]),
                 embed_dim   =np.array([self.embed_dim]),
                 hidden_dim  =np.array([self.hidden_dim]))

    @classmethod
    def load(cls, path):
        d     = np.load(path + '.npz')
        brain = cls(int(d['num_concepts'][0]),
                    int(d['embed_dim'][0]),
                    int(d['hidden_dim'][0]))
        brain.params = {k: d[k] for k in ['E','W1','b1','W2','b2']}
        return brain


# ============================================================
# TRAINING
# ============================================================

def train_concept_brain(facts_raw, concept_map, epochs=60,
                        embed_dim=48, hidden_dim=96,
                        lr=0.003, context_size=3, verbose=True):
    pairs = facts_to_training_pairs(facts_raw, concept_map, context_size)
    if verbose:
        print(f"[v15] Concept IDs   : {concept_map.size()}")
        print(f"[v15] Training pairs: {len(pairs)}")

    brain = ConceptBrain(concept_map.size(), embed_dim, hidden_dim)
    brain.optimizer.lr = lr

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = sum(brain.train_step(c, y) for c, y in pairs)
        avg = total_loss / max(len(pairs), 1)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")

    return brain


# ============================================================
# FACT INDEX
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
        self.domain   = detect_domain(raw)
        self.concepts = NORMALIZER.normalize_set(keyset(raw))
        self._parse()

    def _parse(self):
        words = tokenize(self.raw)
        for i, w in enumerate(words):
            if w in RELATION_WORDS:
                self.subject  = ' '.join(words[:i]).strip()
                self.relation = NORMALIZER.normalize(w)
                self.objects  = [
                    NORMALIZER.normalize(w)
                    for w in words[i+1:]
                    if w not in STOP and len(w) > 2
                    and NORMALIZER.normalize(w) not in WEAK_WORDS
                ]
                return
        self.subject = tokenize(self.raw)[0] if self.raw else ''
        self.objects = [
            NORMALIZER.normalize(w)
            for w in tokenize(self.raw)[1:]
            if w not in STOP and NORMALIZER.normalize(w) not in WEAK_WORDS
        ]


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
        norm   = {NORMALIZER.normalize(c) for c in concepts}
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
        self.edges       = {}
        self.fact_map    = {}
        self.node_domain = {}

    def add_fact(self, fact):
        concepts = list(fact.concepts)
        if len(concepts) > 6:
            concepts = concepts[:6]
        for c in concepts:
            if c not in self.node_domain:
                self.node_domain[c] = fact.domain
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
        return sorted(self.edges.get(c, {}).items(),
                      key=lambda x: -x[1])[:top_k]

    def get_edge_fact(self, c1, c2):
        key = (min(c1,c2), max(c1,c2))
        return self.fact_map.get(key, '')

    def trigger(self, word, top_k=5):
        c = NORMALIZER.normalize(word)
        if c not in self.edges: return []
        return sorted(self.edges[c].items(),
                      key=lambda x: -x[1])[:top_k]

    def validate_concept_sequence(self, concept_ids, concept_map):
        """
        Layer 3: Graph validation.
        Check how many consecutive pairs in the predicted concept
        sequence are connected in the graph.
        Returns a confidence score 0.0 – 1.0.
        """
        if len(concept_ids) < 2:
            return 1.0
        hits = 0
        for i in range(len(concept_ids) - 1):
            c1 = concept_map.decode(concept_ids[i])
            c2 = concept_map.decode(concept_ids[i+1])
            if c2 in self.edges.get(c1, {}):
                hits += 1
        return hits / (len(concept_ids) - 1)


# ============================================================
# CLUSTER MEMORY
# ============================================================

class MemoryCluster:
    def __init__(self, name, domain='general'):
        self.name     = name
        self.domain   = domain
        self.facts    = []
        self.concepts = {}
        self.strength = 1.0

    def add_fact(self, fact_obj):
        self.facts.append(fact_obj.raw)
        for c in fact_obj.concepts:
            self.concepts[c] = self.concepts.get(c, 0) + 1

    def similarity(self, query_concepts):
        if not query_concepts or not self.concepts:
            return 0.0
        score = sum(self.concepts[c] for c in query_concepts if c in self.concepts)
        return score / max(len(self.concepts), 1)

    def top_concepts(self, n=8):
        return sorted(self.concepts.items(), key=lambda x: -x[1])[:n]

    def __repr__(self):
        return (f"Cluster({self.name!r}, facts={len(self.facts)}, "
                f"strength={self.strength:.1f})")


class ClusterMemory:
    def __init__(self):
        self.clusters = {}

    def build(self, fact_index):
        self.clusters = {}
        domain_facts  = {}
        for fact in fact_index.facts:
            d = fact.domain
            domain_facts.setdefault(d, []).append(fact)

        for domain, facts in domain_facts.items():
            concept_freq = {}
            for f in facts:
                for c in f.concepts:
                    concept_freq[c] = concept_freq.get(c, 0) + 1
            top = sorted(concept_freq.items(), key=lambda x: -x[1])[:5]
            for concept, _ in top:
                cname   = f"{domain}:{concept}"
                cluster = MemoryCluster(cname, domain)
                for f in facts:
                    if concept in f.concepts:
                        cluster.add_fact(f)
                if cluster.facts:
                    self.clusters[cname] = cluster
            catch = MemoryCluster(f"{domain}:general", domain)
            for f in facts: catch.add_fact(f)
            self.clusters[f"{domain}:general"] = catch

    def recall(self, query_concepts, top_k=2, query_domain=None):
        """
        Hard domain filter + keyword boost + minimum threshold.
        Fix 2: exact keyword match in cluster → +5 score boost
        Fix 3: sim < 0.05 → reject (no weak matches)
        """
        scored = []
        for cluster in self.clusters.values():
            sim = cluster.similarity(query_concepts)

            # Fix 3: minimum threshold — weak overlap never wins
            if sim < 0.05:
                continue

            # Hard domain reject
            if query_domain and query_domain != 'general':
                if cluster.domain != query_domain and cluster.domain != 'general':
                    continue

            # Fix 2: exact keyword match → strong boost
            # e.g. query has 'memory' and cluster.concepts has 'memory' → +5
            exact_hits = sum(5 for c in query_concepts if c in cluster.concepts)
            sim += exact_hits

            # Suppress :general catch-alls
            if cluster.name.endswith(':general'):
                sim *= 0.10

            scored.append((sim, cluster))

        scored.sort(key=lambda x: -x[0])

        has_specific = any(not c.name.endswith(':general') for _, c in scored)
        filtered = []
        for sim, c in scored:
            if c.name.endswith(':general') and has_specific:
                continue
            filtered.append((sim, c))
            if len(filtered) >= top_k:
                break

        return [c for _, c in filtered]

    def get_local_facts(self, clusters, top_k=12):
        seen, facts = set(), []
        for cluster in clusters:
            for raw in cluster.facts:
                if raw not in seen:
                    seen.add(raw)
                    facts.append(raw)
        return facts[:top_k]

    def stats(self):
        domains = {}
        for name, c in self.clusters.items():
            domains[c.domain] = domains.get(c.domain, 0) + 1
        lines = [f"ClusterMemory: {len(self.clusters)} clusters"]
        for d, n in sorted(domains.items()):
            lines.append(f"  {d}: {n} clusters")
        return '\n'.join(lines)


# ============================================================
# LAYER 5: LANGUAGE DECODER
# concept IDs → natural language sentence
#
# This is the key insight: language generation is DECOUPLED
# from the neural model. The brain predicts concept sequences.
# The decoder renders them as readable sentences.
#
# Strategy:
#   1. Find the best matching fact for the concept sequence
#   2. Return that fact's raw text (clean, factual, grounded)
#   3. Fallback: compose a sentence from concept words
# ============================================================

class LanguageDecoder:
    def __init__(self, fact_index, concept_map):
        self.fact_index  = fact_index
        self.concept_map = concept_map

    def decode(self, concept_ids, question, relevant_facts):
        """
        Convert a sequence of concept IDs back to a natural sentence.

        Priority:
          1. If predicted concepts overlap well with a known fact → return fact
          2. Fallback: compose a simple sentence from the concept words
        """
        if not concept_ids:
            return None

        predicted_concepts = {
            self.concept_map.decode(cid)
            for cid in concept_ids
            if cid != ConceptMap.UNK_ID
        }

        # Step 1: score all relevant facts by overlap with predicted concepts
        best_fact   = None
        best_score  = 0
        for fact in relevant_facts:
            overlap = len(fact.concepts & predicted_concepts)
            if overlap > best_score:
                best_score = overlap
                best_fact  = fact

        if best_fact and best_score >= 2:
            raw = best_fact.raw
            return raw[0].upper() + raw[1:]

        # Step 2: compose from concept words directly
        # Filter out very generic concepts, keep informative ones
        concept_words = [
            self.concept_map.decode(cid)
            for cid in concept_ids
            if cid != ConceptMap.UNK_ID
            and self.concept_map.decode(cid) not in WEAK_WORDS
        ]
        if len(concept_words) >= 2:
            return ' '.join(concept_words).capitalize() + '.'

        # Final fallback: return best relevant fact
        if relevant_facts:
            raw = relevant_facts[0].raw
            return raw[0].upper() + raw[1:]

        return None


# ============================================================
# MEMORY
# ============================================================

class Memory:
    def __init__(self, max_turns=5):
        self.turns      = []
        self.max_turns  = max_turns
        self.short_term = []
        self.max_short  = 10

    def add(self, q, a):
        self.turns.append((q, a))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        self.short_term.append(q)
        if len(self.short_term) > self.max_short:
            self.short_term.pop(0)

    def enrich(self, question):
        kw = keywords(question)
        if len(kw) <= 1 and self.turns:
            last_q, _ = self.turns[-1]
            return question + ' ' + last_q
        return question

    def clear(self):
        self.turns = []; self.short_term = []

    def summary(self):
        if not self.turns:
            return "No conversation history."
        lines = ["Recent turns:"]
        for q, a in self.turns[-3:]:
            lines.append(f"  Q: {q}\n  A: {a[:60]}{'...' if len(a)>60 else ''}")
        return '\n'.join(lines)


# ============================================================
# QUESTION ROUTING
# ============================================================

def is_simple_lookup(question):
    """'what is X' → return best fact directly. No neural needed."""
    q = question.lower().strip()
    return bool(re.match(
        r'^(what\s+is|what\s+are|define|what\s+does\s+\w+\s+mean'
        r'|who\s+is|where\s+is|when\s+is)\b', q))

def is_reasoning_question(question):
    """'how/why/explain' → use concept brain."""
    q = question.lower().strip()
    return bool(re.match(
        r'^(how|why|what\s+connects|what\s+causes|explain|describe)\b', q))


# ============================================================
# REASONING ENGINE
# ============================================================

class ReasoningEngine:
    def __init__(self):
        self.index    = FactIndex()
        self.graph    = ConceptGraph()
        self.facts    = []
        self.clusters = ClusterMemory()

    def load_facts(self, facts_raw, learned_raw=None):
        for raw in facts_raw:
            self._add(raw)
        for raw in (learned_raw or []):
            self._add(raw)
        self.clusters.build(self.index)
        print(f"[v15] Facts: {len(self.facts)}  "
              f"Clusters: {len(self.clusters.clusters)}")

    def _add(self, raw):
        fid  = self.index.add(raw)
        fact = self.index.facts[fid]
        self.graph.add_fact(fact)
        self.facts.append(raw)

    def teach(self, raw):
        self._add(raw)
        self.clusters.build(self.index)
        learned = load_learned()
        learned.append(['', raw])
        save_learned(learned)

    def find_relevant_facts(self, question, top_k=4):
        q_concepts = NORMALIZER.normalize_set(keyset(question))
        q_keywords = set(keywords(question))

        # Fix 1: Force domain lock for simple lookups.
        # "what is memory" → domain derived from the SUBJECT word only,
        # not from the full question. This gives a much tighter signal.
        if is_simple_lookup(question):
            kw = [w for w in keywords(question)
                  if w not in {'what','is','are','define','who','where','when'}]
            if kw:
                # Fix 4: Direct lookup shortcut — deterministic, O(1).
                # Look up the subject concept directly in the fact index.
                # If we get subject-matching facts, return them immediately
                # WITHOUT going through cluster scoring at all.
                direct = self.index.lookup(kw[0])
                # Score by subject match: facts whose subject IS the keyword win
                subj_match = [f for f in direct
                              if kw[0] in tokenize(f.subject)]
                if subj_match:
                    return subj_match[:top_k]
                if direct:
                    return direct[:top_k]

                # Domain from the subject word itself (not the whole question)
                q_domain = detect_domain(kw[0])
            else:
                q_domain = detect_domain(question)
        else:
            q_domain = detect_domain(question)

        best_clusters = self.clusters.recall(q_concepts, top_k=2,
                                             query_domain=q_domain)
        if best_clusters:
            local_raw = self.clusters.get_local_facts(best_clusters,
                                                      top_k=top_k * 3)
            scored = []
            for raw in local_raw:
                fid = next((i for i, f in enumerate(self.index.facts)
                            if f.raw == raw), None)
                if fid is None: continue
                fact_obj      = self.index.facts[fid]
                fact_concepts = NORMALIZER.normalize_set(keyset(raw))
                score = len(q_concepts & fact_concepts)
                if score < 2: continue
                subj_words = set(tokenize(fact_obj.subject))
                if subj_words & q_keywords:
                    score += 3
                scored.append((score, fact_obj))

            scored.sort(key=lambda x: -x[0])
            result = [fo for _, fo in scored[:top_k]]
            if result: return result

            # Relax to overlap ≥ 1
            scored2 = []
            for raw in local_raw:
                fid = next((i for i, f in enumerate(self.index.facts)
                            if f.raw == raw), None)
                if fid is None: continue
                fact_obj      = self.index.facts[fid]
                fact_concepts = NORMALIZER.normalize_set(keyset(raw))
                score = len(q_concepts & fact_concepts)
                if score < 1: continue
                subj_words = set(tokenize(fact_obj.subject))
                if subj_words & q_keywords: score += 3
                scored2.append((score, fact_obj))
            scored2.sort(key=lambda x: -x[0])
            result = [fo for _, fo in scored2[:top_k]]
            if result: return result

        return self.index.lookup_many(keywords(question), top_k=top_k)

    def trigger_recall(self, question):
        q_words  = keywords(question)
        triggers = []
        for w in q_words:
            c = NORMALIZER.normalize(w)
            triggered = self.graph.trigger(c, top_k=3)
            triggers.extend([(nb, wt) for nb, wt in triggered])
        if not triggers: return None
        triggers.sort(key=lambda x: -x[1])
        facts = self.index.lookup(triggers[0][0])
        return facts[0].raw if facts else None

    def active_clusters(self, question):
        q_concepts = NORMALIZER.normalize_set(keyset(question))
        q_domain   = detect_domain(question)
        return self.clusters.recall(q_concepts, top_k=3,
                                    query_domain=q_domain)


# ============================================================
# SELF-CHECK
# ============================================================

def self_check(brain, concept_map, facts_raw, n=20, context_size=3):
    sample  = random.sample(facts_raw, min(n, len(facts_raw)))
    correct = 0; total = 0
    results = []
    for raw in sample:
        seq = fact_to_concept_sequence(raw, concept_map)
        if len(seq) < 2: continue
        ctx = [ConceptMap.UNK_ID] * context_size
        ctx[-1] = seq[0]
        top = brain.predict_top(ctx, k=5)
        top_concepts = [concept_map.decode(i) for i, _ in top]
        expected     = concept_map.decode(seq[1])
        hit          = expected in top_concepts
        if hit: correct += 1
        total += 1
        results.append({'seed': concept_map.decode(seq[0]),
                        'expected': expected,
                        'predicted': top_concepts[0],
                        'hit': hit})
    acc = correct / max(total, 1) * 100
    print(f"\n{'='*52}")
    print(f"  CONCEPT BRAIN SELF-CHECK  (context={context_size})")
    print(f"{'='*52}")
    print(f"  Facts tested    : {total}")
    print(f"  Concept IDs     : {concept_map.size()}  (vs ~600 word tokens)")
    print(f"  Top-5 accuracy  : {acc:.1f}%")
    if   acc >= 70: print("  ✓ Concept brain learned well")
    elif acc >= 45: print("  ~ Partial — train more epochs")
    else:           print("  ✗ Needs more training")
    print(f"\n  Sample predictions (concept-level):")
    for r in results[:5]:
        s = '✓' if r['hit'] else '✗'
        print(f"  {s} '{r['seed']}' → "
              f"expected='{r['expected']}' got='{r['predicted']}'")
    print()
    return acc


# ============================================================
# PATHS
# ============================================================

BRAIN_PATH        = 'v15_brain'
CMAP_PATH         = 'v15_concepts.json'
LEARNED_PATH      = 'v15_learned.json'

def load_learned():
    if os.path.exists(LEARNED_PATH):
        with open(LEARNED_PATH) as f: return json.load(f)
    return []

def save_learned(data):
    with open(LEARNED_PATH, 'w') as f: json.dump(data, f, indent=2)


# ============================================================
# BUILT-IN FACTS
# ============================================================

BUILTIN_FACTS = [
    # --- GRAVITY ---
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
    # --- LIGHT ---
    "light is electromagnetic radiation that travels at 299792 kilometers per second.",
    "light can behave as both a wave and a particle.",
    "light carries energy and can transfer that energy to matter.",
    "light bends when passing near massive objects due to gravity.",
    "visible light is a small part of the electromagnetic spectrum.",
    "light is made of particles called photons.",
    "different colors of light have different wavelengths and frequencies.",
    "ultraviolet light is invisible to humans but can cause sunburn.",
    "infrared light is emitted by warm objects as heat radiation.",
    # --- DNA ---
    "dna is a molecule that carries genetic instructions for life.",
    "dna is made of four bases adenine thymine cytosine and guanine.",
    "dna stores information in sequences of base pairs.",
    "dna replicates itself so cells can divide and copy information.",
    "mutations in dna drive the process of evolution over generations.",
    "dna is found in the nucleus of every cell in the human body.",
    "the human genome contains about three billion base pairs of dna.",
    "dna damage can lead to cancer if repair mechanisms fail.",
    "identical twins share the same dna sequence.",
    # --- EVOLUTION ---
    "evolution is the change in inherited traits of populations over generations.",
    "natural selection favors traits that improve survival and reproduction.",
    "species evolve from common ancestors through accumulated mutations.",
    "evolution produces the diversity of life on earth.",
    "charles darwin first proposed the theory of natural selection.",
    "all life on earth shares a common ancestor from about four billion years ago.",
    "evolution can be observed in bacteria that develop antibiotic resistance.",
    # --- NEURONS ---
    "neurons are cells that transmit electrical signals through the nervous system.",
    "neurons communicate using chemicals called neurotransmitters.",
    "synapses are junctions between neurons where signals pass.",
    "the brain contains about 86 billion neurons.",
    "memory is formed by strengthening connections between neurons.",
    "a neuron has a cell body dendrites and an axon.",
    "electrical signals in neurons are called action potentials.",
    "neuroplasticity is the ability of neurons to form new connections.",
    # --- MEMORY ---
    "memory is the ability to store and recall past experiences.",
    "short term memory holds a small amount of information briefly.",
    "long term memory stores information for extended periods.",
    "sleep helps consolidate memories in the brain.",
    "the hippocampus is important for forming new memories.",
    "working memory holds information actively in mind for immediate use.",
    "episodic memory stores personal experiences and events.",
    "semantic memory stores general knowledge and facts about the world.",
    # --- STARS ---
    "stars are massive balls of hot plasma that produce light and heat.",
    "stars generate energy through nuclear fusion in their cores.",
    "the sun is a medium sized star at the center of our solar system.",
    "when massive stars die they explode in supernovae.",
    "the nearest star to earth besides the sun is proxima centauri.",
    "neutron stars are incredibly dense remnants of supernova explosions.",
    # --- ENERGY ---
    "energy is the ability to do work or cause change.",
    "energy cannot be created or destroyed only transformed.",
    "kinetic energy is the energy of motion.",
    "potential energy is stored energy due to position or configuration.",
    "nuclear energy is released when atomic nuclei split or fuse.",
    "thermal energy is the internal energy of a system from particle motion.",
    "chemical energy is stored in the bonds between atoms.",
    "the law of conservation of energy states that total energy is always constant.",
    # --- BLACK HOLES ---
    "a black hole forms when a massive star collapses under its own gravity.",
    "the event horizon is the boundary beyond which nothing can escape.",
    "black holes warp space and time around them.",
    "hawking radiation is a theoretical process where black holes slowly lose energy.",
    "supermassive black holes exist at the centers of most galaxies.",
    "the first image of a black hole was captured in 2019.",
    "time passes more slowly closer to a black hole.",
    "black holes can merge and produce gravitational waves.",
    # --- WATER ---
    "water is a molecule made of two hydrogen atoms and one oxygen atom.",
    "water exists as liquid solid and gas depending on temperature.",
    "water is essential for all known forms of life.",
    "water covers about 71 percent of earth surface.",
    "water has a high heat capacity allowing it to regulate temperatures.",
    "ice is less dense than liquid water which is why it floats.",
    "water is a universal solvent that dissolves many substances.",
    # --- AI ---
    "artificial intelligence is the simulation of human intelligence by machines.",
    "machine learning allows systems to learn from data without explicit programming.",
    "neural networks are computing systems inspired by biological neurons.",
    "deep learning uses many layers of neural networks to find patterns.",
    "language models learn to predict the next word from large text datasets.",
    "reinforcement learning trains agents by rewarding correct actions.",
    "supervised learning uses labeled data to train prediction models.",
    "transformers are neural network architectures that use attention mechanisms.",
    "gradient descent is the optimization algorithm used to train neural networks.",
    # --- ATOMS ---
    "atoms are the basic units of matter that make up all substances.",
    "an atom consists of a nucleus surrounded by electrons.",
    "the nucleus of an atom contains protons and neutrons.",
    "protons carry positive charge and neutrons carry no charge.",
    "electrons carry negative charge and orbit the nucleus.",
    "the number of protons in an atom determines which element it is.",
    "chemical bonds form when atoms share or transfer electrons.",
    "isotopes are atoms of the same element with different numbers of neutrons.",
    # --- CELLS ---
    "cells are the basic unit of life in all living organisms.",
    "every cell contains dna that encodes the organism genetic information.",
    "the cell membrane controls what enters and leaves the cell.",
    "mitochondria produce energy for the cell through cellular respiration.",
    "the cell nucleus contains the genetic material of the cell.",
    "cells divide through a process called mitosis to produce identical copies.",
    "bacteria are single celled organisms without a true nucleus.",
    # --- CLIMATE ---
    "climate is the long term pattern of weather in a region.",
    "the greenhouse effect traps heat in earth atmosphere.",
    "carbon dioxide is a greenhouse gas that contributes to global warming.",
    "the oceans absorb about 30 percent of the carbon dioxide produced by humans.",
    "global warming is causing ice caps to melt and sea levels to rise.",
    "the atmosphere protects life on earth from harmful solar radiation.",
    "photosynthesis by plants absorbs carbon dioxide and releases oxygen.",
    "deforestation reduces the earth ability to absorb carbon dioxide.",
    # --- MATHEMATICS ---
    "mathematics is the study of numbers patterns and logical structures.",
    "prime numbers are whole numbers greater than one divisible only by themselves.",
    "the pythagorean theorem states the square of the hypotenuse equals the sum of squares.",
    "calculus is the mathematics of change and motion.",
    "algebra uses symbols and rules to manipulate mathematical expressions.",
    "statistics is the study of collecting analyzing and interpreting data.",
    # --- HUMAN BODY ---
    "the human body contains about 37 trillion cells.",
    "the heart pumps blood through the circulatory system.",
    "the lungs exchange oxygen and carbon dioxide with the blood.",
    "the liver filters toxins from the blood and produces bile.",
    "the immune system defends the body against pathogens and disease.",
    "the brain is the control center of the nervous system.",
]


# ============================================================
# MAIN SYSTEM — GFSDMv15
# ============================================================

class GFSDMv15:
    """
    GF-SDM v15 — Neuro-Symbolic AI with Concept-ID Neural Core

    Architecture:
      Layer 1: ConceptMap     — word → concept ID  (~150 IDs vs 600 words)
      Layer 2: ConceptBrain   — concept ID → next concept ID  (neural)
      Layer 3: ConceptGraph   — validates predicted concept sequences
      Layer 4: FactAnchor     — grounds predictions in known facts
      Layer 5: LanguageDecoder — concept IDs → natural sentence

    Key insight:
      The neural model never sees words.
      It only sees concept IDs.
      Language is added last, as a pure decoder step.
      Truth first. Language second.
    """

    CONTEXT = 3

    def __init__(self):
        self.reasoning    = ReasoningEngine()
        self.concept_map  = ConceptMap()
        self.brain        = None
        self.decoder      = None
        self.memory       = Memory(max_turns=5)
        self._session     = 0

        # Load facts
        learned = [a for _, a in load_learned() if a]
        self.reasoning.load_facts(BUILTIN_FACTS, learned)

        # Build concept map from all facts
        all_facts = BUILTIN_FACTS + learned
        self.concept_map.build_from_facts(all_facts)

        # Try loading saved brain
        if os.path.exists(BRAIN_PATH + '.npz') and os.path.exists(CMAP_PATH):
            self.brain = ConceptBrain.load(BRAIN_PATH)
            self.concept_map.load(CMAP_PATH)
            self.decoder = LanguageDecoder(self.reasoning.index,
                                           self.concept_map)
            print(f"[v15] Concept brain loaded. "
                  f"Concepts: {self.concept_map.size()}")
        else:
            print("[v15] No saved brain. Run ai.train() first.")

    # ── TRAINING ──────────────────────────────────────────────

    def train(self, epochs=60, embed_dim=48, hidden_dim=96, lr=0.003):
        learned = [a for _, a in load_learned() if a]
        all_facts = BUILTIN_FACTS + learned

        # Rebuild concept map (includes any new learned facts)
        self.concept_map = ConceptMap()
        self.concept_map.build_from_facts(all_facts)

        print(f"[v15] Facts         : {len(all_facts)}")
        print(f"[v15] Training concept brain (epochs={epochs})...")

        self.brain = train_concept_brain(
            all_facts, self.concept_map,
            epochs=epochs, embed_dim=embed_dim,
            hidden_dim=hidden_dim, lr=lr)

        self.brain.save(BRAIN_PATH)
        self.concept_map.save(CMAP_PATH)
        self.decoder = LanguageDecoder(self.reasoning.index,
                                       self.concept_map)
        print("[v15] Concept brain saved.")

    def train_file(self, path, epochs=40):
        if not os.path.exists(path):
            print(f"[v15] Not found: {path}"); return
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        extra = [s.strip().lower() + '.'
                 for s in re.split(r'[.!?]', text)
                 if len(s.strip()) > 20]
        for e in extra:
            self.reasoning.teach(e)
        self.train(epochs=epochs)

    # ── CHAT ──────────────────────────────────────────────────

    def chat(self, question, temperature=0.65, top_k=6):
        question = question.strip()
        if not question: return "I'm listening."
        self._session += 1

        # Auto-teach
        teach = self._detect_teach(question)
        if teach:
            self.reasoning.teach(teach)
            self.concept_map.add(teach)
            self.memory.add(question, f"Learned: {teach}")
            return f"Understood. Added: {teach}"

        enriched = self.memory.enrich(question)

        # Show active cluster (only meaningful for non-direct-lookup paths)
        active = self.reasoning.active_clusters(enriched)
        cluster_info = ""
        if active and not is_simple_lookup(enriched):
            names = [c.name for c in active[:2]]
            cluster_info = f"[cluster: {', '.join(names)}]\n"

        answer, mode = self._answer(enriched, temperature, top_k)
        self.memory.add(question, answer)

        if mode == 'none':
            tr = self.reasoning.trigger_recall(enriched)
            if tr:
                return f"{cluster_info}(trigger)\n{tr[0].upper()+tr[1:]}"
            return f"{cluster_info}{answer}"
        return f"{cluster_info}({mode})\n{answer}"

    def _answer(self, question, temperature, top_k):
        relevant = self.reasoning.find_relevant_facts(question, top_k=3)
        if not relevant:
            return "I don't have information on that topic.", 'none'

        best = relevant[0]

        # Route: simple lookup → fact directly
        if is_simple_lookup(question):
            raw = best.raw
            return raw[0].upper() + raw[1:], 'direct fact'

        # Route: reasoning question → concept brain
        if is_reasoning_question(question) and self.brain:
            answer = self._concept_brain_answer(question, relevant,
                                                temperature, top_k)
            if answer:
                return answer, 'concept-brain'

        # Fallback: best fact
        raw = best.raw
        return raw[0].upper() + raw[1:], 'direct fact'

    def _concept_brain_answer(self, question, relevant_facts,
                               temperature, top_k):
        """
        Full concept-brain pipeline:
          1. Extract concept IDs from question
          2. ConceptBrain predicts next concept IDs
          3. ConceptGraph validates the sequence
          4. LanguageDecoder renders to sentence
        """
        # Step 1: question → concept IDs
        q_words  = keywords(question)
        ctx_ids  = []
        for w in q_words:
            cid = self.concept_map.encode(w)
            if cid != ConceptMap.UNK_ID:
                ctx_ids.append(cid)

        if not ctx_ids:
            return None

        # Pad/trim to context window
        pad = ConceptMap.UNK_ID
        while len(ctx_ids) < self.CONTEXT:
            ctx_ids = [pad] + ctx_ids
        ctx_ids = ctx_ids[-self.CONTEXT:]

        # Step 2: predict concept sequence
        predicted = list(ctx_ids)
        for _ in range(6):   # predict up to 6 more concepts
            next_id = self.brain.predict(
                predicted[-self.CONTEXT:], temperature, top_k)
            if next_id == ConceptMap.UNK_ID:
                break
            predicted.append(next_id)

        # Remove padding from predicted
        predicted = [i for i in predicted if i != ConceptMap.UNK_ID]

        # Step 3: graph validation
        conf = self.reasoning.graph.validate_concept_sequence(
            predicted, self.concept_map)

        # If confidence too low → don't trust the prediction
        if conf < 0.3:
            return None

        # Step 4: language decode
        return self.decoder.decode(predicted, question, relevant_facts)

    # ── TEACH ─────────────────────────────────────────────────

    def teach(self, fact):
        raw = fact.lower().rstrip('.') + '.'
        self.reasoning.teach(raw)
        self.concept_map.build_from_facts([raw])
        print("[v15] Taught. Run ai.train() to update concept brain.")

    # ── INSPECT ───────────────────────────────────────────────

    def inspect(self, word):
        facts = self.reasoning.index.lookup(word)
        if not facts: return f"No facts for '{word}'."
        lines = [f"Facts about '{word}':"]
        for f in facts[:5]:
            lines.append(f"  · [{f.domain}] {f.raw}")
        return '\n'.join(lines)

    def domain(self, word):
        c = NORMALIZER.normalize(word)
        d = self.reasoning.graph.node_domain.get(c, 'unknown')
        return f"'{word}' → domain: {d}"

    def trigger(self, word):
        c = NORMALIZER.normalize(word)
        t = self.reasoning.graph.trigger(c, top_k=8)
        if not t: return f"No triggers for '{word}'."
        lines = [f"'{word}' triggers:"]
        for nb, wt in t:
            lines.append(f"  → {nb} ({wt:.2f})")
        return '\n'.join(lines)

    def concepts(self, word):
        """Show the concept ID for a word."""
        cid = self.concept_map.encode(word)
        canon = self.concept_map.decode(cid)
        return f"'{word}' → concept='{canon}' id={cid}"

    def check(self, n=20):
        if not self.brain:
            print("No brain trained."); return
        self_check(self.brain, self.concept_map, BUILTIN_FACTS,
                   n=n, context_size=self.CONTEXT)

    def clusters(self):
        return self.reasoning.clusters.stats()

    def active(self, question):
        active = self.reasoning.active_clusters(question)
        if not active:
            return f"No clusters activated for '{question}'."
        lines = [f"Activated clusters for '{question}':"]
        for c in active:
            top = ', '.join(f"{k}({v})" for k, v in c.top_concepts(n=4))
            lines.append(f"  ★ {c.name}  (strength={c.strength:.2f})")
            lines.append(f"    top: {top}  |  facts: {len(c.facts)}")
        return '\n'.join(lines)

    def stats(self):
        lines = [
            f"GF-SDM v15 — Neuro-Symbolic AI",
            f"  Facts        : {len(self.reasoning.facts)}",
            f"  Clusters     : {len(self.reasoning.clusters.clusters)}",
            f"  Concept IDs  : {self.concept_map.size()}",
            f"  Graph nodes  : {len(self.reasoning.graph.edges)}",
            f"  Session      : {self._session}",
            f"",
            f"  Architecture:",
            f"    Layer 1: ConceptMap     — word → concept ID",
            f"    Layer 2: ConceptBrain   — concept → concept  (neural)",
            f"    Layer 3: ConceptGraph   — sequence validation",
            f"    Layer 4: FactAnchor     — ground in real facts",
            f"    Layer 5: LanguageDecoder— IDs → sentence",
        ]
        if self.brain:
            total = sum(v.size for v in self.brain.params.values())
            lines += [
                f"",
                f"  Concept IDs  : {self.concept_map.size()}  (vs ~600 word tokens)",
                f"  Embed dim    : {self.brain.embed_dim}",
                f"  Hidden dim   : {self.brain.hidden_dim}",
                f"  Parameters   : {total:,}",
            ]
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
GF-SDM v15 — Neuro-Symbolic AI
Commands:
  /train [epochs]     — train concept brain (default 60 epochs)
  /train_file <f>     — train from text file
  /check              — self-check concept brain accuracy
  /clusters           — show memory clusters
  /active <question>  — show which cluster activates
  /trigger <word>     — show concept associations
  /inspect <word>     — facts containing word
  /concepts <word>    — show concept ID for word
  /domain <word>      — concept domain
  /stats              — full system stats
  /history            — conversation memory
  /forget             — clear memory
  /teach <sentence>   — add new fact
  /help               — this help
  quit                — exit

Architecture:
  Layer 1: word → concept ID (normalized, deduplicated)
  Layer 2: concept brain predicts next concept ID
  Layer 3: graph validates the predicted sequence
  Layer 4: fact anchors the answer in truth
  Layer 5: decoder renders concept IDs → language
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GF-SDM v15  |  Neuro-Symbolic AI")
    print("  Concept-ID Neural Core — Truth first. Language second.")
    print(f"  {len(BUILTIN_FACTS)} facts | Pure Python + Numpy")
    print("=" * 60 + "\n")

    ai = GFSDMv15()

    if not ai.brain:
        print("[v15] Auto-training concept brain (60 epochs)...")
        ai.train(epochs=60)
        print()

    ai.check(n=15)

    print("\n--- Cluster Activation Demo ---\n")
    for q in ["what is gravity", "what is memory",
              "how does dna work", "what is a black hole"]:
        print(ai.active(q))
        print()

    print("\n--- Answer Demo ---\n")
    demos = [
        "what is gravity",
        "what is memory",
        "how does dna work",
        "what is an atom",
        "what is energy",
        "what is climate",
        "what is a black hole",
        "what is evolution",
        "how does memory form",
        "why does light bend near gravity",
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
        if user == '/clusters':  print(f"\n{ai.clusters()}\n"); continue
        if user == '/stats':     print(f"\n{ai.stats()}\n"); continue
        if user == '/help':      print(HELP); continue
        if user == '/history':   print(f"\n{ai.memory.summary()}\n"); continue
        if user == '/forget':    ai.memory.clear(); print("Memory cleared.\n"); continue
        if user.startswith('/active '):
            print(f"\n{ai.active(user[8:].strip())}\n"); continue
        if user.startswith('/trigger '):
            print(f"\n{ai.trigger(user[9:].strip())}\n"); continue
        if user.startswith('/inspect '):
            print(f"\n{ai.inspect(user[9:].strip())}\n"); continue
        if user.startswith('/concepts '):
            print(f"\n{ai.concepts(user[10:].strip())}\n"); continue
        if user.startswith('/domain '):
            print(f"\n{ai.domain(user[8:].strip())}\n"); continue
        if user.startswith('/teach '):
            ai.teach(user[7:].strip()); continue
        if user.startswith('/train_file '):
            ai.train_file(user[12:].strip()); continue
        if user.startswith('/train'):
            parts  = user.split()
            epochs = int(parts[1]) if len(parts) > 1 else 60
            ai.train(epochs=epochs); continue

        print(f"AI: {ai.chat(user)}\n")

    print("\n[Done]")
