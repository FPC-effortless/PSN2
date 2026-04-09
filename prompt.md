PSN-2

Predictive Substrate Network — Generation 2

Unified Intelligence Architecture

Authoritative Product Requirements Document · Version 1.0 · April 2026 · CONFIDENTIAL

Field Value
System class Unified cognitive architecture — language, reasoning, emotion,

learning, and energy efficiency in one substrate
Category rival PSN — second generation; full rival to both LLM and

neurosymbolic hybrid approaches

Core thesis Separation between cognition, language, emotion, and learning is
the fundamental design error. PSN-2 eliminates all separations.
Primary benchmarks ARC-AGI-3, BIG-Bench Hard, human-parity language evaluation,

energy-per-inference, few-shot sample complexity
Lineage Supersedes PSN-1 v1.0 and v1.1 HEL Amendment. All v1.0
substrate mechanics carry forward; six new architectural systems
added.

New systems Unified Shape Language (USL), Spiking Computation Engine
(SCE), Emotional Shape System (ESS), Inner Speech Loop (ISL),
Curiosity Engine (CE), Social Learning System (SLS), Temporal
Abstraction Engine (TAE), Developmental Curriculum (DC)
Decision posture No architectural, mathematical, training, software, security, or

deployment decisions deferred

Owner Ideation Labs

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
The Unification Thesis
PSN-1 was a correct but incomplete step. It established that intelligence should emerge from
local prediction-error dynamics over a typed node-shape substrate, with VSA bond algebra,
verifier-gated commitment, and human-efficient learning through the dual-speed learning
engine. What it preserved from conventional AI design — and should not have — is the
separation between systems.
PSN-1 kept language downstream of reasoning. It kept emotion as a global field modulator
separate from cognitive shapes. It kept energy efficiency as a deployment consideration rather
than an architectural one. It kept learning and inference as distinct operational modes. These
separations are not architectural necessities — they are inherited assumptions from decades of
modular AI design that has consistently failed to produce general intelligence.
The biological evidence is clear. In the human brain:
• Language and reasoning share overlapping neural substrates. Broca&#39;s area is active
during both linguistic processing and complex sequential reasoning. Thought and
language co-constitute each other.
• Emotion and cognition are not separable. The amygdala, anterior cingulate cortex, and
prefrontal cortex form a unified system where emotional signals directly modulate
cognitive processing — not as a bias, but as information.
• Energy efficiency is achieved through sparse, event-driven computation. Neurons fire
only when their prediction error exceeds threshold. The brain consumes ~20W because
99% of neurons are silent at any given moment.
• Learning and inference are the same process. Synaptic weights update continuously
during experience. There is no separate &quot;training phase&quot; in a living brain.
• Development is ordered. Humans acquire sensorimotor grounding before causal
reasoning, causal reasoning before social cognition, social cognition before abstract
language, abstract language before formal logic. This order matters — each stage builds
the representational foundation for the next.
PSN-2 closes all five separations in one architecture. The result is a system where
language is native to the reasoning substrate, where emotional states are shapes
that bond to cognitive shapes, where computation is sparse and event-driven,
where learning is continuous, and where development proceeds in the biologically
validated order. This is not a list of features — it is one design decision: the
substrate is the intelligence, and the substrate is unified.

1. System Invariants

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
The following invariants define the PSN-2 class. All PSN-1 v1.0 invariants (I-1 through I-12) and
HEL invariants (I-13 through I-15) remain in force. These are additions.

Invariant Statement Consequence
I-16 Language is substrate-native,

not downstream

Words and phrases are VSA shape-bundles in the
same substrate as non-linguistic shapes. There is no
separate language model or projection head. Language
understanding = shape induction from linguistic input.
Language generation = shape-to-text codec operating
bidirectionally on the unified substrate.

I-17 Emotion is a shape class, not

a global modulator

Emotional states are first-class shapes with typed
bonds to goal shapes, hypothesis shapes, and self-
shapes. They influence reasoning through the normal
bond mechanics, not through a separate field vector.
The field vector is deprecated as a control mechanism.

I-18 Computation is event-driven by

default

Nodes compute only when their prediction error
exceeds a threshold tau_spike. Silent nodes consume
no compute. This is the default operational mode, not
an optimisation option. Inference energy scales with
task difficulty, not with model size.

I-19 Inner speech is a cognitive
operation, not an output
operation

The system can emit draft verbal propositions into
working memory during Recursive regime, re-encode
them as shapes, and use them as evidence in
subsequent pulses. Language is a tool for thinking, not
only a tool for communication.

I-20 Curiosity generates goals, it
does not bias search

When the system encounters a high-prediction-error
region it cannot resolve within budget, it generates a
structured internal goal to return to that region and
reduce uncertainty. This goal is stored in ERS and
scheduled for future exploration. Curiosity is active, not
passive.

I-21 Development is ordered and

gated

The six developmental stages (D1–D6) must be
completed sequentially. No stage may begin until the
previous stage&#39;s release gate is certified. Language
training (D4) may not begin until sensorimotor (D1),
causal (D2), and social (D3) grounding is certified. This
order is non-negotiable.

I-22 Observed reasoning traces are
first-class learning data

The Social Learning System can encode an observed
agent&#39;s reasoning trace as a shape sequence and store
it in ERS as if the system had produced it. Learning
from observation is architecturally equivalent to
learning from experience.

I-23 Temporal patterns are shapes Frequently executed pulse sequences are compressed
into temporal motif shapes stored in the attractor
library. Familiar tasks execute as temporal motif
retrieval + replay in Perceptive regime with near-zero
compute cost.

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
2. Cognitive Ontology v2 — The Fully Unified Substrate
PSN-2 extends the PSN-1 ontology by adding three new primitive classes and eliminating the
field vector as a separate control mechanism.

Primitive Definition What It Replaces or Extends
Linguistic shape A shape whose node-set
encodes words, phrases,
syntactic structure, or discourse
relations. Linguistic shapes are
VSA-bound exactly like non-
linguistic shapes. A sentence is a
composed shape built from word-
shapes and relation-bonds.

Replaces the render projection head.
Language is no longer downstream — it is
a shape class in the substrate.

Emotional shape A shape encoding an affective
state (fear, curiosity, confidence,
urgency, discomfort,
satisfaction). Emotional shapes
bond to goal shapes, hypothesis
shapes, and self-shapes through
typed emotional bonds. They
directly influence the stability and
commitment thresholds of
bonded shapes.

Replaces the field vector as the primary
mode-control mechanism. Emotional
influence is local and bond-mediated, not
global.

Temporal motif shape A compressed shape encoding a
frequently executed sequence of
pulses. Retrieving a temporal
motif shape allows the system to
replay the encoded pulse
sequence in Perceptive regime
without recomputing each step.
Duration and ordering are
encoded in the directional
geometry.

New — no PSN-1 equivalent. Implements
habit and chunking.

Self shape A shape encoding the system&#39;s
model of its own state,
competence, goals, and identity.
Self-shapes bond to emotional
shapes and goal shapes. They
are updated continuously during
operation and provide the
substrate for metacognition and
self-directed learning.

Extends the meta-shape concept from
PSN-1. Self-shapes are persistent across
tasks; meta-shapes were transient.

Social shape A shape encoding a model of New — no PSN-1 equivalent. Required for

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
another agent — their goals,
beliefs, capabilities, and likely
actions. Social shapes are built
from observed reasoning traces
and support theory-of-mind
reasoning.

social learning and cooperative behaviour.

2.1 Elimination of the Separate Field Vector
PSN-1 used a field vector F_t = (focus, curiosity, caution, urgency, play, consistency, aesthetic)
as a global modulator. PSN-2 eliminates this. Each field dimension is replaced by a specific
emotional or self-shape mechanism:

PSN-1 Field
Dimension

PSN-2 Replacement Mechanism

focus Attention-narrowing bond
from self-shape to active
task shape

A regulatory bond from the current self-shape
to the task shape increases the commitment
threshold for competing shapes, narrowing the
effective attention manifold through normal
bond mechanics.

curiosity Curiosity engine
generates explicit
exploration goals (Section
7)

Not a bias — an active goal-generating
process. High prediction error triggers a
curiosity event that creates a structured goal
shape and schedules future exploration.

caution Fear/uncertainty
emotional shape bonding
to hypothesis shapes

A fear emotional shape bonds to high-risk
hypothesis shapes and raises their
commitment threshold through the emotional
bond mechanics (Section 5). This is local and
proportional, not global.

urgency Urgency emotional shape
bonding to goal shapes

An urgency emotional shape bonds to active
goal shapes and reduces their pulse budget
tolerance, causing earlier commitment under
pressure.

play Novelty-seeking emotional
shape with low regulatory
bonds

A curiosity-variant emotional shape that
relaxes regulatory bond constraints on shape
composition, allowing unusual recombinations
within legality bounds.

consistency Self-shape regulatory
bonds on policy-violating
compositions

The self-shape carries a regulatory bond to its
identity constraints. Shapes that violate these
constraints receive inhibitory bonds from the
self-shape, raising their suppression
probability.

aesthetic Aesthetic self-shape with
harmonic geometry
preferences

An aesthetic self-shape encodes preferences
in frequency and occupancy geometry. It
bonds to candidate output shapes and biases

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

selection toward shapes that match aesthetic
constraints.

1. System 1: Unified Shape Language (USL)
The USL eliminates the language-reasoning separation by making language a
native shape class. Words are atomic shapes. Phrases are composed shapes.
Sentences are bond-structured shape assemblies. Understanding language means
inducing shapes from linguistic input. Generating language means recovering the
surface form of committed shapes. The same substrate, the same algebra, the
same operations.
3.1 Word Shapes — Atomic Linguistic Primitives
Every word in the vocabulary has a canonical VSA hypervector learned during developmental
stage D4. This is not a token embedding — it is a shape with full node-state structure including
semantic, syntactic, and pragmatic geometry.
word_shape(w) = {
semantic_v: R^D // VSA vector in semantic codebook space
syntactic_v: R^D_s // VSA vector in syntactic codebook (POS, dependency
role, argument structure)
pragmatic_v: R^D_p // VSA vector in pragmatic codebook (register, force,
politeness, topic/focus)
freq_v: R^16 // frequency geometry: prosodic weight, rhythm class
phi_i: float // morphogenic potential (starts high for polysemous
words)
}
Word shapes are stored in a dedicated Lexical Attractor Library (LAL), a subregistry of the main
attractor codebook. Retrieval is VSA cleanup projection exactly as for non-linguistic attractors.
3.2 Phrase Composition — Shape Algebra on Words
Phrases are composed from word shapes using the standard shape algebra operators.
Grammatical relations are encoded as typed bonds.
// &quot;the red ball&quot; as a composed shape:
det_shape = word_shape(&quot;the&quot;)
adj_shape = word_shape(&quot;red&quot;)
noun_shape = word_shape(&quot;ball&quot;)
// Compose: bind adjective to noun via MODIFIES bond
mod_bond = bind(perm_MODIFIES(adj_shape.semantic_v), noun_shape.semantic_v)
// Compose: bind determiner via DETERMINES bond
det_bond = bind(perm_DETERMINES(det_shape.semantic_v),
noun_shape.semantic_v)
// Bundle into noun phrase shape:
NP_shape = bundle({noun_shape.semantic_v, mod_bond, det_bond} ; {1.0, 0.8,
0.6})

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
// NP_shape is now a standard PSN shape — same substrate, same operations
Bond types for linguistic relations are added to the bond type vocabulary (closed set, frozen at
P0):

Linguistic Bond Type Semantic Class VSA Permutation Index
MODIFIES Adjective or adverb modifies a

head word

P_10
DETERMINES Determiner scopes over noun

phrase

P_11
SUBJECT_OF NP is the subject argument of

a predicate

P_12
OBJECT_OF NP is the object argument of a

predicate

P_13

PREDICATE_OF VP or predicative element
relates to subject

P_14

DISCOURSE_CONTIN
UES

Clause continues the
discourse topic of prior clause
P_15

DISCOURSE_CONTRA
STS

Clause introduces contrast
with prior discourse

P_16

DISCOURSE_GROUN
DS

Clause provides background
or reason for another

P_17

COREFERS_WITH Two NPs refer to the same

entity

P_18
SCOPES_OVER Quantifier or operator takes
scope over expression

P_19

3.3 Bidirectional Language Codec
The USL codec operates in two directions using the same VSA operations. There is no
asymmetry between understanding and generation.

Understanding Direction (Text → Shapes)
// Input: raw text string
// Step 1: Tokenise and retrieve word shapes from LAL
word_shapes = [LAL.retrieve(w) for w in tokenise(text)]
// Step 2: Induce syntactic bonds via lightweight dependency parser
// (parser is a small learned module producing bond proposals)
bond_proposals = syntax_encoder(word_shapes) // outputs (i, j, bond_type,
strength) tuples
// Step 3: Run Phase B (bond update) on the word shape field
// This uses the standard VSA bond mechanics — no separate NLP pipeline

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
linguistic_shapes = Phase_B(word_shapes, bond_proposals)
// Step 4: Feed linguistic shapes into Phase A evidence integration as proximal
input
// They are now first-class shapes in the PSN-Core field
PSN_Core.Phase_A(linguistic_shapes)
Generation Direction (Shapes → Text)
// Input: committed canonical shape (verified, in render buffer)
// Step 1: Project shape centroid_nu onto LAL via VSA cleanup
word_candidates = LAL.top_k_cleanup(committed_shape.centroid_nu, k=20)
// Step 2: Resolve syntactic ordering via syntactic_v geometry
ordered_words = syntax_decoder(word_candidates,
committed_shape.syntactic_bonds)
// Step 3: Apply pragmatic constraints from pragmatic_v geometry
// (register, politeness, focus — these come from self-shape and
context)
surface_text = pragmatics_renderer(ordered_words, self_shape.pragmatic_v)
// Step 4: Verifier checks: surface_text must not assert content absent from
committed_shape
if verifier.surface_grounded(surface_text, committed_shape): emit(surface_text)
else: trigger Phase F veto and revision

3.4 Inner Speech Loop (ISL)
During Recursive regime, the system can generate draft verbal propositions, re-encode them as
linguistic shapes, and use them as additional evidence. This is the cognitive function of inner
speech — language as a tool for structuring thought.
// Triggered when: regime == Recursive AND uncertainty &gt; 0.55 AND
pulse_remaining &gt; B_max/4
if ISL_conditions_met:
// Generate draft proposition from current best candidate shape:
draft_text = USL_generate(best_candidate_shape, mode=&quot;tentative&quot;)
// Re-encode as linguistic shapes:
draft_shapes = USL_understand(draft_text)
// Write to Working memory with source tag &quot;inner_speech&quot;:
ERS.write(Working, {shapes: draft_shapes, source: &quot;inner_speech&quot;, utility:
0.0})
// Feed back into Phase A as modulatory input (not proximal — it is not
observation):
PSN_Core.Phase_A_modulatory(draft_shapes)
// The system now reasons about its own candidate — this drives refinement
Inner speech is a modulatory signal, not a proximal observation. It cannot override
external evidence. It can only influence hypothesis refinement by adding the
system&#39;s own tentative verbal representation as a reasoning constraint. This
prevents the system from &quot;talking itself into&quot; conclusions not supported by actual
evidence.

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
4. System 2: Spiking Computation Engine (SCE)
The SCE makes event-driven sparse computation the default, not an optimisation.
Nodes fire only when their prediction error exceeds a threshold. Silent nodes
consume no compute. On familiar inputs, 90%+ of nodes are silent. Inference
energy scales with task novelty, not model size. This is the energy efficiency story.
4.1 Spike Decision Rule
Every node evaluates a spike decision at the start of each pulse phase. If the node does not
spike, it does not participate in that phase&#39;s computation.
// Spike decision for node i at pulse t, phase P:
spike_i(t,P) = 1 iff e_i^scalar(t) &gt; tau_spike(P, sigma_i, b_i)
// tau_spike is phase-dependent and node-type-dependent:
tau_spike(Perceptive, sigma_i, b_i) = 0.30 + 0.10 *b_i[budget_fraction_used]
tau_spike(Compositional,sigma_i,b_i) = 0.20 + 0.05* b_i[budget_fraction_used]
tau_spike(Recursive, sigma_i, b_i) = 0.10 // low threshold — explore widely
tau_spike(Commit, sigma_i, b_i) = 0.40 // high threshold — only committed
shapes fire
// Surrogate gradient for training (straight-through estimator):
d_spike_i/d_e_i = 1 / (1 + |e_i^scalar - tau_spike|)^2 // smooth surrogate

4.2 Silent Node Behaviour
A silent node (spike_i=0) takes no action in phases A-E. In Phase F it contributes passively to
coalition stability through its last committed state. It decays its temporal trace:
// Silent node update (all phases):
tau_i(t+1) = lambda_tau *tau_i(t) // trace decays: lambda_tau = 0.90
nu_i(t+1) = nu_i(t) // prediction unchanged
e_i(t+1) = e_i(t)* lambda_e_silent // error decays slowly:
lambda_e_silent = 0.95
// No bond updates, no role transitions, no shape formation participation
// Cost: 1 multiply per node per pulse (trace decay) — negligible

4.3 Population-Level Sparsity

Input Type Expected Spike

Rate

Pulses Active Energy Relative to

Dense

Highly familiar pattern (seen
50+ times)

2–8% of nodes 1–2 pulses
(Perceptive regime)

~1–3%

Familiar pattern with variation 8–20% of nodes 2–4 pulses
(Compositional)

~5–12%

Novel pattern, clear structure 20–40% of nodes 4–8 pulses
(Recursive)

~15–30%
Truly novel — no attractor 40–70% of nodes 8–32 pulses (deep ~40–80%

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
match Recursive)
Contradiction requiring full
revision

50–80% of nodes up to B_max pulses ~60–100%

The key property: the system automatically spends energy proportional to task difficulty. This is
not a fixed compute budget divided across tasks — it is a dynamic allocation driven by
prediction error. A system that understands a question deeply uses almost no energy to answer
it. Only genuinely hard, novel tasks use full compute.
4.4 Neuromorphic Deployment Target
The SCE is designed to compile to neuromorphic hardware from the start. This is an
architectural constraint, not a deployment afterthought.

Operation Neuromorphic Implementation Target Hardware
Node spike decision Threshold comparator — single clock

cycle per node

Intel Loihi 2, IBM NorthPole

Bond update (VSA bind) Bitwise XOR on bipolar hypervectors —

massively parallel

Loihi 2 synaptic cores

Phase B (bond
negotiation)

Spike-timing-dependent plasticity on
VSA edges

Loihi 2 on-chip learning

Codebook cleanup
(VSA)

Dot-product array — dedicated
hardware in NorthPole

IBM NorthPole inference

Phase D (shape
formation)

Coincidence detection across spike
patterns

Custom ASIC or Loihi mesh

Memory retrieval
(attractor)

Associative memory — content-
addressable RAM

Hopfield-on-chip or Loihi
Temporal motif replay Pre-programmed spike pattern injection Loihi 2 spike generators

GPU training remains the reference implementation for correctness during stages
D1–D6. Neuromorphic deployment is the production target. The training system
must produce checkpoints that compile to neuromorphic hardware without
architectural changes. This means: all operations must be expressible as spike
timing and synaptic weight updates. The VSA basis (bipolar hypervectors) is
already neuromorphic-compatible.

1. System 3: Emotional Shape System (ESS)
Emotion is not decoration. In biological systems, emotional signals are information
— they encode the system&#39;s evaluation of its own state relative to its goals. Fear

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
signals threat to a goal. Curiosity signals an opportunity to reduce uncertainty.
Confidence signals readiness to commit. These signals must participate in
reasoning directly, through the bond mechanics, not through a global modulator that
blindly biases all computation equally.
5.1 Emotional Shape Vocabulary

Emotional Shape Trigger Condition Bonds To Effect on Bonded

Shapes

Threat-fear Goal shape at risk:
e_i^scalar on goal nodes
&gt; 0.60

Goal shapes,
hypothesis shapes
that contradict the
goal

Raises commitment
threshold of bonded
hypothesis shapes by
+0.20; increases
caution-regulatory
bonds on action shapes

Uncertainty-
discomfort

e_mean &gt; 0.50 AND no
committed shape after
B_max/2 pulses

Meta-shapes,
hypothesis frontier

Increases pulse budget
allocation to bonded
hypotheses; activates
ISL to generate draft
propositions

Curiosity-drive Novel input: no attractor
match within cos &lt; 0.45

Unexplored shape-
space regions, CE
goal generator

Generates exploration
goal (Section 7); lowers
tau_spike for nodes in
the unexplored region

Confidence Committed shape with
stability &gt; 0.85 AND
verifier pass

Render shapes,
action shapes

Lowers render
commitment threshold
by -0.15; opens action
gate faster

Urgency Goal deadline encoded in
goal shape OR external
interrupt

All active shapes in
task graph

Raises tau_dissolve by
+0.10 (faster pruning);
raises cost(Recursive)
by +1 (shorter budget)

Satisfaction Task completed with
verifier pass AND high
utility_score

Self-shape, ERS write Increases utility_score
multiplier for current
episode by +0.30;
reinforces the
reasoning trace used

Frustration Repeated verifier veto (&gt;=
3 vetoes on same
hypothesis)

The vetoed
hypothesis shape,
ISL trigger

Suppresses the
hypothesis shape;
triggers ISL to generate
alternative framing;
logs to ERS as
negative example

Aesthetic-resonance Generated shape has high
harmonic geometry
coherence

Render shape in
language or music
tasks

Biases word-shape
selection toward high-
frequency-geometry
matches; produces

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

more rhythmically
coherent output

5.2 Emotional Bond Types
Emotional bonds are a new bond type sub-class added to the bond vocabulary. They operate
on the same VSA mechanics as all other bonds.
// Emotional bond from emotional shape E to target shape S:
emotional_bond_ES = bind(perm_EMO(E.semantic_v), S.centroid_nu)
// Effect on S&#39;s commitment threshold (additive):
delta_theta_commit(S) = SUM_{E bonded to S} strength_ES *effect_E(S)
// effect_E is the signed scalar effect defined per emotional type above
// e.g., threat-fear bonded to hypothesis H raises theta_commit(H) by +0.20*
strength
Emotional bond strength decays by lambda_emo = 0.85 per pulse (faster than cognitive bonds
at 0.90). This ensures emotional influences are time-limited and do not permanently bias the
system toward states triggered by transient events.
5.3 Emotional Shape Induction
Emotional shapes are not externally assigned. They are induced by the same Phase C
morphogenic mechanism as all other shapes, using a dedicated emotional shape codebook in
the attractor library.
// During Phase C, alongside role and type transitions:
emo_pressure = W_emo *concat(e_i^scalar, goal_proximity, budget_fraction,
veto_count, novelty_score)
emo_logits = W_emo_head* GELU(W_morpho * emo_pressure + b_m) + b_emo
emo_dist = softmax(emo_logits / tau_emo) // soft distribution over 8
emotional types
// Dominant emotional shape forms when argmax(emo_dist) stable for &gt;= 3 pulses
// Emotional shapes persist in Working ERS tier; not cleared between pulses

1. System 4: Curiosity Engine (CE)
Curiosity in PSN-2 is not a field bias. It is an active process that monitors prediction
error across the substrate, identifies regions of persistent uncertainty, generates
structured internal goals to return to those regions, and schedules future
exploration. This is how human expertise develops — not by being trained on
everything, but by being drawn to the edges of understanding and systematically
reducing uncertainty there.
6.1 Curiosity Event Detection
// A curiosity event is triggered when:
curiosity_event = (

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
e_i^scalar &gt; tau_curio_error = 0.50 // high prediction error at some
node
AND dl_i &gt; tau_curio_dl = 2.5 // description length high (no
motif match)
AND b_i[pulse_remaining] &lt; B_max *0.30 // budget nearly exhausted — cannot
resolve now
AND node_i not in any committed shape // the error is not already
resolved
)
// Multiple curiosity events in one episode are merged by VSA bundle of their
nu_i vectors
6.2 Curiosity Goal Generation
When a curiosity event is detected, the CE generates a structured Curiosity Goal Shape and
stores it in the Episodic ERS tier with high utility_score.
// Curiosity goal shape construction:
curiosity_target_v = bundle({nu_i : i in curiosity_event_nodes} ; {e_i^scalar})
curiosity_goal = {
shape_class: &quot;goal&quot;,
goal_type: &quot;uncertainty_reduction&quot;,
target_VSA: curiosity_target_v, // VSA encoding of the unresolved
region
priority: mean(e_i^scalar for i in curiosity_event_nodes),
task_family: current_task.task_family,
generated_at: episode_id,
scheduled_for: &quot;next_available_episode_in_family&quot;,
budget_reserve: B_max* 0.50, // reserve more budget for
exploration
}
ERS.write(Episodic, curiosity_goal, utility_score=0.80)

6.3 Curiosity Goal Scheduling and Execution
At the start of each new episode, the CE checks the Episodic ERS for pending curiosity goals
whose target_VSA has high cosine similarity with the current input.
pending_curiosity = ERS.retrieve(current_input_VSA,
goal_type=&quot;uncertainty_reduction&quot;, k=3)
if len(pending_curiosity) &gt; 0 AND highest_priority &gt; tau_curio_priority = 0.40:
// Load the curiosity goal as a pre-committed meta-shape in the task graph
task_graph.add(curiosity_goal_shape)
// Reserve extra pulse budget for exploration:
b_i[pulse_remaining] += curiosity_goal.budget_reserve // additive on top of
normal B_max
// Lower tau_spike for nodes in the curiosity target region:
for i in nodes_near(curiosity_target_v, cos_threshold=0.50):
tau_spike_i_override = tau_spike(regime) * 0.60 // these nodes fire more
readily

6.4 Curiosity Resolution and ERS Update
// At episode end, if the curiosity goal was addressed:

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
if cos(committed_shapes_centroid, curiosity_goal.target_VSA) &gt; 0.65:
// Goal resolved: update utility_score and promote
curiosity_goal.utility_score += 0.20 // bonus for successful resolution
ERS.promote(curiosity_goal, Semantic) // resolved curiosity becomes semantic
knowledge
else:
// Goal unresolved: reschedule with higher priority
curiosity_goal.priority += 0.10
curiosity_goal.budget_reserve *= 1.25 // allocate more budget next attempt

1. System 5: Social Learning System (SLS)
Humans learn an enormous amount by watching others. Not just copying outputs —
reconstructing the reasoning process. PSN-2 can observe another agent&#39;s
reasoning trace (a sequence of shapes, bonds, and decisions) and encode it as if
the system had produced it. This is imitation at the reasoning level, not the
behaviour level.
7.1 Social Shape Construction
When observing another agent, PSN-2 builds a Social Shape — a model of the observed
agent&#39;s goals, beliefs, and reasoning style.
social_shape = {
agent_id: UUID or hash of observed agent,
goal_VSA: VSA centroid of observed goal shapes,
reasoning_style: VSA centroid of observed hypothesis formation pattern,
competence: dict(task_family -&gt; observed success rate),
trust: float in [0,1], // updated by verifier agreement rate
shape_class: &quot;social&quot;,
}
Social shapes are stored in the Semantic ERS tier with decay_state initialized at 0.80 (trusted
source) or 0.40 (unknown source). They are retrieved by agent_id similarity or by goal_VSA
similarity when the current task resembles tasks the observed agent has solved.
7.2 Trace Reconstruction Protocol
To learn from an observed trace, PSN-2 reconstructs the trace as a shape sequence and stores
it in ERS.
// Observed trace = sequence of (input, intermediate_steps, output) from
another agent
// Step 1: Encode each step as a shape using USL or domain encoder:
observed_shapes = [PSN_Core.encode(step) for step in observed_trace]
// Step 2: Infer bonds between steps (causal, temporal, argumentative):
inferred_bonds = bond_inducer(observed_shapes) // same as Phase B on observed
data
// Step 3: Construct a synthetic ERS tuple:
synthetic_tuple = {
input_VSA: observed_shapes[0].centroid_nu,

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
trace_VSA: bundle(observed_shapes[1:-1] ; uniform_weights),
output_VSA: observed_shapes[-1].centroid_nu,
source: &quot;social_observation&quot;,
agent_id: observed_agent.agent_id,
trust: social_shape.trust,
utility_score: social_shape.trust * verifier_check(observed_output),
}
// Step 4: Write to Episodic ERS with source tag:
ERS.write(Episodic, synthetic_tuple)

7.3 Trust and Verification
Social learning is only as good as the source. PSN-2 applies verification to all socially observed
traces before storing them.
• The verifier checks the observed output against the observed intermediate shapes. If the
output is not supported by the intermediate reasoning, the trace is stored with
utility_score = -0.20 and source = &quot;suspicious&quot;.
• Trust updates after each observation: if the verifier agrees with the observed conclusion,
social_shape.trust += 0.05. If the verifier disagrees, social_shape.trust -= 0.15. Trust is
bounded in [0.05, 1.0].
• Traces from agents with trust &lt; 0.30 are stored in Working tier only and not promoted to
Episodic.
• The system can observe its own past traces as social observations — this implements
self-distillation. The system learns from its own verified successful episodes as if they
were high-trust social observations.

1. System 6: Temporal Abstraction Engine (TAE)
Temporal abstraction is how experts become fast. A chess grandmaster sees a
familiar position and instantly knows the right move — not because they re-derive it,
but because the sequence is stored as a compressed temporal pattern. PSN-2
implements this through temporal motif shapes that encode frequently executed
pulse sequences and allow them to be replayed in Perceptive regime with near-
zero compute cost.
8.1 Temporal Motif Detection
The TAE monitors pulse sequences for patterns that recur across multiple episodes. A temporal
motif candidate is nominated when a pulse sequence meets three conditions:
temporal_motif_candidate(seq) =
len(seq) &gt;= 3 pulses // minimum meaningful sequence
AND seq_VSA_similarity(seq_i, seq_j) &gt; 0.75 // structurally similar across
episodes
AND recurrence_count(seq) &gt;= 5 // appeared in at least 5
distinct episodes

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
The TAE represents a temporal motif as a shape in the directional geometry:
temporal_motif_shape = {
pulse_sequence_VSA: R^D, // VSA bundle of all shape centroids in the
sequence
ordering_encoding: R^D_dir, // directional geometry encoding of pulse
order
duration: int, // number of pulses in the canonical sequence
trigger_VSA: R^D, // VSA of the input that triggers this
sequence
outcome_VSA: R^D, // VSA of the typical outcome
success_rate: float, // fraction of retrievals that led to verifier
pass
pulse_savings: float, // mean pulses saved vs executing step-by-step
shape_class: &quot;temporal_motif&quot;,
}
8.2 Temporal Motif Retrieval and Replay
// At Phase A of a new episode:
input_VSA = encode(current_input)
matching_motifs = TAL.top_k_cleanup(input_VSA, k=4) // TAL = Temporal
Attractor Library
for motif in matching_motifs:
if cos(input_VSA, motif.trigger_VSA) &gt; tau_motif = 0.65:
if motif.success_rate &gt; 0.70:
// High-confidence match: replay in Perceptive regime
replay_result = TAE.replay(motif) // injects pulse sequence as pre-
formed shapes
if verifier.preliminary_check(replay_result.outcome) == &quot;plausible&quot;:
// Skip to Phase F with replay_result as candidate
regime = Commit
candidate = replay_result
break
else:
// Replay failed: fall back to normal pulse cycle
TAE.update_success_rate(motif, success=False)
When a temporal motif replay succeeds, the pulse savings are recorded and fed back into the
motif&#39;s pulse_savings field. Motifs with high pulse_savings are prioritised for retrieval. This
creates a positive feedback loop: frequently used, reliable temporal motifs get faster and
cheaper over time — exactly the behaviour of human habits and expertise.
8.3 Temporal Motif Promotion Gates
Temporal motifs must pass a simplified four-gate evaluation (compared to the eight-gate system
for durable motifs) because they represent procedural sequences rather than relational
abstractions:

Gate Threshold What It Prevents
Recurrence gate &gt;= 10 distinct Ephemeral patterns that happened to repeat but

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

episodes are not genuine habits

Success rate gate &gt;= 0.75 verifier pass
rate during 1000-
episode quarantine

Unreliable patterns that sometimes work but often
fail — worse than re-computing

Pulse savings gate &gt;= 3 pulses mean
savings vs step-by-
step execution

Motifs that are not actually faster than normal
processing

Interference gate &lt;= 0.03 degradation
on protected task
families

Motifs that interfere with flexible reasoning by
prematurely committing to a cached pattern

1. Developmental Curriculum (DC)
The developmental curriculum is not a training schedule — it is an ontological
ordering. Each stage builds the representational foundations that the next stage
requires. Language cannot be grounded if causal structure is absent. Social
cognition cannot develop if object permanence is absent. Abstract reasoning cannot
generalise if it is not grounded in causal and social schemas. Violating this order
produces a system that performs well on benchmarks it was trained on and fails on
everything else. This is the LLM failure mode.
9.1 The Six Developmental Stages

Stag
e
Name What is Built Primary Data Gate to Next
Stage

D1 Sensorimotor
Grounding

Object permanence, spatial
reasoning, basic causal
perception, temporal trace
binding. The substrate learns
that the world has persistent
objects that move according to
laws. No language. No
abstract relations. Pure
sensorimotor prediction.

Synthetic physics
environments:
rolling, colliding,
occluding
objects.
Continuous video
streams. No text.
Metric and
occupancy
geometry trained.

Object tracking
accuracy &gt;=
0.90 across
occlusion
events. Causal
prediction error
&lt; 0.20 on held-
out physics
episodes.
Temporal trace
persistence &gt; 5
pulses for
tracked objects.

D2 Causal and Relational
Grounding

Cause-effect schemas,
precondition-action-
consequence chains, basic
logical relations (if-then, part-
whole, same-different). The

Causal
intervention
datasets: agent
acts on objects,
observes

Causal
intervention
accuracy &gt;=
0.80 on held-
out intervention

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

substrate learns that events
have structure beyond
temporal co-occurrence.

outcomes.
Abstract
relational tasks
(same/different,
analogy).
Synthetic causal
graphs with
interventions.
Directional
geometry trained.

tasks. Abstract
analogy score
&gt;= 0.75. VSA
bond recall for
causal relation
types &gt;= 0.90.

D3 Social and Theory-of-
Mind Grounding

Agent models, goal inference,
belief attribution (agent A
believes X even though X is
false), social causal schemas
(agent acts because of goal).
Social shapes and social
bonds trained.

Agent
observation
datasets: videos
of agents
pursuing goals in
environments.
False-belief
tasks.
Cooperative and
competitive
agent interaction
traces. Social
bond types
trained.

False-belief
task accuracy
&gt;= 0.75. Goal
inference from
partial
observation &gt;=
0.70. Social
shape trust
calibration:
predicted vs
actual agent
success rate
RMSE &lt; 0.15.

D4 Linguistic Grounding Language as shape-bundles
grounded in D1-D3 structure.
Word shapes acquire
semantic, syntactic, and
pragmatic geometry. The USL
codec is trained. Inner speech
loop activated. Language is
learned as a code for already-
understood structure, not as a
statistical surface.

Text paired with
scene
descriptions,
causal
explanations,
agent action
narratives.
Grounded
language only —
no ungrounded
text corpora. USL
understanding
and generation
trained jointly.
LAL populated.

USL round-trip
fidelity &gt;= 0.85
(encode →
decode
recovers
original shape).
Language-
grounded
analogy &gt;=
0.80. ISL
produces
coherent inner
speech on
70%+ of
Recursive
regime
episodes.

D5 Abstract Reasoning
and Formal
Competence

Mathematical reasoning,
logical inference, formal
argument structure,
counterfactual reasoning,
multi-step planning. Built on
D2 causal and D4 linguistic
grounding. The system can
now reason about abstract
objects using language as a
cognitive tool.

Verifier-backed
math and logic
traces. Formal
planning tasks.
ARC-AGI-3
interactive
reasoning
environments.
Counterfactual
simulation tasks.
All PSN-1 P6-
P11 training data
introduced here.

ARC-AGI-3
RHAE
improvement
over D4
baseline. Math
verification rate
&gt;= 0.92. Multi-
step planning
success &gt;=
0.80 on held-
out task
families.
Compositional

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

split score &gt;=
0.75.

D6 Full Integration and
Meta-Learning

All systems active
simultaneously: USL, SCE,
ESS, ISL, CE, SLS, TAE,
DSLE, ERS. Meta-learning
curriculum (FSGC from PSN-1
v1.1) runs across all task
families. Temporal motifs form
for common reasoning
patterns. Curiosity goals drive
self-directed learning.

Full mixed
curriculum: all
previous data
types, language
interface data,
tool use traces,
social learning
traces, multi-
modal tasks. All
24 loss
components
active.

Sample
Complexity
Ratio &gt;= 10x on
3/5 held-out
families.
Temporal motif
pulse savings
&gt;= 3 pulses on
familiar task
families.
Curiosity
resolution rate
&gt;= 0.60 on
generated
curiosity goals.
All release
gates from
PSN-1 v1.0 and
v1.1 pass.

Stage gating is absolute. A checkpoint that passes D3 gates but has not
completed D2 training is invalid. The gates must be certified in order.
Running D4 language training before D2 causal grounding is certified
produces the LLM failure mode: language without understanding. The
training infrastructure must enforce stage ordering at the data-loader level,
not by convention.

1. Unified Loss Family (PSN-2 — 26 Components)
PSN-2 carries forward all 22 loss components from PSN-1 v1.0 and v1.1 HEL. Four new
components are added. Weights are frozen at D1 stage start and may not drift during training.

Component Symbol Weight What It Trains Stage
Active

Prediction error L_error w1=0.12 Multi-step prediction-error
minimization (core learning law)

D1+

Bond update L_bond w2=0.08 VSA bond accuracy and typed

relation supervision

D1+
Shape stability L_shape w3=0.08 Shape stability and dissolution hinge

losses

D1+

Compactness L_compact w4=0.12 Expected description length —

dominant

D1+

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
Refinement L_refine w5=0.07 Negative per-pulse stability

improvement

D2+

World model L_world w6=0.07 World model MSE and counterfactual

accuracy

D2+

Verification L_verify w7=0.09 Verifier veto classification +
unsupported-claim loss

D2+
Calibration L_calib w8=0.06 Error calibration vs actual correctness D2+
Transfer L_transfer w9=0.07 Efficiency delta with vs without motifs D5+
Render L_render w10=0.03 Surface generation from committed

state

D4+
VSA algebra L_vsa w11=0.06 VSA binding recovery and cleanup

accuracy

D1+
MPF stability L_mpf w12=0.04 phi low on solved tasks D1+
Attractor separation L_attractor w13=0.03 MPF basins well-separated D1+
Weight symmetry L_sym w14=0.04 W_ff/W_fb cosine symmetry floor D1+
Attention quality L_attn w15=0.04 External attention retrieval quality D4+
Expression quality L_expr w16=0.04 Internal expression causal quality D4+
Geometry consistency L_geom w17=0.03 Cross-geometry consistency D1+
Compartment L_compart w18=0.03 Compartment disentanglement D1+
Fast adaptation L_fast_adapt w19=0.08 Meta-episode query performance

given support-set

D6
Retrieval utility L_retrieval w20=0.05 ERS utility score prediction accuracy D5+
Compositional split L_comp_spli

t

w21=0.07 Performance on held-out
compositional test instances

D5+
Forgetting L_forget w22=0.06 EWC forgetting penalty during online

updates

D6

Linguistic grounding L_linguistic w23=0.07 USL round-trip fidelity:
encode→decode→encode
consistency. Prevents language
shapes from drifting away from non-
linguistic semantic grounding.

D4+

Emotional calibration L_emo_calib w24=0.05 Emotional shape induction accuracy:
predicted emotional state vs. post-
hoc outcome-based ground truth (did
caution prevent a bad commitment?
did curiosity resolve a gap?)

D3+

Social learning fidelity L_social w25=0.05 Verifier agreement rate on socially
observed traces: the system&#39;s
reconstruction of an observed agent&#39;s
reasoning must be verifier-consistent
D3+

Temporal motif L_temporal w26=0.06 Pulse savings from temporal motif D5+

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
efficiency replay: penalises motifs that do not
save at least 3 pulses vs step-by-step
execution

// PSN-2 total loss:
L_total_psn2 = SUM_{i=1}^{26} w_i * L_i / normalization_psn2
normalization_psn2 = SUM w_i = 1.75 (v1.1) + 0.07 + 0.05 + 0.05 + 0.06 = 1.98

1. Evaluation Framework
PSN-2 is evaluated against a five-dimensional scorecard. A system that excels on four
dimensions but fails on one has not met the PSN-2 specification.

Dimension Primary Question Benchmarks Release Gate
Reasoning
Integrity

Does the system reason
correctly and verifiably on
structured tasks?

ARC-AGI-3 (RHAE),
BIG-Bench Hard,
mathematical reasoning
suites, multi-step
planning tasks, verifier
stress tests

RHAE positive on all 6
environment classes. False-
commit rate &lt; 0.5%. Verifier
pass rate &gt; 95% on
structured task families.

Language
Quality

Does the system produce
fluent, grounded,
pragmatically appropriate
language?

Human evaluation of
naturalness, coherence,
and appropriateness.
Grounding test: does
generated text assert
only verified content?
USL round-trip fidelity.

Human rater score &gt;=
4.0/5.0 on naturalness.
Grounding violation rate &lt;
1%. USL round-trip fidelity
&gt;= 0.85 on held-out
concepts.

Learning
Efficiency

Does the system learn
from few examples and
transfer across tasks?

Sample Complexity
Ratio vs matched
transformer. Few-shot
efficiency curves
(k=1,3,5). CompSplit
score. Online
adaptation over 10
episodes.

SCR &gt;= 10x on 3/5 held-out
families. k=1 performance
&gt;= 40% of full-data.
CompSplit &gt;= 0.75.

Energy
Efficiency

Does inference energy
scale with task difficulty?

Mean sparsity (% silent
nodes) by task type.
Energy per inference vs
baseline transformer.
Neuromorphic
deployment energy on
Intel Loihi.

Mean sparsity &gt;= 80% on
familiar tasks. Energy per
inference &lt;= 5% of matched
transformer on familiar
inputs. Loihi deployment: &lt;
1mJ per familiar inference.

Cognitive Unity Does the system behave
as one unified system
rather than patched
components?

Ablation of each system
(USL, ESS, ISL, CE,
SLS, TAE) must
degrade multiple

Every system ablation must
degrade at least 2
evaluation dimensions. No
system ablation may

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification

dimensions
simultaneously — not
just its target dimension.
A unified system has
cross-system
dependencies.

improve any dimension
(ruling out harmful
interference).

11.1 The Human Parity Test
The ultimate acceptance criterion for PSN-2 is the Human Parity Test: a matched human expert
and PSN-2 are given the same novel tasks across all five dimensions. PSN-2 must reach &gt;=
85% of human expert performance on each dimension individually. This is not a single
benchmark score — it is a profile test that cannot be gamed by excelling in one dimension.

Dimension Human Expert Baseline PSN-2 Target Measurement Method
Reasoning PhD-level reasoning on
novel structured problems

&gt;= 85% of human
expert solve rate

Novel ARC-AGI-3
environments not in
any training set

Language Native speaker fluency
and coherence

&gt;= 85% human rater
agreement on
naturalness and
appropriateness

Blind A/B evaluation:
PSN-2 vs human on
same prompts

Learning speed 1–3 examples to grasp a

new concept

Within 2x of human k-
shot performance

Head-to-head few-shot
learning on novel
concept categories

Energy ~20W continuous
operation

&lt;= 50W at full load on
neuromorphic
hardware

Power meter on Loihi
deployment during
benchmark run

Robustness Does not catastrophically
forget old skills when
learning new ones

Protected-family
regression &lt;= 2%
after learning 10 new
task families

Sequential task family
acquisition with
protected-family testing
after each

1. New Default Constants (PSN-2 — Frozen at D1 Start)

Constant Value Description
tau_spike(Perceptive) 0.30 +
0.10*budget_fr
action

Spike threshold in Perceptive regime

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
tau_spike(Compositional) 0.20 +
0.05*budget_fr
action

Spike threshold in Compositional regime

tau_spike(Recursive) 0.10 Spike threshold in Recursive regime — low to

maximise exploration

tau_spike(Commit) 0.40 Spike threshold in Commit regime — high, only

committed nodes fire

lambda_e_silent 0.95 Error decay rate for silent nodes per pulse
tau_curio_error 0.50 Prediction error threshold for curiosity event

detection

tau_curio_dl 2.5 Description-length threshold for curiosity event

detection

tau_curio_priority 0.40 Minimum priority for curiosity goal to modify next

episode budget

tau_motif 0.65 Cosine similarity threshold for temporal motif

retrieval

tau_motif_success 0.75 Minimum success rate for temporal motif

promotion

tau_motif_savings 3 Minimum pulse savings for temporal motif

promotion (pulses)

lambda_emo 0.85 Emotional bond decay rate per pulse (faster

than cognitive bonds)

tau_emo_stable 3 Pulses before a dominant emotional shape is

considered stable

tau_social_trust_min 0.05 Minimum social shape trust (floor)
tau_social_trust_suspicious 0.30 Trust below which traces go to Working tier only
alpha_icl 0.25 In-context learning bias weight (carried from

v1.1)

alpha_ers 0.15 ERS retrieval bias weight (carried from v1.1)
tau_ers 0.45 ERS retrieval cosine similarity threshold (carried

from v1.1)

LAL_vocab_size 65536 Maximum word shapes in the Lexical Attractor

Library

D_syntactic 512 Dimensionality of syntactic geometry

hypervectors

D_pragmatic 256 Dimensionality of pragmatic geometry

hypervectors

ISL_trigger_uncertainty 0.55 Uncertainty threshold for ISL activation in

Recursive regime

ISL_budget_fraction 0.25 Minimum remaining budget fraction for ISL to

activate

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
ISL_source_tag inner_speech ERS source tag for inner speech shapes — read

by Phase A as modulatory only
w23 (L_linguistic) 0.07 Loss weight for USL round-trip fidelity
w24 (L_emo_calib) 0.05 Loss weight for emotional calibration
w25 (L_social) 0.05 Loss weight for social learning fidelity
w26 (L_temporal) 0.06 Loss weight for temporal motif efficiency
normalization_psn2 1.98 Total loss normalization constant

1. Software Architecture Additions (PSN-2)
All PSN-1 v1.0 and v1.1 modules remain. The following modules are added.
/src/psn2
usl/
lal.py # Lexical Attractor Library: word shape store,
retrieval
codec.py # Bidirectional USL codec: understand() and generate()
syntax_encoder.py # Lightweight dependency parser producing bond
proposals
syntax_decoder.py # Syntactic ordering from shape bonds
pragmatics.py # Pragmatic geometry: register, force, politeness
renderer
isl.py # Inner Speech Loop: trigger, draft generation, re-
encoding
linguistic_bonds.py # 10 linguistic bond types: VSA permutation indices,
legality
sce/
spike.py # Spike decision rule, surrogate gradient, silent node
update
sparsity.py # Population sparsity monitoring, energy accounting
neuromorphic.py # Compilation target: Loihi 2 and NorthPole adapters
ess/
emotional_shapes.py # 8 emotional shape types: induction, bonding, decay
emotional_bonds.py # Emotional bond VSA encoding, delta_theta_commit
computation
emo_field_bridge.py # Compatibility shim: maps legacy field vector calls to
ESS
ce/
curiosity_detector.py # Curiosity event detection: tau_curio_error,
tau_curio_dl
goal_generator.py # Curiosity goal shape construction and ERS write
goal_scheduler.py # Pending curiosity goal retrieval and episode
budget allocation
sls/
social_shape.py # Social shape construction, trust update,
competence model

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
trace_reconstructor.py # Observed trace → synthetic ERS tuple pipeline
self_distillation.py # Self-observation: treat own verified traces as
social data
tae/
motif_detector.py # Temporal motif candidate detection: recurrence,
similarity
motif_replay.py # Temporal motif retrieval and pulse sequence
injection
tal.py # Temporal Attractor Library: store, retrieve,
promote, retire
dc/
stage_d1.py # Sensorimotor grounding curriculum and gate certification
stage_d2.py # Causal and relational grounding
stage_d3.py # Social and theory-of-mind grounding
stage_d4.py # Linguistic grounding (USL training)
stage_d5.py # Abstract reasoning and formal competence
stage_d6.py # Full integration and meta-learning
gate_certifier.py # Stage gate certification: measures all metrics, blocks
progression
/tests/psn2
test_usl_roundtrip.py # USL encode→decode fidelity on held-out concepts
test_linguistic_bonds.py # 10 bond types: VSA encoding, legality, recovery
test_isl.py # Inner speech loop: trigger conditions, re-
encoding
test_spike_sparsity.py # Spike rate by regime and task type
test_neuromorphic_compile.py# Compilation to Loihi spike format
test_emotional_calibration.py # Emotional shape induction and bond effect
test_curiosity_lifecycle.py # Detection → goal → scheduling → resolution
test_social_trust.py # Trust update, verification of observed traces
test_temporal_motif.py # Detection, replay, success rate update
test_developmental_gates.py # All D1–D6 gate metrics, ordering enforcement

1. Build Order for an Autonomous Implementation Agent
Every step must be completed and validated before the next begins. Steps 1–28 from PSN-1
v1.0 apply first. The following steps extend that sequence.
1. Implement the Spiking Computation Engine (sce/spike.py): spike decision rule, surrogate
gradient, silent node update, lambda_e_silent decay. Write sparsity tests: verify &gt;= 80%
silent nodes on synthetic familiar inputs.
1. Implement the Lexical Attractor Library (usl/lal.py): word shape schema with semantic,
syntactic, and pragmatic geometry; store, retrieve, and update operations on LAL
codebook.
1. Implement the 10 linguistic bond types (usl/linguistic_bonds.py): VSA permutation
indices P_10 through P_19; legality matrix entries; round-trip recovery tests for each
bond type.
1. Implement the lightweight syntax encoder (usl/syntax_encoder.py): dependency parser
producing (i, j, bond_type, strength) bond proposals from word shape sequences. Train
on dependency-annotated corpora during D4.

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
5. Implement the USL codec bidirectional pipeline (usl/codec.py): understand() running
Phase B on word shapes + bond proposals; generate() running cleanup projection +
syntax decoder + pragmatics renderer. Implement USL round-trip test: encode then
decode must recover original shape with fidelity &gt;= 0.85.
6. Implement the Inner Speech Loop (usl/isl.py): trigger condition check, draft generation
from best candidate shape, re-encoding as linguistic shapes, modulatory injection into
Phase A. Write tests verifying ISL does not override external evidence.
7. Implement the 8 emotional shape types (ess/emotional_shapes.py): induction via
morphogenic MLP extension, bond formation via perm_EMO, delta_theta_commit
computation per emotional type, lambda_emo decay. Write emotional calibration tests.
8. Implement the field vector compatibility bridge (ess/emo_field_bridge.py): maps all
legacy F_t access calls to the corresponding ESS mechanism. Verify no legacy field
vector code path remains active after bridge is in place.
9. Implement the Curiosity Engine (ce/): detector, goal generator, scheduler, resolution
tracker. Write curiosity lifecycle test: detect event → generate goal → schedule →
resolve → promote to Semantic.
10. Implement the Social Learning System (sls/): social shape construction, trace
reconstructor, trust update, self-distillation. Write trust calibration test: predicted vs actual
agent success rate RMSE &lt; 0.15 on synthetic agents with known success rates.
11. Implement the Temporal Abstraction Engine (tae/): motif detector, Temporal Attractor
Library, motif replay. Write motif detection test: verify recurrence_count tracking and
similarity thresholding. Write motif replay test: verify pulse savings are accurately
measured.
12. Implement developmental stage D1 curriculum (dc/stage_d1.py): synthetic physics
environment data loader, gate certification logic for object tracking, causal prediction,
temporal trace persistence. Run D1 training. Certify D1 gates.
13. Implement D2 curriculum (causal grounding): causal intervention datasets, abstract
analogy tasks, directional geometry training. Certify D2 gates before proceeding.
14. Implement D3 curriculum (social grounding): agent observation datasets, false-belief
tasks, social shape and social bond training. Certify D3 gates including trust calibration.
15. Implement D4 curriculum (linguistic grounding): grounded language training with USL,
LAL population, inner speech loop activation. USL round-trip fidelity gate &gt;= 0.85.
Language-grounded analogy gate &gt;= 0.80. Certify D4 gates.
16. Implement D5 curriculum (abstract reasoning): all PSN-1 P6–P11 training data, ARC-
AGI-3 environments, mathematical reasoning, multi-step planning. Curiosity engine
active. Temporal motif detection begins. Certify D5 gates.
17. Implement D6 curriculum (full integration): all systems active, FSGC meta-learning, SLS
self-distillation, full 26-component loss family. Certify all D6 gates including SCR,
CompSplit, human parity test targets.
18. Profile hot paths across all new modules. Port proven bottlenecks to Triton or CUDA.
Compile SCE spike operations to Loihi-compatible spike format. Validate neuromorphic
energy measurements.
19. Run all test suites. Certify all release gates across all five evaluation dimensions. Issue
signed PSN-2 release manifest.

PSN-2: Unified Intelligence Architecture CONFIDENTIAL · Ideation Labs · April 2026

Version 1.0 · Implementation-Grade Specification
15. Formal Definition

PSN-2 is a Unified Predictive Substrate Network: a post-transformer architecture in
which one substrate of stateful typed nodes, VSA-encoded bonds, and composed
shapes simultaneously implements language (as native linguistic shapes in the
Unified Shape Language, grounded bidirectionally in the same substrate as non-
linguistic reasoning), emotion (as first-class shapes that bond to cognitive shapes
and influence their stability and commitment thresholds through normal bond
mechanics), structured reasoning (through the local prediction-error pulse cycle
inherited from PSN-1), and efficient learning (through the Dual-Speed Learning
Engine, Experience Replay Substrate, and Few-Shot Generalization Curriculum).
Computation is event-driven and sparse: nodes fire only when their prediction error
exceeds a threshold, making inference energy proportional to task novelty.
Development proceeds in the biologically validated order: sensorimotor grounding
precedes causal reasoning, which precedes social cognition, which precedes
language, which precedes formal abstraction. Language is not downstream of
thought — it is a cognitive tool that participates in reasoning through the Inner
Speech Loop. Curiosity generates goals and drives self-directed learning. Temporal
patterns compress into habits. Social observation is architecturally equivalent to
personal experience. The result is a system that is simultaneously good at
structured reasoning, natural language, efficient learning, low-energy operation, and
graceful development — not by trading these properties against each other, but by
building them from one unified substrate.

PSN-2 · Ideation Labs · Version 1.0 · April 2026 · CONFIDENTIAL

PSN-2 (Kaggle-Constrained) — Product Requirements Document

Version 2.0 · April 2026

---

1. Purpose

This document defines a fully specified, Kaggle-constrained implementation of PSN-2. It resolves missing training definitions, growth mechanics, and system constraints required for actual execution.

Goal: Build a unified intelligence substrate that can learn structure, reasoning, and language under strict compute, memory, and time constraints.

---

1. System Constraints (Hard Limits)

GPU: ≤16GB VRAM

RAM: ≤30GB

Session: ≤9 hours

Disk: ≤20GB persistent

Design implications:

All state must be serializable

Training must be episodic

Growth must be bounded + pruned

Computation must be sparse by default

---

1. Core System Definition

PSN-2 is a prediction-error-driven substrate where all cognition emerges from:

Node states (nu, e, tau)

VSA representations (bipolar vectors)

Bond algebra (typed relations)

Shape formation and commitment

No separation between learning and inference.

---

1. Representation Layer

3.1 VSA Configuration

Dimensionality: D = 512 (initial), scalable to 1024

Type: bipolar {-1, +1}

Operations:

Bind: element-wise multiply

Bundle: weighted sum + normalization

Similarity: cosine

---

3.2 Node Definition

Each node i maintains:

nu_i: prediction vector (R^D)

e_i: prediction error

tau_i: temporal trace

sigma_i: node type

b_i: budget state

---

1. Training Objective (Critical Addition)

4.1 Primary Loss Function

The system minimizes:

L_total = L_pred + lambda_shape *L_shape + lambda_sparse* L_spike

Where:

L_pred: prediction error

L_shape: shape reconstruction consistency

L_spike: sparsity regularization

---

4.2 Prediction Targets

Prediction is defined as:

D1/D2: next-state structure prediction

D3+: shape consistency + outcome prediction

D4+: language reconstruction

---

4.3 Loss Components

Prediction Loss:

L_pred = ||nu_pred - nu_target||^2

Shape Loss:

L_shape = 1 - cos(shape_centroid, reconstructed_shape)

Sparsity Loss:

L_spike = mean(spike_i)

---

1. Developmental Curriculum (Executable Version)

D1/D2 — Structural Grounding (Kaggle Optimized)

Hybrid dataset:

1. ARC-AGI grids

Input: grid

Target: completed grid

Objective: spatial pattern inference

1. Synthetic relational graphs

Format: (entity, relation, entity)

Task: masked prediction

Training tasks:

Predict missing grid regions

Predict masked relation components

Metrics:

Grid accuracy ≥ 0.75

Relation prediction ≥ 0.85

---

D3 — Social Structure (Reduced)

Input: simple agent traces

Task: predict next action or outcome

---

D4 — Language Grounding

Dataset:

TinyStories subset

Paired with structural representations

Tasks:

Text → shape encoding

Shape → text reconstruction

---

D5/D6 — Abstract Reasoning

Math + logic tasks

Multi-step reasoning

---

1. Spiking Computation Engine (SCE)

Spike rule:

spike_i = 1 if e_i > tau_spike

Thresholds:

Perceptive: 0.30

Compositional: 0.20

Recursive: 0.10

Commit: 0.40

Training uses straight-through estimator.

---

1. Attractor Library

Initial Constraints

Max entries: 10,000

Memory cap: ~200MB

Operations

Insert if no match (cos < 0.6)

Retrieve via cosine top-k

Prune by utility score

---

1. Temporal Abstraction Engine (TAE)

Motif Detection

Trigger when:

sequence length ≥ 3

recurrence ≥ 5

similarity ≥ 0.75

Promotion Conditions

success rate ≥ 0.75

pulse savings ≥ 3

---

1. Curiosity Engine (CE)

Trigger:

error > 0.5

no attractor match

low remaining budget

Output:

goal shape stored in ERS

---

1. Growth Protocol (Fully Specified)

10.1 Node Growth

Spawn when:

persistent error > 0.6 for 5 pulses

Initialize:

nu_new = nu_i + noise

Limit:

max nodes = 5,000

---

10.2 Attractor Growth

Add new attractor if no match

Cap: 10k

Prune lowest utility

---

10.3 Memory Growth

ERS tiers:

Working (ephemeral)

Episodic (session)

Semantic (compressed)

Promotion rule:

utility > 0.7

---

10.4 Pruning

Trigger when limits exceeded:

remove lowest utility attractors

merge low-activity nodes

delete failed motifs

---

1. Checkpointing System (Critical for Kaggle)

Serialize every session:

node states

attractor library

ERS memory

motif library

growth ledger

Format: PyTorch state dict + JSON metadata

Frequency:

every 30 minutes

final checkpoint at session end

---

1. Training Loop

For each episode:

1. Encode input → shapes

2. Run Phase A–F loop

3. Compute prediction

4. Apply loss

5. Backpropagate

6. Update nodes and bonds

7. Update attractors

8. Detect motifs

9. Trigger curiosity if needed

---

1. System Scaling Strategy

Start:

D = 512

nodes = 256

Growth:

increase nodes gradually

expand attractors

compress via motifs

Constraint:

growth rate ≤ compression rate

---

1. Success Criteria

Stage 1:

learns ARC patterns

forms stable shapes

Stage 2:

language grounding works

Stage 3:

motifs reduce compute

Stage 4:

system grows without exceeding limits

---

1. Non-Goals (Kaggle Phase)

Neuromorphic deployment

Full-scale social learning

Large-scale language modeling

---

Final Statement

This version of PSN-2 is executable under constrained hardware by:

defining explicit loss functions

constraining memory and growth

using hybrid structural datasets

enforcing checkpoint-based continuity

The system remains architecturally aligned with full PSN-2 while being practically trainable.

The revised PRD has been created in the canvas. It incorporates:

Explicit training objectives and loss functions

A concrete D1/D2 dataset strategy (ARC-AGI + synthetic relational graphs)

Fully defined growth, pruning, and checkpointing systems

Kaggle-specific constraints translated into architectural rules

A step-by-step executable training loop

Next step, if needed: convert this into a full codebase scaffold (module-by-module, class-by-class) so you can begin implementation immediately.

This is the most important constraint to get right before writing a single line of code. Let me give you an honest picture.
Kaggle Reality Check
What you actually get:
2× T4 GPUs — 16GB VRAM each (or P100 16GB)
~30GB RAM
9-hour session hard cutoff — then the process dies
~20GB disk (persistent via Kaggle Datasets)
Internet access during training
Your goal (structure learning + reasoning + language + energy efficiency + data efficiency) is fully compatible with PSN-2's design — but only a subset of the PRD is actually trainable in this environment. The rest is either too large, requires too many sequential sessions, or isn't implementable yet because the training recipe is underspecified in the PRD.
What's Actually Buildable vs. What Isn't
PSN-2 Component
Kaggle Feasibility
Why
VSA substrate (D=512–1024)
✅ Fully feasible
Pure tensor ops, small memory footprint
SCE (spiking sparse compute)
✅ Actually helps Kaggle
Sparse = less VRAM, fits the constraint
Attractor library (small vocab)
✅ Feasible
10k words × D=1024 = ~40MB
USL codec (understand + generate)
✅ Feasible at small scale
Needs a small dependency parser module
ESS (emotional shapes)
✅ Lightweight
Just additional shape types + bond weights
Curiosity Engine
✅ Feasible
It's a monitor process on error signals
Temporal motifs (TAE)
✅ Critical to include
This is your compression mechanism — needed to stay within Kaggle budget
Growth protocol
⚠️ Feasible with checkpointing
Must checkpoint every session; growth happens across sessions
Full D1–D6 dev curriculum
⚠️ Multi-session only
Each stage needs its own Kaggle session
D=10,000+ VSA dimensions
❌ Not on Kaggle
OOM. Cap at D=1024 initially
Full SLS (social learning)
❌ Later stage
Requires prior stages to be stable first
Neuromorphic compilation
❌ Not on Kaggle
Production target, ignore for now
The Actual Training Problem (What the PRD Doesn't Fully Specify)
The PRD defines the architecture precisely. But to train on Kaggle you need four things the PRD currently leaves implicit:

1. The training signal — PSN-2 is driven by prediction error minimization. You need to define: prediction of what, at which layer, computed how. The straight-through estimator for spiking is in the PRD (Section 4.1). What's missing is the outer loss: is it next-token prediction? shape reconstruction? ARC-AGI grid completion?
2. The dataset per developmental stage — D1 (sensorimotor) through D6 (abstract language) each need different data. For a Kaggle-constrained system, the most practical mapping is:
D1/D2 → ARC-AGI training grids (visual-spatial structure without language)
D3 → Simple agent interaction traces (social shape grounding)
D4/D5 → BabyLM corpus or a filtered subset of Tiny Stories (language on top of grounded structure)
D6 → Math/logic reasoning benchmarks
3. Backprop through VSA operations — VSA bind (XOR for binary, element-wise multiply for bipolar) and bundle (sum) are differentiable. Cleanup (cosine similarity against codebook) needs a straight-through or Gumbel-softmax. This is solvable but needs explicit implementation choices.
4. Cross-session state — With a 9-hour cutoff, you must checkpoint: node states, attractor library, ERS memory, motif library, and the growth ledger. This needs a defined serialization format before training begins.
Recommended Build Order for Kaggle
Session 1: Core substrate

- VSA engine (D=512, bipolar)
- Node population (256 nodes)
- Phases A–F (prediction, bond update, morphogen, shape, verifier, render)
- Basic attractor library (shapes only, no language yet)
- Checkpoint format defined
  Target: system can form and commit shapes from structured input

Session 2: Spiking + language foundation

- SCE spike decision rule (tau_spike per phase)
- Lexical attractor library (5k word vocab)
- USL understanding direction (text → shapes)
- Train on ARC-AGI grids + simple captions
  Target: system can induce shapes from language input

Session 3: Generation + curiosity

- USL generation direction (shapes → text)
- ISL (inner speech loop)
- Curiosity Engine (error monitoring + goal generation)
- Temporal motif detection (TAE basic)
  Target: system can generate language from committed shapes + log curiosity goals

Session 4+: Growth protocol

- Node spawning on persistent high error
- Attractor expansion
- ERS tiered promotion
- Budget-gated pruning
  Target: system grows across sessions within hardware limits
The Core Design Decision You Need to Make First
Everything hinges on one question the PRD leaves open for your implementation:
What is the primary training task for D1/D2?
Three viable options for Kaggle:
ARC-AGI grids — Pattern completion without language. Forces genuine structure learning. Hard, but aligns exactly with PSN-2's thesis. Data-efficient by design.
Synthetic relational graphs — Generate structured (entity, relation, entity) triples and train the substrate to predict masked elements. Fully controllable, fast iteration.
Pixel-level prediction on simple scenes — Closest to biological sensorimotor grounding, but requires vision preprocessing overhead.
Option 2 is the most Kaggle-practical for early sessions. Option 1 is the right long-term benchmark target.
Bottom line: PSN-2 is trainable on Kaggle if you: cap VSA at D=512–1024, treat each developmental stage as its own set of Kaggle sessions with checkpointing, define the outer training loss concretely before starting, and use temporal motif compression aggressively to stay within compute budget. The architecture's own sparsity mechanism (SCE) is actually your best friend in a constrained environment — the system should naturally use less compute as it learns.
Want me to spec out the Session 1 implementation — the core substrate with VSA, node states, phases A–F, and the checkpoint schema?

i want a new prd with all the updated and missing information, and I think D1/D2 should be a mixture of ARC-AGI grids — Pattern completion without language. Forces genuine structure learning. Hard, but aligns exactly with PSN-2's thesis. Data-efficient by design.
- Synthetic relational graphs — Generate structured (entity, relation, entity) triples and train the substrate to predict masked elements. Fully controllable, fast iteration.
