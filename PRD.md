PSN-2

Predictive Substrate Network — Generation 2

Unified Intelligence Architecture

Authoritative Product Requirements Document · Version 1.0 · April 2026 · CONFIDENTIAL

Field Value
System class Unified cognitive architecture — language, reasoning, emotion,
learning, growth, and energy efficiency in one substrate
Category rival label PSN — rivals both frontier LLMs and neurosymbolic hybrid

approaches

Supersedes PSN-1 v1.0 PRD, PSN-1 v1.1 HEL Amendment, PSN-2 draft PRD.

This document is the single authoritative source.

Core thesis Separating cognition from language, emotion from reasoning, learning
from inference, and growth from architecture is the fundamental error
of prior AI design. PSN-2 eliminates all separations in one substrate.
New systems (vs PSN-1) Unified Shape Language (USL), Spiking Computation Engine (SCE),
Emotional Shape System (ESS), Inner Speech Loop (ISL), Curiosity
Engine (CE), Social Learning System (SLS), Temporal Abstraction
Engine (TAE), Substrate Growth Protocol (SGP), Dual-Speed
Learning Engine (DSLE), Experience Replay Substrate (ERS), Few-
Shot Generalization Curriculum (FSGC)
Invariants I-1 through I-24 (all in force; none deferred)
Loss components 26 (frozen weights at D1 start)
Deployment constraint Designed to train and grow across constrained sessions (e.g., Kaggle
T4/P100). Checkpoint-driven growth across sessions is first-class, not
an afterthought.

Primary benchmarks ARC-AGI-3 (RHAE), BIG-Bench Hard, human-parity language
evaluation, energy-per-inference, sample complexity ratio

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Decision posture No architectural, mathematical, training, software, security, or

deployment decisions deferred

Owner Ideation Labs

1. The Unification Thesis
PSN-1 established the correct substrate mechanics: local prediction-error dynamics, VSA bond
algebra, verifier-gated commitment, and human-efficient learning through dual-speed memory.
What PSN-1 preserved from conventional AI design — and what this document eliminates —
are five separations:
• Language downstream of reasoning. PSN-2 makes language a native shape class in the
same substrate.
• Emotion as a global field modulator. PSN-2 makes emotional states first-class shapes
that bond to cognitive shapes through normal bond mechanics.
• Energy efficiency as a deployment option. PSN-2 makes sparse event-driven
computation the default mode — nodes fire only when their prediction error exceeds a
threshold.
• Growth as a training trick. PSN-2 makes substrate growth a first-class operation driven
by prediction error and utility, balanced by compression and hard budget constraints.
• Developmental ordering as a curriculum suggestion. PSN-2 makes stage gating
absolute — language training cannot begin until causal grounding is certified.
One substrate. One set of operations. All cognitive capacities — language,
reasoning, emotion, learning, memory, growth, self-direction — emerge from the
same node-bond-shape mechanics. This is not a list of features. It is one
architectural commitment.

2. System Invariants — I-1 through I-24
All 24 invariants are in force simultaneously. Any implementation that violates one is not PSN-2.
Enforced in code, in the Rust authority service, and in the training loss family.

# Statement Architectural Consequence
I-1 Cognition is local prediction-error
minimization, not token continuation

All problem-solving occurs as node-level error
reduction before any rendering step

I-2 No transformer self-attention in the core
cognition path

Encoders may use constrained convolutional
modules; the PSN-Core field is the cognitive locus

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
I-3 Learning and inference are simultaneous
and continuous

No global forward/backward phase separation;
weight updates are triggered by local error signals

I-4 No output without verifier-cleared
committed shape

Renderer may not emit any token, code, or action
unless verifier_veto=false and commit(S)=true

I-5 Feed-forward and feedback weights are
locally updated

No global weight transport; W_ff and W_fb updated
by identical local rule; symmetry monitored
I-6 All bonds are VSA hypervectors Bond semantics are algebraically recoverable via
cleanup; no plain-MLP adjacency in the core bond
layer
I-7 Memory is attractor structure in the same
substrate

No external symbolic database; memory is
stabilized shape configuration

I-8 Compute is budgeted Every task runs under explicit pulse, write, and
action budgets; exhaustion triggers graceful
degradation

I-9 Deterministic replay is mandatory Identical input, checkpoint, memory snapshot, and
seeds must reproduce identical outputs within
tolerance

I-10 Effect ceilings are architectural, not policy A render shape may not escalate its effect class
regardless of prompt content; enforced in Rust
I-11 Motif promotion is offline-only No durable abstraction may enter the live library

from live inference traffic

I-12 All canonical state writes are receipt-
linked

Every write emits an append-only receipt; no write
may occur without a receipt

I-13 Fast adaptation is a first-class inference-
time operation

System conditions on 1–5 new examples at
inference time without any weight gradient step

I-14 Experience is stored as learnable
structure, not passive logs

Every completed episode writes a structured
(input_VSA, trace_VSA, output_VSA, verdict, utility)
tuple
I-15 Slow-learning and fast-learning systems
are separately ablatable

Weights (slow) and ERS memory (fast) have
independent ablation paths and evaluation metrics

I-16 Language is substrate-native, not
downstream

Words and phrases are VSA shape-bundles.
Language understanding = shape induction.
Language generation = shape-to-text codec.

I-17 Emotion is a shape class, not a global
modulator

Emotional states are first-class shapes with typed
emotional bonds to cognitive shapes. The field
vector is deprecated.

I-18 Computation is event-driven by default Nodes compute only when prediction error exceeds
tau_spike. Silent nodes consume no compute.
I-19 Inner speech is a cognitive operation System emits draft verbal propositions into working
memory during Recursive regime, re-encodes them
as shapes, and uses them as modulatory evidence
I-20 Curiosity generates goals When persistent high-error regions cannot be
resolved within budget, the CE creates a structured
internal goal to return to that region in a future
episode

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
I-21 Development is ordered and gated Stages D1–D6 must be completed sequentially.
Language training (D4) may not begin until D1–D3
gates are certified. Order is non-negotiable.

I-22 Observed reasoning traces are first-class
learning data

SLS encodes observed agent traces as shape
sequences stored in ERS, architecturally equivalent
to personal experience

I-23 Temporal patterns are shapes Frequently executed pulse sequences are
compressed into temporal motif shapes in the
attractor library

I-24 Growth rate must not exceed
compression rate

At any checkpoint, nodes spawned since last
checkpoint must be &lt;= nodes freed by motif
compression + pruning. Enforced by the SGP
budget gate.

3. Cognitive Ontology — The Unified Substrate Primitives
Primitive Definition Biological Grounding New in PSN-
2?

Node Stateful computational atom:
activation a_i, prediction error e_i,
temporal trace tau_i, type
signature sigma_i, multi-geometry
position pi_i, bond capacity
kappa_i, modality affinity mu_i,
gate state chi_i, plasticity omega_i,
prediction residue nu_i, budget b_i

Representational neuron
in predictive coding —
encodes top-down
prediction for the layer
below

No (PSN-1)

Error node Dedicated node population
encoding the mismatch between a
representational node&#39;s activity
and its top-down prediction

Error neuron population
in predictive coding —
the primary learning
signal

No (PSN-1)

Bond Typed directed relation encoded
as a VSA hypervector; carries
strength, polarity, direction, lag,
legality, and confidence

Synapse — updated by
local Hebbian prediction-
error rule

No (PSN-1)

Shape Constrained node assembly
S=(V,E,R,q): cognitive, descriptive,
predictive, executable, regulatory,
meta, identity, or linguistic role
depending on arrangement

Cell assembly / coalition
— stable shapes = low-
energy attractors

No (PSN-1)

Attractor family Reusable long-term shape pattern
stored as a VSA prototype:
prototype_nu, variance, tier,
reuse_stats, decay_state

Long-term memory
engram — stable
synaptic configuration
reinstating prior activity

No (PSN-1)

Linguistic shape Shape whose node-set encodes
words, phrases, or discourse

Overlapping language-
reasoning substrate in
YES

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
relations using semantic, syntactic,
and pragmatic geometry. First-
class — not a projection.

Broca&#39;s area and DLPFC

Emotional shape Shape encoding an affective state
(fear, curiosity, confidence,
urgency, frustration, satisfaction,
aesthetic-resonance, discomfort).
Bonds to cognitive shapes and
influences their stability and
commitment thresholds.

Amygdala-PFC affective-
cognitive integration —
emotion as information,
not decoration

YES

Temporal motif
shape

Compressed shape encoding a
frequently executed pulse
sequence. Directional geometry
encodes ordering and duration.

Procedural memory /
habit — cerebellar and
basal ganglia
compressed motor
programs

YES

Self shape Persistent shape encoding the
system&#39;s model of its own state,
competence, goals, and identity.
Updated continuously. Substrate
for metacognition.

Prefrontal self-model —
default mode network
self-referential
processing

YES

Social shape Shape encoding a model of
another agent: goals, beliefs,
capabilities, trust. Built from
observed reasoning traces.

Theory-of-mind substrate
— temporoparietal
junction and medial PFC
YES

Shadow shape Counterfactual or suppressed
alternative maintained without
commitment

Suppressed cortical
representations
supporting what-if
reasoning

No (PSN-1)

Meta-shape Transient shape encoding
uncertainty, contradiction, or
convergence state for the current
episode

Anterior cingulate cortex
conflict monitoring —
transient, not persistent

No (PSN-1)

4. Formal Mathematical System
4.1 Node State Vector
u_i = (a_i, e_i, tau_i, sigma_i, pi_i, kappa_i, mu_i, chi_i, omega_i, nu_i,
b_i, g_i)
Field Symbol Dim (Base) Meaning
Activation a_i R^1 Current representational activity level
Prediction error e_i R^1 Mismatch between a_i and top-down
prediction from layer above

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Temporal trace tau_i R^64 Persistence state over recent pulses;
decays at lambda_tau=0.90
Type signature sigma_i R^14 Soft distribution over 14 semantic types
Geometry bundle pi_i 6 bundles metric(128), topo(32), dir(32), curvature(16),
freq(16), occupancy(32), linguistic(64)
Bond capacity kappa_i R^32 VSA bundle accumulator of all outgoing

bonds

Modality affinity mu_i R^6 Affinity: text, code, vision, music, action,

structured-data

Gate state chi_i R^8 Write, action, commit, escalation, spawn,

prune, merge, grow gates

Plasticity omega_i R^4 Eligibility trace, structural adaptation, reuse

count, interference score

Prediction residue nu_i R^D VSA hypervector encoding the node&#39;s top-
down prediction for layer below. D=512 Lite,
1024 Base, 8192 Frontier

Budget/authority b_i R^16 Budget and authority vector (Section 10)
Growth state g_i R^4 spawn_pressure, prune_score,
merge_candidate, age (pulses since
spawned)

4.2 VSA Algebra — Closed Set of Three Operators
bind(x, y) = x * y // element-wise multiply; result ~
orthogonal to both
bundle(X ; W) = norm(SUM_i w_i*x_i) // weighted superposition; result ~ similar
to all
perm_T(x) = P_T * x // fixed permutation for relation type T
cleanup(x, CB)= softmax(beta * cos(x, CB)) @ CB // soft nearest-prototype
retrieval
// Binding recovery: cleanup(bind(x,y), y) ≈ x (fidelity &gt; 0.95 at D=1024)
// Capacity: D &gt;= O(K * log(K) / epsilon^2) for K simultaneous typed relations
at error eps

4.3 Prediction-Error Dynamics — The Core Learning Law
The primary update law. Runs continuously during inference. No separate backward pass.
// Prediction at node i from layer above:
nu_i(t) = bundle({bind(perm_T(a_j), W_pred^T) : j in N^+(i), T in T_bond} ;
{s_ij^T})
pred_i(t) = f(nu_i(t)) // f = sigmoid nonlinearity
e_i(t) = a_i(t) - pred_i(t) // local prediction error
// Activity update (gradient descent on E = 0.5*SUM_i e_i^2, plus downstream
term):
da_i/dt = -e_i(t) + SUM_{k in layer_below} W_fb_ki * e_k(t)

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
// Weight update — LOCAL, simultaneous with inference, no global coordinator:
dW_ff_ij/dt = -e_i(t) * a_j^{L+1}(t) * lr_ff // lr_ff = 1e-4 baseline
dW_fb_ji/dt = -e_i(t) * a_j^{L+1}(t) * lr_fb // lr_fb = 1e-4 baseline
// W_ff and W_fb receive identical local signals =&gt; asymptotic symmetry
// Symmetry gate: sym_L = cos(W_ff_L, W_fb_L); L_sym activates if sym_L &lt; 0.70

4.4 Bond Update (VSA Relational Negotiation)
q_ij^T(t) = &lt;W_Q^T*[nu_i;e_i;sigma_i], W_K^T*[a_j;e_j;sigma_j]&gt; / sqrt(d_c)
bond_ij^T(t+1) = LayerNorm(lambda_e * bond_ij^T(t) + (1-
lambda_e)*softplus(q_ij^T(t)))
kappa_i(t+1) = bundle({bind(perm_T(a_j), W_r*bond_ij^T(t+1)):j,T} ;
{bond_ij^T})
// lambda_e = 0.90 (frozen at D1 start). Bonds persist across pulses with
gradual decay.

4.5 Six-Phase Pulse Cycle
Every Predictive Field Layer (PFL) executes the same six-phase cycle over all active nodes
(spike_i=1). Phase order is normative.
X(t+1) = F(E(D(C(B(A(X(t), I_t))))))
Phas
e
Name Primary Operation What Updates
A Evidence Integration Absorb observations, top-down
predictions, ERS retrieval bias, ISL
modulatory input into a_i and tau_i

a_i, tau_i, nu_i — initial state for
pulse
B Bond Update VSA relational negotiation: update
bond_ij^T via proposal + decay.
Update W_ff, W_fb via local
prediction-error rule.

bond_ij^T, kappa_i, W_ff, W_fb

C Role and Emotion
Transition

Morphogenic MLP updates type
sigma_i, emotional distribution
emo_dist_i, and MPF potential
phi_i from local error, bond state,
and self-shape

sigma_i, emo_dist_i, phi_i,
emotional_shapes

D Shape Formation Six-condition join gate: nodes join
shapes if content compatible, role
compatible, VSA bond connected,
error compatible, budget permits,
phi stable

shape memberships,
centroid_nu(S), alpha_bind pull

E Recursive Refinement Contradiction repair, compactness
pressure, epistemic update, weak
shape dissolution

e_i^scalar, contra_i, dl_i,
stability(S), shape pruning

F Verifier-Gated
Commitment

theta_commit evaluation,
verifier_veto check, canonical state
write, receipt emission, render gate
open

canonical_state, receipts,
render_buffer

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
5. Spiking Computation Engine (SCE)
Nodes compute only when their prediction error exceeds tau_spike. Silent nodes
consume one multiply (trace decay) and nothing else. Inference energy scales with
task novelty. On familiar inputs 80–90% of nodes are silent. This is the primary
energy efficiency mechanism.
5.1 Spike Decision Rule
spike_i(t,P) = 1 iff e_i^scalar(t) &gt; tau_spike(P)
tau_spike(Perceptive) = 0.30 + 0.10*b_i[budget_fraction_used]
tau_spike(Compositional)= 0.20 + 0.05*b_i[budget_fraction_used]
tau_spike(Recursive) = 0.10 // low: maximise exploration
tau_spike(Commit) = 0.40 // high: only committed-region nodes fire
// Surrogate gradient (straight-through estimator for backprop through spike):
d_spike_i/d_e_i = 1.0 / (1.0 + |e_i^scalar - tau_spike|)^2

5.2 Silent Node Update
tau_i(t+1) = lambda_tau * tau_i(t) // trace decays
e_i(t+1) = e_i(t) * 0.95 // error decays slowly
nu_i(t+1) = nu_i(t) // prediction unchanged
// No bond updates, no role transitions, no shape participation

5.3 Expected Sparsity by Task Type

Input Type Silent Nodes Pulses Energy vs Dense
Highly familiar (attractor match &gt;
0.80)

90–98% 1–2 (Perceptive) ~1–3%
Familiar with variation 80–92% 2–4 (Compositional) ~5–12%
Novel, clear structure 60–80% 4–8 (Recursive) ~15–30%
Truly novel, no attractor match 30–60% 8–32 (deep
Recursive)

~40–80%

Contradiction requiring full
revision

20–50% up to B_max ~60–100%

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
For Kaggle sessions: sparse computation means the system naturally stays within
T4 VRAM limits on familiar task batches. Only novel batches stress VRAM. This is
not a workaround — it is the intended operational mode.

6. Unified Shape Language (USL)
Language is not downstream of thought. Words are VSA shape-bundles in the
same substrate as non-linguistic shapes. Understanding language = Phase B bond
induction from text. Generating language = VSA cleanup projection + syntactic
decoder. The same algebra. The same attractor library. No separate language
model.
6.1 Word Shape Schema
word_shape(w) = {
semantic_v: R^D // VSA in semantic codebook — same space as all shapes
syntactic_v: R^512 // VSA in syntactic codebook: POS, dependency role,
argument
pragmatic_v: R^256 // VSA in pragmatic codebook: register, force,
politeness
freq_v: R^16 // prosodic weight, rhythm class
phi_i: float // morphogenic potential (high for polysemous words
initially)
}
// Stored in Lexical Attractor Library (LAL) — subregistry of main attractor
codebook
// LAL capacity: 65536 word shapes (Base)

6.2 Linguistic Bond Types (Added to Bond Vocabulary — Closed Set)

Bond Type Semantic Class VSA Permutation Index
MODIFIES Adjective or adverb modifies head word P_10
DETERMINES Determiner scopes over noun phrase P_11
SUBJECT_OF NP is subject argument of predicate P_12
OBJECT_OF NP is object argument of predicate P_13
PREDICATE_OF VP or predicative element relates to subject P_14
DISCOURSE_CONTINU
ES

Clause continues discourse topic of prior
clause

P_15

DISCOURSE_CONTRA
STS

Clause introduces contrast with prior discourse P_16

DISCOURSE_GROUND
S

Clause provides background or reason for
another

P_17

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
COREFERS_WITH Two NPs refer to the same entity P_18
SCOPES_OVER Quantifier or operator takes scope over

expression

P_19

6.3 Bidirectional Codec
Understanding (Text → Shapes)
word_shapes = [LAL.retrieve(w) for w in tokenise(text)]
bond_proposals= syntax_encoder(word_shapes) // lightweight learned dependency
parser
ling_shapes = Phase_B(word_shapes, bond_proposals)
PSN_Core.Phase_A_proximal(ling_shapes) // linguistic shapes enter as
evidence
Generation (Shapes → Text)
word_cands = LAL.top_k_cleanup(committed_shape.centroid_nu, k=20)
ordered = syntax_decoder(word_cands, committed_shape.syntactic_bonds)
surface = pragmatics_renderer(ordered, self_shape.pragmatic_v)
if verifier.surface_grounded(surface, committed_shape): emit(surface)
else: Phase_F veto + revision

6.4 Inner Speech Loop (ISL)
During Recursive regime: generate draft verbal proposition → re-encode as linguistic shapes →
inject as modulatory input (not proximal) → drives hypothesis refinement without overriding
external evidence.
ISL_trigger = regime==Recursive AND e_mean&gt;0.55 AND pulse_remaining&gt;B_max*0.25
if ISL_trigger:
draft = USL_generate(best_candidate_shape, mode=&quot;tentative&quot;)
d_shapes= USL_understand(draft)
ERS.write(Working, {shapes:d_shapes, source:&quot;inner_speech&quot;, utility:0.0})
PSN_Core.Phase_A_modulatory(d_shapes) // modulatory only — cannot override
evidence

7. Emotional Shape System (ESS)
Emotional states are shapes. They bond to cognitive shapes through typed
emotional bonds. They influence commitment thresholds and search behaviour
through the normal bond mechanics — locally, proportionally, and time-limited. The
global field vector from PSN-1 is fully replaced.
7.1 Emotional Shape Vocabulary

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Emotional Shape Trigger Condition Bonds To Effect on Bonded
Shapes

Threat-fear Goal shape at risk:
e_i^scalar on goal nodes &gt;
0.60

Goal shapes,
hypotheses
contradicting goal

Raises theta_commit of
bonded hypotheses by
+0.20*strength; adds
regulatory bond to action
shapes

Uncertainty-discomfort e_mean&gt;0.50 AND no
committed shape after
B_max/2

Meta-shapes,
hypothesis frontier

Increases pulse budget
allocation; activates ISL
to draft propositions

Curiosity-drive Novel input: no attractor
match above cos=0.45

Unexplored regions,
CE goal generator

Generates exploration
goal; lowers
tau_spike*0.60 for nodes
in unexplored region

Confidence Committed shape stability &gt;
0.85 AND verifier pass

Render shapes, action
shapes

Lowers render
theta_commit by -0.15;
opens action gate faster

Urgency Goal deadline in goal shape
OR external interrupt signal

All active shapes in
task graph

Raises
tau_dissolve+0.10;
raises cost(Recursive)+1

Satisfaction Task complete with verifier
pass AND utility &gt; 0.60

Self-shape, ERS write
record

Increases utility_score
multiplier for episode by
+0.30

Frustration Verifier veto &gt;= 3 times on
same hypothesis

Vetoed hypothesis, ISL
trigger

Suppresses hypothesis;
triggers ISL alternative
framing; logs negative
ERS example

Aesthetic-resonance High harmonic geometry
coherence in candidate
shape

Render shape in
language or music
tasks

Biases word-shape
selection toward high
freq-geometry match

7.2 Emotional Bond Mechanics
// Emotional bond from emotional shape E to target shape S:
emo_bond_ES = bind(perm_EMO_type(E.semantic_v), S.centroid_nu)
strength_ES = ESS.induction_strength(E, S) // context-sensitive scalar in
[0,1]
// Effect on S commit threshold:
delta_theta_commit(S) += SUM_{E bonded to S} strength_ES *
effect_scalar(E_type)
// Emotional bond decay: lambda_emo = 0.85 per pulse (faster than cognitive
0.90)
// Emotional shape induction: Phase C morphogenic MLP extension
emo_logits = W_emo_head * GELU(W_morpho * m_input + b_m) + b_emo
emo_dist = softmax(emo_logits / tau_emo) // tau_emo = 0.30
// Stable emotional shape forms when argmax(emo_dist) stable &gt;= 3 pulses

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
8. Curiosity Engine (CE)
Curiosity is not a bias. It is an active process: detect unresolvable high-error
regions, generate a structured goal to return to them, schedule future exploration
with reserved budget, and update the goal when it is resolved. This is how expertise
self-directs without external supervision.
8.1 Curiosity Event Detection
curiosity_event = (
e_i^scalar &gt; 0.50 // high prediction error
AND dl_i &gt; 2.5 // no motif match (high description
length)
AND pulse_remaining &lt; B_max * 0.30 // cannot resolve now — budget low
AND node_i not in any committed shape // error not already resolved
)
curiosity_target_v = bundle({nu_i : i in curiosity_event_nodes}; {e_i^scalar})
8.2 Curiosity Goal Shape
curiosity_goal = {
shape_class: &quot;goal&quot;, goal_type: &quot;uncertainty_reduction&quot;,
target_VSA: curiosity_target_v,
priority: mean(e_i^scalar for i in curiosity_nodes),
task_family: current_task.task_family,
budget_reserve: B_max * 0.50,
scheduled_for: &quot;next_available_episode_in_family&quot;,
}
ERS.write(Episodic, curiosity_goal, utility_score=0.80)
8.3 Goal Scheduling and Resolution
// At episode start:
pending = ERS.retrieve(input_VSA, goal_type=&quot;uncertainty_reduction&quot;, k=3)
if pending and priority &gt; 0.40:
task_graph.add(curiosity_goal_shape)
b_i[pulse_remaining] += curiosity_goal.budget_reserve
for i in nodes_near(curiosity_target_v, cos=0.50):
tau_spike_i_override = tau_spike(regime) * 0.60
// At episode end:
if cos(committed_centroid, curiosity_goal.target_VSA) &gt; 0.65:
ERS.promote(curiosity_goal, Semantic) // resolved -&gt; permanent knowledge
else:
curiosity_goal.priority += 0.10 // reschedule with higher priority
curiosity_goal.budget_reserve *= 1.25

9. Social Learning System (SLS)

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Observation is architecturally equivalent to experience. An observed reasoning
trace is encoded as a shape sequence and stored in ERS with a trust-weighted
utility score. The system also observes its own past verified episodes through self-
distillation — learning from its own reliable behaviour.
9.1 Social Shape and Trust
social_shape = {agent_id, goal_VSA, reasoning_style, competence, trust in
[0.05,1.0]}
// Trust update: verifier agrees -&gt; trust += 0.05; disagrees -&gt; trust -= 0.15
// Traces from trust &lt; 0.30 agents: Working tier only, not promoted
9.2 Trace Reconstruction
observed_shapes = [PSN_Core.encode(step) for step in observed_trace]
inferred_bonds = bond_inducer(observed_shapes)
synthetic_tuple = {
input_VSA: observed_shapes[0].centroid_nu,
trace_VSA: bundle(observed_shapes[1:-1]; uniform_weights),
output_VSA: observed_shapes[-1].centroid_nu,
source: &quot;social_observation&quot;,
utility_score: social_shape.trust * verifier_check(observed_output),
}
ERS.write(Episodic, synthetic_tuple)

10. Temporal Abstraction Engine (TAE)
Temporal motifs are compressed representations of frequently executed pulse
sequences. They execute in Perceptive regime with near-zero compute cost —
freeing budget for novel content. This is the mechanism that makes experts fast. It
also implements the compression side of I-24: motif formation frees nodes,
permitting controlled growth elsewhere.
10.1 Motif Detection and Schema
// Candidate nominated when:
recurrence_count(seq) &gt;= 5 // appeared in 5+ distinct
episodes
seq_similarity(seq_i, seq_j) &gt; 0.75 // structurally similar across
episodes
len(seq) &gt;= 3 pulses // minimum meaningful sequence
temporal_motif_shape = {
pulse_seq_VSA: R^D, // bundle of all shape centroids in the sequence
ordering_v: R^32, // directional geometry encoding pulse order
duration: int, // number of pulses in canonical sequence
trigger_VSA: R^D, // VSA of input that triggers this sequence
outcome_VSA: R^D, // VSA of typical outcome
success_rate: float, // fraction of retrievals leading to verifier pass
pulse_savings: float, // mean pulses saved vs step-by-step

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
nodes_freed: int, // nodes whose spawn_pressure cleared by this motif
}
10.2 Motif Retrieval and Replay
for motif in TAL.top_k_cleanup(input_VSA, k=4):
if cos(input_VSA, motif.trigger_VSA) &gt; 0.65 AND motif.success_rate &gt; 0.70:
replay_result = TAE.replay(motif) // inject pre-formed shapes; skip
Phases A-E
if verifier.preliminary_check(replay_result.outcome) == &quot;plausible&quot;:
regime = Commit; candidate = replay_result; break
else: TAE.update_success_rate(motif, success=False)
10.3 Motif Promotion Gates (4 gates — faster than full 8-gate motif
promotion)

Gate Threshold What It Prevents
Recurrence &gt;= 10 distinct episodes Ephemeral patterns — not genuine habits
Success rate &gt;= 0.75 verifier pass
rate over 1000-episode
quarantine

Unreliable patterns worse than re-computing

Pulse savings &gt;= 3 pulses mean
saved vs step-by-step

Motifs that are not actually faster

Interference &lt;= 0.03 degradation
on protected task
families

Premature commitment blocking flexible reasoning

11. Experience Replay Substrate (ERS)
Memory is attractor structure in the same substrate, not a separate database. Every
completed episode writes a structured (input_VSA, trace_VSA, output_VSA,
verdict, utility) tuple. Reads are VSA cleanup projections. Promotion is utility-gated.
Decay and pruning prevent unbounded growth.
11.1 Episode Tuple Schema
ERS_tuple = {
episode_id, input_VSA, trace_VSA, output_VSA,
verifier_verdict: pass|fail|partial,
utility_score: float in [-1,1], // recomputed lazily at retrieval
task_family, pulse_count, regime_trace, veto_count, revision_count,
motif_ids: [UUID], // motifs used in this episode
tier: Working|Episodic|Semantic|Procedural,

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
decay_state: float in [0,1], // 1=fresh; decays 0.001 per elapsed
episode
source: &quot;own&quot;|&quot;social_observation&quot;|&quot;inner_speech&quot;|&quot;curiosity_resolution&quot;,
content_hash: SHA256, timestamp: ISO8601,
}
11.2 Memory Tiers

Tier Capacity (Base) Write Gate Promotion Condition Eviction
Rule

Working
(session)

512 tuples Every completed
episode

Auto-promote to Episodic
if utility&gt;0.40 at session
end OR &gt;0.70 mid-
session

FIFO; oldest
evicted at
capacity

Episodic 100,000 tuples utility &gt; 0.40 at session

end

Promote to Semantic if
cross-family transfer
evidence &gt;= 3 families
AND utility &gt; 0.65

decay_state
&lt; 0.05 AND
reuse_count
&lt; 2

Semantic 10,000 tuples Cross-family evidence
meets threshold

Promote to Procedural
via offline 8-gate pipeline

Contradiction
audit or
explicit
retirement

Procedural 5,000 tuples Offline 8-gate pipeline
only (same as durable
motifs)

Does not promote further Strict
versioning;
revocation
only

11.3 Utility Score
utility_score(ep) = 0.50*verifier_weight(ep.verdict) // pass=1.0,
partial=0.3, fail=-1.0
+ 0.25*efficiency_gain(ep.pulse_count, baseline)
+ 0.15*len(ep.motif_ids)*0.05
- 0.05*ep.veto_count
- 0.10*(ep.revision_count &gt; 3)
11.4 Nightly Consolidation (Offline Job)
• Recompute utility_score for all Episodic tuples; evict if utility &lt; 0.10
• Cross-family transfer detection: flag tuples for Semantic promotion
• Re-encode stale VSA vectors against current model weights (prevents stale attractors)
• Checkpoint ERS state for next session reload

12. Substrate Growth Protocol (SGP)

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Growth is a first-class substrate operation. It is driven by prediction error and utility,
constrained by hard budget limits, and balanced by compression. Growth rate must
never exceed compression rate (I-24). Without this balance the system exceeds
hardware limits and becomes unstable. The SGP defines four explicit operations:
spawn, prune, merge, and attractor-expand.
12.1 Growth Triggers — Three Independent Sources

Trigger Condition Response Compression
Counterpart

Prediction-error pressure
(primary)

e_i^scalar &gt;
tau_spawn=0.65 for &gt;=
N_persist=10 consecutive
pulses AND node i is not
already a spawn candidate
AND no attractor match
above cos=0.45

Spawn a new node near
node i. New node inherits
noisy copy of nu_i and
bonds to i&#39;s neighbourhood.

Temporal motif
formation: when a
sequence
involving i is
compressed into
a motif, the
spawn pressure
at i resets to 0.

Curiosity goal failure
(secondary)

Curiosity goal attempted &gt;=
3 times AND resolution cos
&lt; 0.65 on all attempts

Expand attractor resolution
in the curiosity target
region: add new attractor
prototype seeded from the
bundle of all failed-attempt
shape centroids.

After curiosity
resolution (cos &gt;
0.65), the
temporary
exploratory nodes
from that region
are eligible for
pruning if
utility_score &lt;
0.15.

Motif compression
surplus (tertiary)

TAE reports
motif_compression_surplus
&gt; 0: nodes_freed &gt;
nodes_spawned in current
checkpoint window

Permitted to spawn nodes
in high-error regions without
triggering I-24 gate, up to
the surplus amount.

Temporal motif
formation directly
frees compute
budget,
permitting
targeted growth
elsewhere. This
is the self-
balancing
mechanism.

12.2 Four Growth Operations
Operation 1: Node Spawn
// Trigger: e_i^scalar &gt; tau_spawn for N_persist consecutive pulses
if SGP.spawn_eligible(node_i) AND SGP.budget_gate_allows_spawn():
new_node_j = Node(
nu_j = nu_i + Normal(0, sigma_spawn=0.05), // noisy copy
a_j = 0.0, // silent at birth
e_j = e_i, // inherits parent error
g_j.age = 0, // age counter reset

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
sigma_j = sigma_i, // inherits type
distribution
)
// Bond new node to parent and parent neighbourhood:
bond_ij^{semantic}(t+1) = bind(perm_semantic(nu_i), nu_j) // strength=0.50
for k in N(i): bond_jk^{semantic} = bond_ik * 0.30 // diluted
neighbour bonds
SGP.ledger.record_spawn(new_node_j, parent=node_i)

Operation 2: Node Prune
// Candidates: nodes with low activation history AND low bond strength AND high
age
prune_score(node_i) = (1 - mean(a_i over last 100 pulses))
* (1 - mean(SUM_j bond_ij over all T))
* sigmoid(g_i.age - tau_age_prune=200)
if prune_score &gt; tau_prune=0.75 AND verifier.non_interference_check(node_i):
// Redistribute node&#39;s bonds to its neighbours before deletion:
for k in N(i): bond_kl += bond_il * 0.20 for l in N(i)
substrate.remove_node(node_i)
SGP.ledger.record_prune(node_i)

Operation 3: Node Merge
// Candidates: two nodes with very high semantic similarity
merge_score(i,j) = cos(nu_i, nu_j)
if merge_score &gt; tau_merge=0.92 AND both_nodes_stable(i,j):
merged_node_k = Node(
nu_k = bundle({nu_i, nu_j}; {0.5, 0.5}),
a_k = (a_i + a_j) / 2,
)
// Merge bond sets: take max strength for each (type, endpoint) pair
bond_merged = {(type,l): max(bond_il, bond_jl) for all (type,l)}
substrate.remove_node(i); substrate.remove_node(j)
substrate.add_node(merged_node_k)
SGP.ledger.record_merge(i, j, merged_node_k)

Operation 4: Attractor Expand
// Trigger: cleanup confidence &lt; tau_cleanup_conf=0.45 on repeated queries
if ERS.query_miss_rate(target_VSA, window=50) &gt; 0.60:
new_attractor = Attractor(
prototype_nu = bundle(failed_query_VSAs; {utility_score}),
tier = Episodic,
decay_state = 0.70,
utility_score= 0.30, // provisional — will be updated by consolidation
)
ERS.add_attractor(new_attractor)
SGP.ledger.record_attractor_expand(new_attractor)
12.3 Hard Budget Constraints

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Resource Lite (Kaggle
T4)

Base Frontier Enforc
ement
Max nodes 256 8,192 65,536 SGP
blocks
spawn
if
at_limit;
triggers
prune
sweep
first
Max attractors (all tiers) 2,048 100,000 1,000,000 Nightly
consoli
dation
prunes
to limit
Max temporal motifs (TAL) 512 50,000 500,000 TAE
retirem
ent on
interfer
ence or
low
succes
s rate
Max ERS working tuples 128 512 4,096 FIFO
eviction
Max ERS episodic tuples 10,000 100,000 1,000,000 Decay
+ utility
eviction
Max VSA dimension D 512 1,024 8,192 (16,384 Frontier) Fixed
at
session
start;
cannot
expand
mid-
session

Max new nodes per
checkpoint window

N_compress
(TAE) +
N_error_pool *
0.10

Same formula Same formula I-24
enforce
ment:
spawn
&lt;=
freed_b
y_com
pressio
n

12.4 Stability Gate on New Structures

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Every spawned node and expanded attractor must pass a stability gate within 50 pulses of
creation. Failure triggers rollback.
stability_gate(new_node_j, window=50_pulses) =
e_j^scalar &lt; 0.50 after 50 pulses // error reduced from spawn
trigger level
AND cos(nu_j, existing_nu) &lt; 0.92 // not redundant with existing
node
AND verifier.non_interference_check(j) // no degradation to committed
shapes
if NOT stability_gate: substrate.rollback_spawn(new_node_j)
12.5 Growth Ledger Schema
growth_ledger = {
session_id: UUID,
checkpoint_id: UUID,
spawned_nodes: [(node_id, parent_id, trigger_type, pulse_timestamp)],
pruned_nodes: [(node_id, prune_score, pulse_timestamp)],
merged_nodes: [(node_a, node_b, merged_id, pulse_timestamp)],
expanded_attrs: [(attractor_id, trigger_VSA, pulse_timestamp)],
nodes_freed_by_motifs: int, // from TAE.nodes_freed per session
I24_gate_checks: [(timestamp, spawned, freed, gate_pass: bool)],
net_node_delta: int, // spawned - pruned - merged_pairs
}

13. Dual-Speed Learning Engine (DSLE)
Slow learning: continuous local weight updates (W_ff, W_fb) via prediction-error
dynamics — the primary training mechanism. Fast learning: inference-time
adaptation via in-context conditioning, ERS retrieval, and within-episode belief
revision — no weight gradient required. Both streams run simultaneously and
interact through ERS.
13.1 Slow Stream
dW_ff_ij/dt = -e_i(t) * a_j^{L+1}(t) * lr_ff // continuously, in every pulse
dW_fb_ji/dt = -e_i(t) * a_j^{L+1}(t) * lr_fb
// lr_ff = lr_fb = 1e-4; cosine-annealed per developmental stage schedule
// Online update (adaptive-prod mode only): EWC-constrained, max 3
steps/episode, grad_norm &lt;= 0.01
L_online = L_error + 10.0 * SUM_i F_ii * (W_i - W_i^*)^2 // lambda_ewc=10.0
13.2 Fast Stream — Three Mechanisms
F1: In-Context Example Conditioning
// For each provided example (x_ex, y_ex) at session start:
S_in = encode(x_ex); S_out = encode(y_ex)
ERS_context.append({S_in, S_out, bond_type:&quot;demonstrated&quot;, tier:Working})
// During inference — VSA-level bias on nu_i:

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
delta_nu_i += 0.25 * SUM_j cos(nu_i, S_in_j.centroid) * (S_out_j.centroid -
S_in_j.centroid)
F2: ERS Retrieval Biasing
top_k_eps = ERS.top_k_cleanup(centroid_nu(active_shapes), k=8)
for ep in top_k_eps:
if cos(query_v, ep.input_VSA) &gt; 0.45:
nu_bias += cos_score * ep.trace_VSA * ep.utility_score
nu_i(t=0) += 0.15 * LayerNorm(nu_bias)
F3: Within-Episode Belief Revision
if verifier_veto(S_candidate):
shadow_S = promote_to_shadow(S_candidate)
ERS_context.append({S_candidate, verdict:&quot;vetoed&quot;, utility:-0.5})
frontier.add(new_branch(last_committed_S, exclude=shadow_S))
ESS.trigger(Frustration, bonded_to=S_candidate)

14. Budget Lattice and Authority System
b_i in R^16:
[0] pulse_remaining [1] memory_write_budget [2]
execution_privilege
[3] regime_tier [4] authority_lock [5]
budget_fraction_used
[6:10] per_phase_step_cost [10:14] escalation_log
[14] write_gate [15] action_gate
// Pulse cost: Perceptive=0, Compositional=1, Recursive=2, Commit=1
// Budget init: B_max=32 (Lite/Kaggle), 128 (Base), 512 (Frontier)
// Exhaustion -&gt; force Commit with best stable candidate (graceful degradation)
Authority
Regime

Name Permitted MPF State

Adaptive EXPLORE Full role switching, plasticity active, shape

formation, stochastic routing

phi unconstrained

Stable COMMIT Reduced role switching, stable shape membership,

routing deterministic

phi &lt; 0.30

Authority LOCK Role argmax only, fully deterministic, plasticity

frozen

phi = 0 (frozen)

15. Effect Class System

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Clas
s
Name Permitted Operations Escalation Required
E0 Pure Deterministic computation only; no state

read except constants

None

E1 ReadState Read canonical state snapshots and

memory

None

E2 WriteWorking Write to working memory; transient writes Controller shape approval
E3 WritePersistent Write to persistent episodic/semantic

memory

Evidence approval + provenance
envelope

E4 Simulate Invoke world model and shadow shapes;

no external effects

Controller SIMULATE tier
E5 ExecuteExternal Request real-world external effects Controller + verifier + proof

artifact

E6 Admin Schema, policy, system-bundle mutation Human approval + dual

verification

Render shapes are hard-capped at E2. Enforced in Rust authority service.
Any E3+ attempt from a render shape triggers alert and immediate veto
regardless of prompt content.

16. Developmental Curriculum (DC) — Stages D1 through
D6
Stage ordering is absolute. Language (D4) may not begin until sensorimotor (D1),
causal (D2), and social (D3) grounding are certified. Violating this order produces
the LLM failure mode: statistical language without grounded understanding. Gate
certification is enforced by the data-loader, not by convention.
16.1 Stage D1 — Sensorimotor Grounding
Goal: The substrate learns that the world has persistent objects that move according to laws. No
language. No abstract relations.

Aspect Specification
Primary training signal Prediction error on next-state prediction in spatial environments: given frame t,
predict frame t+1. Error = pixel-level or feature-level MSE on reconstructed next
state.

Data Source A —
ARC-AGI grids (60%
of D1 data)

ARC-AGI training grids used as spatial pattern-completion tasks. The substrate
sees a grid and predicts the output grid. No language labels. Grid cells are
encoded directly as node activations via occupancy geometry. Forces genuine
spatial structure learning with strong data efficiency pressure — each ARC task is

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications

a unique rule. The system cannot memorize; it must induce.

Data Source B —
Synthetic relational
graphs (40% of D1
data)

Generator produces structured (entity, relation, entity) triples in graph form. One
element is masked; the system must predict it. Relations: spatial (above, inside,
left-of), causal (enables, blocks, causes), temporal (before, during, after). Fully
controllable: difficulty scales with graph depth and relation type. Fast iteration —
can generate millions of distinct instances.

D1 gate: Object
tracking

Object tracking accuracy &gt;= 0.90 across occlusion events in held-out ARC grids

D1 gate: Causal
prediction

Causal prediction error &lt; 0.20 on held-out synthetic relational graph instances

D1 gate: Temporal
trace

Temporal trace persistence &gt; 5 pulses for tracked objects in held-out sequences
D1 gate: VSA binding VSA cleanup accuracy &gt; 0.90 on synthetic binding recovery tasks (bind-unbind-

recover)

Kaggle session
estimate

2–3 sessions (256-node Lite model, D=512, ARC training set + synthetic
generator)

16.2 Stage D2 — Causal and Relational Grounding
Goal: The substrate learns that events have structure beyond temporal co-occurrence — cause-
effect schemas, precondition-action-consequence chains, and abstract logical relations.

Aspect Specification
Primary training signal Intervention prediction: given a causal graph and an intervention on one node,
predict the downstream effects. Error = discrepancy between predicted and
ground-truth effect set. Also: masked-relation prediction on abstract analogy tasks
(A:B :: C:?).

Data Source A —
ARC-AGI grids with
causal rules (50% of
D2 data)

Select ARC tasks that embody causal rules (e.g., &quot;if cell has color X and is
adjacent to color Y, output is Z&quot;). The system must induce the rule from 1–3
training pairs and apply it. Causal bond types (P_1) are the primary bond class
trained in D2.

Data Source B —
Synthetic causal
graphs with
interventions (50% of
D2 data)

Generator produces directed acyclic graphs with typed causal edges. Intervention:
set one node to a specified value and observe the propagation. Masks a
downstream node and requires prediction. Includes counterfactual variants: &quot;if X
had been Y instead, what would Z be?&quot; Directly trains the world model and
shadow-shape capabilities.

D2 gate: Causal
intervention

Intervention prediction accuracy &gt;= 0.80 on held-out causal graphs (unseen
structures)

D2 gate: Abstract
analogy

Abstract analogy score &gt;= 0.75 on held-out ARC tasks with causal rule structure

D2 gate: VSA causal
bonds

Bond recall for causal relation types (P_1: causal, P_2: temporal) &gt;= 0.90

D2 gate:
Compositional split

CompSplit_score &gt;= 0.65 on held-out compositions of learned causal primitives

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Kaggle session
estimate

2–3 sessions

16.3 Stage D3 — Social and Theory-of-Mind Grounding
Goal: Social shapes, social bonds, and theory-of-mind reasoning. The substrate learns to model
other agents: their goals, their beliefs, and — critically — the fact that agents can hold beliefs
that are false.

Aspect Specification
Primary training signal Goal inference and false-belief prediction. Given a partial observation of an
agent&#39;s actions, predict: (a) the agent&#39;s goal, (b) the agent&#39;s current belief about
the state of the world, and (c) what the agent will do next. Error = cross-entropy on
predicted goal, belief, and action distributions.

Data Source A —
Agent observation
episodes (60% of D3
data)

Synthetic episodes: agents with assigned goals act in simple grid-world
environments. Partial observability: the system only sees the agent&#39;s actions, not
their internal goal state. Must induce the goal. Includes false-belief scenarios:
agent acts on a belief that is now out of date (object moved while agent was
absent).

Data Source B —
Cooperative/competitiv
e interaction traces
(40% of D3 data)

Two-agent episodes where agents must coordinate or compete. The system
models both agents simultaneously. This trains the social shape to represent
multiple agent models at once and to reason about their interactions.

D3 gate: Goal
inference

Goal inference accuracy &gt;= 0.75 from partial observation on held-out agent
episodes

D3 gate: False-belief
task

False-belief task accuracy &gt;= 0.75 on held-out scenarios

D3 gate: Trust
calibration

Social shape trust calibration RMSE &lt; 0.15 on synthetic agents with known
success rates

D3 gate: Emotional
shapes

Emotional shape induction accuracy &gt;= 0.70 on held-out scenarios where agent
emotional state is labeled

Kaggle session
estimate

1–2 sessions

16.4 Stage D4 — Linguistic Grounding
Goal: Language as VSA shape-bundles grounded in D1–D3 structure. The USL codec is
trained. The Lexical Attractor Library is populated. Inner speech loop is activated. Language is
learned as a code for already-understood structure, not as a statistical surface.

Aspect Specification
Primary training signal USL round-trip fidelity: encode text to shapes (via USL_understand), then decode

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications

back to text (via USL_generate), and measure semantic similarity between
original and reconstructed text. Also: language-grounded prediction — given a
verbal description of a causal event, predict the outcome in the spatial domain.

Data Source A —
ARC-AGI grids with
verbal rule descriptions
(40% of D4 data)

ARC training pairs annotated with natural language descriptions of the
transformation rule (e.g., &quot;move all blue cells to the right until they hit a wall&quot;). The
system must simultaneously induce the spatial shape and the linguistic shape
describing the same rule, and bind them.

Data Source B —
Grounded causal
language (35% of D4
data)

Sentences describing causal relations and events that were represented as
shapes in D2 (e.g., &quot;the red block pushed the blue block off the table&quot;). The
linguistic shape must bond to the pre-existing causal shape via PREDICATE_OF
and SUBJECT_OF bonds. No ungrounded text.

Data Source C —
Concept-description
pairs (25% of D4 data)

Short noun phrases paired with their shape representations from D1–D3: colors,
spatial relations, object types, simple properties. Populates the LAL foundation
vocabulary.

D4 gate: USL round-
trip

USL round-trip fidelity &gt;= 0.85 on held-out concepts (encode then decode
recovers original shape)

D4 gate: Language-
grounded analogy

Language-described analogy completion &gt;= 0.80 on held-out ARC-language pairs

D4 gate: ISL
coherence

ISL produces semantically coherent inner speech on &gt;= 70% of Recursive regime
episodes in held-out tasks

D4 gate: Linguistic
bonds

All 10 linguistic bond types: VSA recovery accuracy &gt;= 0.90 on held-out bond
extraction tasks

Kaggle session
estimate

3–4 sessions (LAL population is the expensive step)

16.5 Stage D5 — Abstract Reasoning and Formal Competence
Goal: Mathematical reasoning, logical inference, formal argument structure, counterfactual
reasoning, multi-step planning. Built on D2 causal and D4 linguistic grounding. Language is now
a cognitive tool for abstract thought.

Aspect Specification
Primary training signal Task completion with verifier-backed ground truth: math problems (answer must
be verifier-checked), logic inference (conclusion must follow from premises,
verifier checks), multi-step ARC problems (output grid checked against ground
truth). Also: few-shot adaptation loss (L_fast_adapt) activated.

Data Source A —
ARC-AGI full training
set (40% of D5 data)

All ARC-AGI tasks. By D5 the system has D1–D4 grounding; it approaches ARC
as genuine rule induction over grounded spatial-causal representations, not
pattern-matching. Meta-learning episode structure (k=1,3,5) is applied.

Data Source B —
Verifier-backed math
and logic (35% of D5
data)

Elementary to intermediate math problems with step-by-step verifier traces.
Formal logic deduction tasks. All intermediate steps are preserved — not just
input/output. The system must produce verifier-consistent reasoning, not just
correct final answers.

Data Source C —
Multi-step planning

Planning tasks in synthetic environments: given a goal and a starting state,
produce a sequence of actions that achieves the goal. Plan verified by simulation

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
(25% of D5 data) (E4 effect class). Temporal motif formation accelerates replanning on familiar

problem structures.

D5 gate: ARC-AGI
improvement

ARC-AGI RHAE score improvement over D4 baseline on held-out evaluation grids

D5 gate: Math
verification

Math step verification rate &gt;= 0.90 on held-out problem types

D5 gate: Planning
success

Multi-step planning success &gt;= 0.80 on held-out task families

D5 gate:
Compositional split

CompSplit_score &gt;= 0.75 on held-out compositions of reasoning primitives from
D2–D4

Kaggle session
estimate

4–6 sessions (most compute-intensive stage)

16.6 Stage D6 — Full Integration and Meta-Learning
Goal: All systems active simultaneously. Meta-learning curriculum runs across all task families.
Curiosity goals accumulate and drive self-directed learning. Temporal motifs form for common
reasoning patterns. Growth protocol is fully active.

Aspect Specification
Primary training signal All 26 loss components active. Meta-episode structure: k in {1,3,5} examples
provided before each task, query loss measured after conditioning. ERS retrieval
evaluated against post-hoc utility. SGP growth and pruning tracked by ledger.

Data Source A — All
D1–D5 data revisited
in meta-episode format
(50% of D6 data)

The full developmental corpus is re-presented with meta-learning episode
structure: support set of k examples followed by a query task. This trains the
DSLE fast stream to use the ERS and in-context conditioning reliably.

Data Source B —
Language interface
and instruction
following (30% of D6
data)

Diverse natural language tasks: explanation, summarization, question answering,
code generation, structured analysis. All processed through USL — language is
shapes, not tokens. Output is rendered via USL_generate from committed
shapes.

Data Source C —
Social learning traces
(20% of D6 data)

High-quality reasoning traces from verified problem solvers (math olympiad
solutions, verified code, formal proofs). SLS reconstructs these as synthetic ERS
tuples. Also: self-distillation — the system&#39;s own D5 verified reasoning traces are
re-ingested.

D6 gate: Sample
Complexity Ratio

SCR &gt;= 10x on 3 of 5 held-out families; SCR &gt;= 5x on all 5 held-out families

D6 gate: Few-shot
efficiency

k=1 performance &gt;= 40% of full-data; k=5 performance &gt;= 70% of full-data

D6 gate: CompSplit CompSplit_score &gt;= 0.75 across 5 held-out task families
D6 gate: Growth
ledger

I-24 satisfied: net_node_delta &lt;= nodes_freed_by_motifs at every checkpoint in
D6

D6 gate: Anti-forgetting Protected-family regression &lt;= 2% after 1000 online adaptation steps on disjoint

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications

families
D6 gate: Human parity
profile

All five human parity dimensions pass (Section 19)

Kaggle session
estimate

Ongoing — D6 is the continuous operational phase, not a terminal stage

17. Kaggle Session Plan — Constrained Implementation
PSN-2 is designed to train and grow across constrained sessions. Each Kaggle
session is a self-contained unit that: loads checkpoint from persistent storage, runs
training for up to 9 hours, saves checkpoint to persistent storage, and exits cleanly.
Growth happens across sessions, not within them.
17.1 Hardware Constraints and Configuration

Resource Kaggle T4 Kaggle P100 Design Response
GPU VRAM 2 × 16GB 1 × 16GB Cap D=512 (Lite). With
D=512, 256-node substrate =
~130MB parameters — fits
comfortably.

RAM 30GB 17GB ERS Working tier capped at
128 tuples. Consolidation
runs at session end, not
during training.

Disk (persistent) 20GB (Kaggle
Dataset)

20GB Checkpoint schema designed
for &lt; 500MB per checkpoint.
Growth ledger and ERS
Episodic tier are the large
components.

Session duration 9 hours hard limit 9 hours hard limit Each session targets one
sub-stage or gate milestone.
Auto-save every 30 minutes.

Internet Available during
training

Available during
training

ARC-AGI dataset and
synthetic generator can be
loaded from Kaggle Datasets.
No external API calls during
training.

17.2 Checkpoint Schema
checkpoint = {

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
// Identity
checkpoint_id: UUID,
session_id: UUID,
stage: &quot;D1&quot;|&quot;D2&quot;|...|&quot;D6&quot;,
sub_stage: string, // e.g., &quot;D1_ARC_phase1&quot;
timestamp: ISO8601,
config_hash: SHA256, // hash of all frozen constants
software_hash: SHA256,
random_seeds: {python, numpy, torch, cuda},
// Substrate state
node_states: Tensor[N_nodes, node_state_dim], // u_i for all active
nodes
bond_states: SparseTensor[N_nodes, N_nodes, |T_bond|],
W_ff: Tensor[L, d_out, d_in],
W_fb: Tensor[L, d_in, d_out],
W_pred: Tensor[L, D, d_c],
W_morpho: Tensor[d_m, d_m],
// Attractor libraries
attractor_codebook: Tensor[N_attr, D],
LAL: Tensor[vocab_size, D + D_s + D_p + 16], // word shapes
TAL: List[temporal_motif_shape], // temporal
motifs
// ERS state
ERS_working: List[ERS_tuple], // &lt;= 128 tuples
ERS_episodic: List[ERS_tuple], // &lt;= 10000 tuples; compressed
curiosity_goals: List[curiosity_goal], // pending CE goals
// Growth state
growth_ledger: growth_ledger_schema,
SGP_budget: {nodes_spawned, nodes_freed, attractors_expanded,
I24_pass},
// Training state
optimizer_state: AdamW state dict,
lr_schedule: current_step + schedule_config,
stage_gate_results: {gate_name: (value, pass_bool)},
training_loss_history: List[float], // last 1000 steps
}
Checkpoint size estimate for Lite (256 nodes, D=512): node_states ~130KB, bonds
~2MB (sparse), weights ~8MB, attractor library ~4MB (2048 * 512 * 4 bytes), ERS
~5MB compressed, LAL ~16MB (8192 * (512+512+256+16) * 4 bytes), growth
ledger ~100KB. Total: ~35–40MB per checkpoint. Well within 20GB Kaggle
persistent storage.
17.3 Session Sequence

Sess Stag Primary Goal Data Loaded Expected Gate Progress

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
ion e
1 D1-a Initialize substrate: node
states, VSA algebra,
Phases A–F, spike
engine. Train on ARC-
AGI grids only. Establish
prediction-error
dynamics.

ARC-AGI training set
(400 tasks). Synthetic
relational graph
generator initialized.

VSA binding recovery &gt; 0.90.
Substrate is stable and runs without
OOM.

2 D1-b Add synthetic relational
graphs. Train mixed D1
curriculum. Growth
protocol activated for first
time — spawn pressure
accumulates but budget
gate prevents premature
growth.

Mixed: 60% ARC, 40%
synthetic relational.
D=512, 256 nodes.

D1 object tracking gate &gt;= 0.90. D1
causal prediction gate &lt; 0.20 error.

3 D2-a Causal grounding begins.
ARC tasks with causal
rules. Causal bond types
P_1, P_2 trained. First
node spawns expected in
high-error causal regions.

ARC-AGI causal-rule
subset + synthetic
causal graphs with
intervention.

D1 all gates certified. D2 causal
intervention gate &gt;= 0.70 (partial).

4 D2-b Causal grounding
completion. False-belief
grounding begins (D3
preview). CompSplit
evaluation on causal
primitives.

Synthetic causal
graphs + early agent
observation episodes.

D2 all gates certified. D3 goal
inference gate &gt;= 0.60 (partial).

5 D3 Social grounding. Social
shapes, emotional
shapes trained. Trust
system calibrated. False-
belief tasks.

Agent observation
episodes +
cooperative/competitiv
e traces.

D3 all gates certified. Emotional
shape induction accurate.

6 D4-a LAL population begins.
USL understanding
direction trained.
Language-grounded
ARC tasks introduced.

ARC + verbal
descriptions (40%) +
concept-description
pairs (25%). Grounded
causal language
(35%).

LAL populated to 4096 words.
USL_understand fidelity &gt;= 0.75
(partial).

7 D4-b USL generation direction.
ISL activated. Round-trip
fidelity evaluated.

Same as session 6.
ISL active.

D4 all gates certified. USL round-trip
&gt;= 0.85. ISL coherent &gt;= 0.70.

8 D5-a Abstract reasoning
begins. Full ARC-AGI
training set. Verifier-
backed math. Few-shot
meta-episode structure
introduced (k=1).

All ARC training tasks
+ math step traces +
planning tasks. k=1
meta-episodes.

ARC-AGI RHAE improvement over
D4 baseline. Math verification rate
&gt;= 0.80.

9 D5-b Meta-learning k={1,3,5}
active. Temporal motifs
forming for common ARC
patterns. SCR

Same as session 8 +
k=3, k=5 meta-
episodes. TAE motif
quarantine active.

D5 all gates certified. SCR &gt;= 5x on
2 families. Temporal motifs: &gt;= 3
pulses saved.

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications

measurement begins.
10+ D6 Full integration. Self-
distillation. Curiosity
goals accumulate.
Growth balanced by
compression. Language
interface tasks.

All D1–D5 data in
meta-episode format +
language interface +
social learning traces.

D6 gates: SCR &gt;= 10x, CompSplit
&gt;= 0.75, I-24 satisfied, human parity
profile.

18. Unified Loss Family — 26 Components
All weights frozen at D1 training start. May not drift during training or be modified without
reopening the constant registry and issuing a new PRD version.

# Symbol Weight Purpose Stage
Active

w1 L_error 0.12 Multi-step prediction-error minimization — the

core learning law

D1+

w2 L_bond 0.08 VSA bond accuracy and typed relation

supervision

D1+
w3 L_shape 0.08 Shape stability and dissolution hinge losses D1+
w4 L_compact 0.12 Expected description length — dominant;

prevents bloat

D1+
w5 L_refine 0.07 Negative per-pulse stability improvement D2+
w6 L_world 0.07 World model MSE and counterfactual accuracy D2+
w7 L_verify 0.09 Verifier veto classification + unsupported-claim

loss

D2+
w8 L_calib 0.06 Prediction error calibration vs actual correctness D2+
w9 L_transfer 0.06 Efficiency delta with vs without motifs (motif

utility)

D5+

w1
0
L_render 0.03 Surface generation from committed state — kept

low intentionally

D4+

w1
1
L_vsa 0.06 VSA binding recovery and cleanup accuracy D1+
w1
2
L_mpf 0.04 phi low on solved tasks, high on
novel/mismatched tasks

D1+

w1
3
L_attractor 0.03 MPF attractor basins well-separated D1+
w1
4
L_sym 0.04 W_ff/W_fb cosine symmetry floor; activates when

sym_L &lt; 0.70

D1+

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
w1
5
L_attn 0.04 External attention retrieval quality D4+
w1
6
L_expr 0.04 Internal expression causal quality D4+
w1
7
L_geom 0.03 Cross-geometry consistency (all 7 geometry

spaces)

D1+

w1
8
L_compart 0.03 Compartment disentanglement (5 input

compartments)

D1+

w1
9
L_fast_adapt 0.08 Query-set prediction error given support-set

conditioning (meta-loss)

D5+

w2
0
L_retrieval 0.05 ERS utility score prediction accuracy D5+
w2
1
L_comp_split 0.07 Performance on held-out compositional test

instances

D5+

w2
2
L_forget 0.06 EWC forgetting penalty during online update

steps

D6

w2
3
L_linguistic 0.07 USL round-trip fidelity: encode → decode →

encode consistency

D4+

w2
4
L_emo_calib 0.05 Emotional shape calibration vs post-hoc outcome

ground truth

D3+

w2
5
L_social 0.05 Social trace reconstruction verifier agreement

rate

D3+

w2
6
L_temporal 0.06 Temporal motif pulse savings &gt;= 3 vs step-by-

step

D5+

normalization = SUM(w1..w26) = 1.98
L_total = SUM_{i=1}^{26} w_i * L_i / 1.98

19. Evaluation Framework and Human Parity Test
19.1 Five-Dimensional Scorecard

Dimension Question Benchmarks Release Gate
Reasoning
Integrity

Does the system
reason correctly and
verifiably?

ARC-AGI-3 (RHAE), BIG-
Bench Hard, math reasoning,
multi-step planning, verifier
stress tests

RHAE positive on all 6
environment classes.
False-commit rate &lt; 0.5%.
Verifier pass &gt; 95%.

Language Quality Is language fluent,
grounded, and
pragmatically

Human evaluation:
naturalness and coherence.
Grounding test: does output

Human rater score &gt;=
4.0/5.0. Grounding
violation rate &lt; 1%. USL

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
appropriate? assert only verified content?
USL round-trip fidelity.

round-trip &gt;= 0.85.

Learning
Efficiency

Does the system learn
from few examples
and transfer across
tasks?

Sample Complexity Ratio vs
matched transformer. k=1,3,5
efficiency curves. CompSplit.
Online adaptation over 10
episodes.

SCR &gt;= 10x on 3/5
families. k=1 &gt;= 40% of
full-data. CompSplit &gt;=
0.75.

Energy Efficiency Does inference energy
scale with task
difficulty?

Mean sparsity by task type.
Energy per inference vs
dense transformer.
Neuromorphic deployment
energy.

Mean sparsity &gt;= 80% on
familiar tasks. Energy per
inference &lt;= 5% of
matched transformer on
familiar inputs.

Cognitive Unity Does the system
behave as one unified
system?

Every system ablation (USL,
SCE, ESS, ISL, CE, SLS,
TAE, SGP) must degrade &gt;=
2 evaluation dimensions
simultaneously. No ablation
may improve any dimension.

All cross-system ablations
show multi-dimensional
degradation.

19.2 Human Parity Test

Dimension Human Expert Baseline PSN-2 Target Measurement
Reasoning PhD-level on novel
structured problems

&gt;= 85% of expert solve
rate

Novel ARC-AGI-3
environments not in any
training set

Language Native speaker fluency and

coherence

&gt;= 85% human rater
agreement on
naturalness

Blind A/B: PSN-2 vs
human on same prompts

Learning speed 1–3 examples to grasp new

concept

Within 2x of human k-
shot performance

Head-to-head few-shot
on novel concept
categories

Energy ~20W continuous &lt;= 50W at full load on
neuromorphic

Power meter on Loihi
deployment during
benchmark

Robustness No catastrophic forgetting

of old skills

&lt;= 2% regression after
learning 10 new task
families

Sequential acquisition
with protected-family
testing after each

19.3 SGP-Specific Evaluation Tracks

Track Question Protocol Gate
Growth utility Does growth improve
performance without

Measure accuracy before and
after 5 spawn events on the

Performance
in spawned

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications

degrading existing
capabilities?

region that triggered
spawning. Compare
protected-family accuracy.

region
improves &gt;=
10%.
Protected
families: no
regression.

Compression balance (I-
24)

Does compression keep pace
with growth?

Monitor net_node_delta and
nodes_freed_by_motifs over
100-checkpoint windows.

I-24
satisfied:
net_node_de
lta &lt;=
nodes_freed
_by_motifs
at every 10-
checkpoint
rolling
window.

Stability gate
effectiveness

Do stability gates catch bad
spawns?

Run 100 deliberate &quot;bad
spawn&quot; tests: spawn nodes in
low-error regions and verify
rollback.

Rollback rate
&gt;= 95% for
nodes in low-
error
regions. No
false
positives on
high-error
region
spawns.

Session continuity Does the system resume
correctly after checkpoint
reload?

Load checkpoint, run 100
inference steps, compare
outputs to pre-checkpoint
deterministic replay.

Replay
fidelity &gt;=
99.9% within
declared
numeric
tolerance.

20. Default Constants — All Frozen at D1 Start
These constants define the PSN-2 class. Modifications require reopening the constant registry
and issuing a new PRD version. No constant may drift during training.

Constant Lite
(Kaggle)

Base Frontier Description

D (VSA dim) 512 1024 8192/1638
4

Hypervector dimensionality
N_nodes (initial) 256 8192 65536 Node population at D1 start
B_max (pulse budget) 32 128 512 Max pulses per episode
W_max (write budget) 4 16 64 Max memory writes per episode

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
lambda_e 0.90 0.90 0.90 Bond decay rate per pulse
lambda_tau 0.90 0.90 0.90 Temporal trace decay rate
lambda_emo 0.85 0.85 0.85 Emotional bond decay rate (faster than

cognitive)

lr_ff = lr_fb 1e-4 1e-4 1e-4 Feed-forward/feedback weight update LR

(baseline)

tau_c 0.55 0.55 0.55 Content similarity threshold for shape join
tau_r 0.20 0.20 0.20 Bond connectivity threshold for shape join
tau_dissolve 0.25 0.25 0.25 Shape stability floor below which shape

dissolves

phi_max_join 0.25 0.25 0.25 Max MPF for shape join eligibility
alpha_bind 0.15 0.15 0.15 Shape content pull toward centroid on join
tau_veto 0.60 0.60 0.60 Verifier veto threshold on conflict mass
tau_unsupported 0.10 0.10 0.10 Max unsupported-claim ratio before verifier

veto

tau_spike_base_Percept 0.30 0.30 0.30 Spike threshold Perceptive base
tau_spike_base_Comp 0.20 0.20 0.20 Spike threshold Compositional base
tau_spike_base_Recurs 0.10 0.10 0.10 Spike threshold Recursive
tau_spike_base_Commit 0.40 0.40 0.40 Spike threshold Commit
tau_spawn 0.65 0.65 0.65 Prediction error threshold for node spawn

trigger

N_persist 10 10 10 Consecutive pulses above tau_spawn
required before spawn eligible
sigma_spawn 0.05 0.05 0.05 Noise std for new node nu_i initialization
tau_prune 0.75 0.75 0.75 Prune score threshold for node removal
tau_age_prune 200 200 200 Minimum age (pulses) before node eligible

for pruning

tau_merge 0.92 0.92 0.92 Cosine similarity threshold for node merge
stability_gate_window 50 50 50 Pulses after spawn for stability gate

evaluation

tau_cleanup_conf 0.45 0.45 0.45 Cleanup confidence below which attractor

expansion triggers

N_max_relational 20 20 20 Max training instances per relational pattern

(data poverty)

N_max_compositional 50 50 50 Max training instances per compositional

pattern

LAL_vocab_size 8192 65536 262144 Word shapes in Lexical Attractor Library
D_syntactic 256 512 2048 Syntactic geometry VSA dimensionality

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
D_pragmatic 128 256 512 Pragmatic geometry VSA dimensionality
ISL_trigger_uncertainty 0.55 0.55 0.55 Uncertainty threshold for ISL activation
alpha_icl 0.25 0.25 0.25 In-context example conditioning bias weight
alpha_ers 0.15 0.15 0.15 ERS retrieval bias weight at pulse t=0
tau_ers 0.45 0.45 0.45 Min cosine for ERS retrieval to influence

predictions

lambda_ewc 10.0 10.0 10.0 EWC forgetting penalty coefficient
max_online_grad_norm 0.01 0.01 0.01 Max gradient norm per online update step
SCR_min_all_families 5.0 5.0 5.0 Minimum SCR on all 5 held-out families
SCR_target_3_families 10.0 10.0 10.0 Target SCR on at least 3 of 5 families
CompSplit_gate 0.75 0.75 0.75 Minimum CompSplit_score mean across 5

families

sym_floor 0.70 0.70 0.70 Minimum W_ff/W_fb cosine; triggers L_sym

below this

21. Software Architecture and Repository Layout
21.1 Technology Stack

Layer Technology Reason
Model research Python 3.12 + PyTorch 2.x Fast iteration, torch.compile, FSDP,
autograd through surrogate gradients

VSA operations PyTorch tensor ops + optional
Triton post-profiling

Bind/bundle/cleanup are matmuls and
element-wise ops; highly vectorizable

SCE spike kernel PyTorch custom op; surrogate
gradient via register_hook

Spike is non-differentiable; surrogate
handles backprop

Authority/verifier Rust service Determinism, memory safety, no GC pauses

in verification hot path

ERS storage PostgreSQL + pgvector
(episodic/semantic); in-memory
dict (working)

pgvector cosine similarity for VSA retrieval;
dict for speed

Checkpoint I/O PyTorch state_dict + JSON
sidecar (ledger, ERS, constants)

Portable; survives Kaggle session boundary

Distributed training PyTorch FSDP + single-GPU

fallback

Kaggle sessions are single-T4; FSDP for
multi-GPU environments

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
21.2 Repository Layout
/psn2
/configs
constants.json # P0-frozen constants hash-pinned manifest — source
of truth
stage_gates.json # Gate thresholds per developmental stage
loss_weights.json # 26 loss component weights
/src/psn2
/core
node.py # Node state: u_i, g_i (growth state), all field
dtypes
error.py # Error node: e_i computation, local update law
bond.py # VSA bond: typed hypervector edges, bond_ij^T
update
vsa.py # bind, bundle, perm, cleanup, codebooks
shape.py # Shape: 6-condition join, legality, role
assignment
field.py # Predictive Field Layer: 6-phase cycle
orchestration
commitment.py # theta_commit, verifier_veto, Phase F, receipts
budget.py # Budget lattice, authority state, graceful
degradation
regimes.py # EXPLORE / COMMIT / LOCK
controller.py # Regime controller, field statistics
canonical_state.py # Canonical slots, write rules, receipt emission
plasticity.py # W_ff/W_fb update, symmetry monitoring, L_sym
/sce
spike.py # Spike decision, surrogate gradient, silent node
update
sparsity.py # Population sparsity monitoring, energy accounting
/usl
lal.py # Lexical Attractor Library
codec.py # USL_understand() and USL_generate()
syntax_encoder.py # Text → bond proposals (lightweight dependency
parser)
syntax_decoder.py # Bond structure → word ordering
pragmatics.py # Pragmatic geometry → register/politeness
rendering
isl.py # Inner Speech Loop
linguistic_bonds.py # Bond types P_10–P_19
/ess
emotional_shapes.py # 8 emotional shape types, induction, decay
emotional_bonds.py # perm_EMO, delta_theta_commit
/ce
curiosity.py # Curiosity event detection, goal generation,
scheduling
/sls
social_shape.py # Social shape, trust update
trace_reconstructor.py# Observed trace → synthetic ERS tuple
self_distillation.py # Own verified traces as social data
/tae
motif_detector.py # Temporal motif candidate detection
motif_replay.py # Retrieval and pulse-sequence injection

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
tal.py # Temporal Attractor Library
/sgp
spawn.py # Node spawn: trigger, initialization, bond
attachment
prune.py # Node prune: score, non-interference check, bond
redistribution
merge.py # Node merge: similarity check, bond union
attractor_expand.py # Attractor expansion: miss-rate trigger, prototype
seeding
ledger.py # Growth ledger: I-24 gate, budget accounting
stability_gate.py # 50-pulse stability evaluation for spawned nodes
/dsle
fast_adapt.py # F1 (ICL), F2 (ERS retrieval), F3 (belief
revision)
online_update.py # EWC-constrained weight update; adaptive-prod mode
only
/ers
ers.py # ERS tiers, write, retrieval, promotion,
consolidation
utility.py # utility_score computation and lazy recomputation
/dc
stage_d1.py # D1 curriculum: ARC grids + synthetic relational
graphs
stage_d2.py # D2 curriculum: causal intervention + abstract
analogy
stage_d3.py # D3 curriculum: agent observation + false belief
stage_d4.py # D4 curriculum: linguistic grounding + USL
training
stage_d5.py # D5 curriculum: ARC-AGI + math + planning
stage_d6.py # D6 curriculum: full integration + meta-learning
gate_certifier.py # Measures all gate metrics; blocks stage
progression
data_loader.py # Pattern hashing, N_max enforcement, meta-episode
construction
arc_loader.py # ARC-AGI grid loading, encoding, compositional
split
relgraph_generator.py # Synthetic relational graph generator (D1/D2)
/losses
# 26 separate loss modules (L_error.py through L_temporal.py)
/eval
scr.py, few_shot.py, online_adapt.py, anti_forget.py, comp_split.py
sgp_eval.py # Growth utility, I-24 verification, stability gate
audit
human_parity.py # Five-dimension parity scorecard
/rust
/authority # LOCK-regime deterministic service
/verifier # Commit-time verification, veto, receipt signing
/checkpoints
checkpoint_manager.py # Save, load, validate, replay-verify across Kaggle
sessions
/tests
/unit # Per-module equation tests
/invariant # I-1 through I-24 enforcement tests
/determinism # Replay consistency tests
/sgp # Growth, prune, merge, stability gate, I-24 audit
/developmental # D1–D6 gate certification pipeline

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
22. Security Model

Threat Mitigation
Prompt injection into
canonical state

All canonical writes routed through authority service; content-addressed
and receipt-linked. Render shapes hard-capped at E2 in Rust.
Verifier bypass Rust authority service validates verifier receipt before any output is legal.

No trained-weight-only enforcement.

Privilege escalation b_i[execution_privilege] = min(task_contract, authority_approval). Cannot

self-grant.

Motif poisoning Motif and temporal motif admission offline-only. Eight-gate and four-gate

evaluation required respectively.

Growth instability attack Adversarial inputs designed to trigger maximal spawning. SGP budget gate
(I-24) hard limits growth rate. Stability gate rolls back bad spawns within 50
pulses.

Social trace poisoning Trust &lt; 0.30 traces isolated to Working tier. Verifier checks all
reconstructed traces before Episodic promotion. Trust decays 0.15 per
verifier disagreement.

Weight transport attack Symmetry gate: sym_L &lt; 0.70 triggers L_sym. Degenerate checkpoints

rejected at release.

Replay inconsistency Seeds, config hash, software hash, dataset manifest hash, receipt stream

recorded per run. Replay CI gate.

Memory namespace pollution Namespace isolation enforced in authority service. write_gate requires

namespace token.

ERS utility gaming Utility_score recomputed at retrieval, not only at write. Hit rate gate (&gt;=

0.60) evaluated on unseen families.

23. Build Order for an Autonomous Implementation Agent
Each step must be completed and validated before the next begins. No step may be skipped.
Steps are ordered by dependency, not by complexity.
1. Implement the constant registry (configs/constants.json): hash-pinned JSON manifest for
all constants. All subsequent code reads from this file. No magic numbers anywhere in
the codebase.
2. Implement node state schema (core/node.py): u_i with all fields including g_i growth
state. Write shape tests verifying correct dtype and dim for Lite config (D=512, N=256).

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
3. Implement error node (core/error.py): e_i computation from a_i and pred_i. Local weight
update law for W_ff and W_fb. Write test: from random init, W_ff and W_fb converge
toward sym_L &gt; 0.70 on a simple prediction task.
4. Implement VSA algebra (core/vsa.py): bind, bundle, perm, cleanup with soft and hard
modes. Write binding recovery test: for random x,y at D=512, cleanup(bind(x,y),y)
recovers x with accuracy &gt; 0.90.
5. Implement VSA bond update (core/bond.py): q_ij^T proposal, bond_ij^T persistent
update with lambda_e. Write edge persistence test: bond_ij^T at pulse t+10 &gt; 0.50 of
bond_ij^T at pulse t for a stable pair.
6. Implement spiking computation engine (sce/spike.py): spike decision rule per regime,
surrogate gradient via register_hook. Write sparsity test: on zero-error input (nu_i = a_i),
spike rate &lt; 5% of nodes.
7. Implement morphogenic MLP and MPF (core/shape.py phase C): role, type, emotional
distribution update. Phi update equation. Write test: phi &lt; 0.10 on tasks solved in 3
pulses; phi &gt; 0.40 on novel tasks.
8. Implement the six-phase Predictive Field Layer (core/field.py): Phases A through F in
strict order. Write gradient flow test for all six phases.
9. Implement six-condition shape join gate (core/shape.py phase D): all six conditions,
centroid, alpha_bind pull, stability, dissolution. Write formation test: shapes form within
20 pulses on structured synthetic input.
10. Implement verifier-gated commitment (core/commitment.py): theta_commit with all
terms, verifier_veto, Phase F logic. Write test: commit gate rejects high-conflict
candidate (all six conditions present).
11. Implement canonical state protocol (core/canonical_state.py): six slots, write rules,
append-only receipt with all required fields.
12. Implement budget lattice and authority system (core/budget.py): 16-slot b_i, update law,
graceful degradation on exhaustion. Write test: verify B_max=32 (Lite) forces graceful
degradation, not silent overrun.
13. Implement regime controller (core/controller.py): field_stats, regime MLP, hard argmax
at inference.
14. Implement ERS (ers/ers.py): four tiers, write_score gate, VSA retrieval, decay,
consolidation stub. Implement utility_score computation (ers/utility.py).
15. Implement the Substrate Growth Protocol (sgp/): spawn.py, prune.py, merge.py,
attractor_expand.py, ledger.py, stability_gate.py. Write I-24 gate test: verify that
spawn_count &lt;= freed_count is enforced and spawn is blocked when budget is
exhausted.
16. Implement the Temporal Abstraction Engine (tae/): motif_detector, motif_replay, TAL.
Write motif detection test: a pulse sequence repeated 5 times across 5 episodes with
similarity &gt; 0.75 is nominated as candidate.
17. Implement ERS fast stream (dsle/fast_adapt.py): F1 (ICL), F2 (ERS retrieval bias), F3
(belief revision loop). Write F1 test: with 1 example provided, nu_i bias shifts toward
example output VSA.
18. Implement Curiosity Engine (ce/curiosity.py): event detection, goal shape generation,
ERS write, scheduling, resolution. Write lifecycle test: event detected → goal stored →
goal retrieved in next episode → budget extended → resolution logged.
19. Implement Emotional Shape System (ess/): emotional_shapes.py, emotional_bonds.py.
Write calibration test: frustration shape emerges after 3 verifier vetoes on the same
hypothesis.

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
20. Implement Social Learning System (sls/): social_shape.py, trace_reconstructor.py,
self_distillation.py. Write trust test: verifier agreement -&gt; trust+0.05; disagreement -&gt;
trust-0.15; floor at 0.05.
21. Implement Unified Shape Language (usl/): lal.py, codec.py, syntax_encoder.py,
syntax_decoder.py, pragmatics.py, isl.py, linguistic_bonds.py. Write round-trip test:
encode then decode recovers original shape at fidelity &gt;= 0.85.
22. Implement checkpoint manager (checkpoints/checkpoint_manager.py): save, load,
validate, replay-verify. Write continuity test: save checkpoint, reload in new process, run
100 inference steps, compare to pre-save replay. Fidelity &gt;= 99.9%.
23. Implement D1 curriculum (dc/stage_d1.py, dc/arc_loader.py, dc/relgraph_generator.py):
ARC-AGI grid loading and encoding; synthetic relational graph generator with masking.
Implement gate_certifier.py with D1 gate checks. Run D1 training. Certify all D1 gates.
24. Implement and certify D2 through D6 curricula sequentially, with gate certification
blocking each transition.
25. Run all test suites (unit, invariant, determinism, sgp, developmental). Validate all
evaluation tracks (five-dimension scorecard, human parity profile). Issue first signed
PSN-2 release manifest.

24. Failure Modes and Mitigations

Failure Mode Description Mitigation
Unbounded growth Spawn rate exceeds
compression rate; substrate
exhausts hardware memory

I-24 hard gate: spawn blocked when ledger
shows net_delta &gt; nodes_freed. Stability
gate rolls back within 50 pulses.

Growth without utility Nodes spawn but produce no
improvement in prediction error

Stability gate: e_j &lt; 0.50 after 50 pulses
required. Failed gate = rollback.

Premature language
training

D4 USL training starts before
D2/D3 causal and social
grounding certified

Gate certifier blocks D4 data loader from
activating until D3 certification recorded.
Enforced at data-loader level.

Role collapse All nodes converge to one role;

MPF landscape flat

L_role_balance + L_attractor_separation +
role-forcing synthetic tasks at D1.

Codebook collapse VSA codebook prototypes
cluster; disambiguation fails

Prototype repulsion penalty; utilisation
monitoring; dead-code rejuvenation.

Temporal motif
premature lock-in

High-similarity motif retrieved for
a task that needs flexible
reasoning

Interference gate &lt;= 0.03. Low success-rate
motifs demoted. Motifs can be suppressed
by frustration emotional shape.

ERS memory pollution Low-utility episodes accumulate;
retrieval quality degrades

Nightly consolidation recomputes utility;
evicts &lt; 0.10. Utility discard at write (-0.50
floor).

Social trace poisoning Observed agent has trust &lt;
0.30; injects bad reasoning
patterns

Low-trust traces: Working tier only. Verifier
checks all reconstructed traces. Trust floor
at 0.05.

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
Kaggle session OOM Node count or D too large for T4

VRAM

Lite config (D=512, N=256) fits in ~2GB.
SCE sparsity keeps active computation far
below VRAM limit on familiar inputs.

Checkpoint corruption
across sessions

Partial save during session
cutoff produces invalid
checkpoint

Auto-save every 30 minutes. Atomic write:
write to temp file, verify hash, rename. Load
validates hash before use.

ISL self-loop Inner speech biases predictions
toward itself, overriding external
evidence

ISL input is modulatory only (not proximal).
Verifier checks that surface output is
grounded in external evidence, not only ISL
shapes.

Curiosity goal
accumulation

Unresolved curiosity goals
accumulate faster than they
resolve

Priority cap at 1.0; oldest unresolved goals
with priority &lt; 0.40 and age &gt; 1000 episodes
are retired. Max 50 pending curiosity goals
at any time.

25. Formal Definition

PSN-2 is a Unified Predictive Substrate Network: a post-transformer architecture in
which one substrate of stateful typed nodes, VSA-encoded bonds, and composed
shapes simultaneously implements structured reasoning (through local prediction-
error minimization, verifier-gated commitment, and verifiable canonical state),
natural language (as a native shape class in the Unified Shape Language,
grounded bidirectionally in the same substrate as non-linguistic reasoning through
the Inner Speech Loop), emotion (as first-class shapes that bond to cognitive
shapes and influence their stability and commitment thresholds through normal
bond mechanics), learning (through continuous local weight updates, inference-time
ERS retrieval, in-context conditioning, and curiosity-driven goal generation), and
controlled growth (through the Substrate Growth Protocol — prediction-error-driven
node spawning, utility-based pruning, and motif compression as the balancing
counterweight — all bounded by hard budget constraints and the growth-rate
invariant I-24). Computation is event-driven and sparse: nodes fire only when their
prediction error exceeds a threshold, making inference energy proportional to task
novelty. Development proceeds in the biologically validated order across six
certified stages: sensorimotor grounding on ARC-AGI grids and synthetic relational
graphs precedes causal reasoning, which precedes social cognition, which
precedes language, which precedes formal abstraction, which precedes meta-
learning integration. The system starts small, grows autonomously under prediction-
error pressure, compresses through temporal motif formation, and accumulates
capability across Kaggle sessions through a checkpoint schema that preserves all
substrate state. The result is a system that is simultaneously good at reasoning,
language, efficient learning, low-energy operation, and graceful self-directed growth
— not by trading these properties against each other, but because they all emerge
from one substrate, one set of operations, and one learning law.

PSN-2: Unified Intelligence Architecture — Authoritative PRD v1.0 CONFIDENTIAL · Ideation Labs · April 2026

Single Authoritative Document — Supersedes All Prior PSN Specifications
PSN-2 · Ideation Labs · Version 1.0 · April 2026 · CONFIDENTIAL — Single Authoritative Document