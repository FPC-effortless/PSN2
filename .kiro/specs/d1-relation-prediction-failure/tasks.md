# Implementation Plan

## Phase 1: Exploratory Testing (Run on UNFIXED Code)

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Relation Prediction Catastrophic Failure
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test that entity prediction accuracy exceeds 50% after 5000 training steps on relational graph batches
  - The test assertions should match the Expected Behavior Properties from design:
    - Entity prediction accuracy > 50% (error < 0.50)
    - Gradient norms for entity_encoder > 0.01 (not vanishing)
    - Bond formation rate > 30% of pulses
  - Run test on UNFIXED code during compositional phase with graph batches
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found:
    - Entity prediction accuracy remains at ~1.5% (random baseline)
    - Gradient norms near zero or highly unstable
    - Bond formation occurs in < 10% of pulses
    - Neighborhood context vectors have low variance
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Grid Task Performance Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (ARC grid batches, perceptive phase)
  - Observe: Object tracking accuracy is 80%+ on grid batches
  - Observe: Grid decoder produces accurate spatial predictions
  - Observe: Other D1 gates (temporal_trace_persistence, vsa_binding_accuracy) pass
  - Write property-based tests capturing observed behavior patterns:
    - For all ARC grid batches, object tracking accuracy >= 80%
    - For all perceptive phase inputs, committed_shape is stable
    - For all grid batches, loss computation is consistent
    - For all DataParallel training, both GPUs are utilized
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

## Phase 2: Implementation

- [x] 3. Fix for D1 relation prediction failure

  - [x] 3.1 Enhance neighborhood context encoding in `encode_graph_neighborhood`
    - Modify `encode_graph_neighborhood` to preserve more relational structure
    - Use attention-weighted aggregation based on relation type instead of simple averaging
    - Add relation type embeddings directly to the neighborhood context
    - Ensure the neighborhood vector carries discriminative signal
    - _Bug_Condition: isBugCondition(input) where input["type"] == "graph" AND phase == "compositional" AND entity_prediction_accuracy < 0.10_
    - _Expected_Behavior: Entity prediction accuracy > 50% after sufficient training (from design Expected Behavior Properties)_
    - _Preservation: Grid task performance (80%+ object tracking) and other D1 gates remain unchanged (from design Preservation Requirements)_
    - _Requirements: 1.3, 2.3, 2.4_

  - [x] 3.2 Improve entity decoder input by using direct graph encoding
    - Replace noisy `active_shape` with direct `encode_graph(entities, relations)` output
    - This provides cleaner gradient path: entity_decoder → encode_graph → entity_encoder
    - Bypass noisy node bank early in training
    - Ensure gradients flow directly to entity/relation encoders
    - _Bug_Condition: isBugCondition(input) where gradients fail to reach entity_encoder_
    - _Expected_Behavior: Gradient norms for entity_encoder > 0.01 (from design Expected Behavior Properties)_
    - _Preservation: Grid encoder/decoder paths remain unchanged (from design Preservation Requirements)_
    - _Requirements: 1.4, 2.4_

  - [x] 3.3 Add relation-aware prediction mechanisms
    - Add relation-conditioned prediction head to entity decoder
    - Explicitly use relation embeddings from masked entity's neighborhood
    - Help model learn "entity X is related to entity Y via relation R"
    - Add attention mechanism over relation types
    - _Bug_Condition: isBugCondition(input) where relation structure is lost in global shape_
    - _Expected_Behavior: Entity prediction leverages relational structure (from design Expected Behavior Properties)_
    - _Preservation: Relation decoder for grid tasks remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.3, 2.3_

  - [x] 3.4 Strengthen bond utilization during prediction
    - Ensure bonds formed in Phase B are used during entity prediction
    - Add bond-based context retrieval: query bonds for masked entity slot
    - Use `BondSystem.recover_source` to retrieve related entities via bond vectors
    - Integrate bond context into entity decoder input
    - _Bug_Condition: isBugCondition(input) where bonds are formed but not utilized_
    - _Expected_Behavior: Bonds improve relational reasoning (from design Expected Behavior Properties)_
    - _Preservation: Bond formation in other phases remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.5, 2.5_

  - [x] 3.5 Adjust loss weighting for compositional phase
    - Increase entity loss weight from 1.0 to 2.0 during compositional phase
    - Reduce shape loss weight from 0.1 to 0.05 to prevent gradient interference
    - Prioritize entity prediction learning early in training
    - Ensure loss balancing doesn't affect grid task loss computation
    - _Bug_Condition: isBugCondition(input) where entity loss is dominated by other terms_
    - _Expected_Behavior: Entity prediction loss drives learning (from design Expected Behavior Properties)_
    - _Preservation: Loss computation for grid tasks remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.4, 2.4_

  - [x] 3.6 Add gradient monitoring and logging
    - Log gradient norms for entity_encoder, relation_encoder, entity_decoder
    - Add logging every 100 steps during training
    - Diagnose gradient flow issues (vanishing or exploding gradients)
    - Add gradient clipping if needed (max_norm=1.0)
    - _Bug_Condition: isBugCondition(input) where gradients are vanishing or unstable_
    - _Expected_Behavior: Gradient norms remain stable (from design Expected Behavior Properties)_
    - _Preservation: Gradient computation for grid tasks remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.4, 2.4_

  - [x] 3.7 Ensure bond formation reliability in `_phase_b`
    - Lower activation threshold or ensure at least 2 nodes active during compositional phase
    - Add logging to track bond formation frequency
    - Verify bonds are formed between active nodes consistently
    - Ensure bond formation doesn't interfere with perceptive phase
    - _Bug_Condition: isBugCondition(input) where bond formation is sparse due to low activation_
    - _Expected_Behavior: Bond formation rate > 30% of pulses (from design Expected Behavior Properties)_
    - _Preservation: Bond formation in other phases remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.5, 2.5_

  - [x] 3.8 Adjust bond decay parameters
    - Reduce bond decay from LAMBDA_BOND = 0.90 to 0.95 (5% decay instead of 10%)
    - Give bonds more persistence to be utilized before decaying
    - Ensure decay adjustment doesn't affect other bond types
    - Verify bonds persist long enough for prediction
    - _Bug_Condition: isBugCondition(input) where bonds decay before being utilized_
    - _Expected_Behavior: Bonds persist across multiple pulses (from design Expected Behavior Properties)_
    - _Preservation: Bond decay in other phases remains unchanged (from design Preservation Requirements)_
    - _Requirements: 1.5, 2.5_

  - [x] 3.9 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Relation Prediction Accuracy Exceeds 50%
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify entity prediction accuracy > 50% after 5000 training steps
    - Verify gradient norms for entity_encoder > 0.01
    - Verify bond formation rate > 30%
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.10 Verify preservation tests still pass
    - **Property 2: Preservation** - Grid Task Performance Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Verify object tracking accuracy >= 80% on grid batches
    - Verify perceptive phase committed_shape is stable
    - Verify loss computation for grid tasks is consistent
    - Verify DataParallel training continues to work
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run full test suite including exploration and preservation tests
  - Verify D1 gate certifier passes with relation_prediction_error < 0.50
  - Run integration test: 5000 steps on mixed batches (60% ARC, 40% graph)
  - Verify checkpoint save/load preserves entity prediction accuracy
  - Ask the user if questions arise or if additional validation is needed
