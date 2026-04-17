# Bugfix Requirements Document

## Introduction

The PSN-2 training system fails the D1 stage gate check due to extremely low relation prediction accuracy (0.0150 or 1.5%). The `causal_prediction_error` gate requires error < 0.50 (equivalent to accuracy > 0.50 or 50%), but the system achieves only 1.5% accuracy on relational graph tasks. This blocks progression to D2 stage despite other gates passing (object tracking: 80.95%, temporal trace persistence: 26.8, VSA binding accuracy: passing).

The D1 stage uses a 60/40 data mix of ARC-AGI-2 grids and synthetic relational graphs. While grid-based tasks perform well, the compositional phase responsible for relational reasoning is failing catastrophically. This suggests a fundamental issue in how the system encodes, processes, or decodes relational graph structures during the compositional phase.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the system processes relational graph batches during D1 training THEN the relation prediction accuracy remains at 0.0150 (1.5%) after 20,000 training steps

1.2 WHEN the D1 gate certifier evaluates `causal_prediction_error` THEN it fails because 1.0 - 0.0150 = 0.9850 error is not < 0.50 threshold

1.3 WHEN the compositional phase processes graph batches with masked entities THEN the entity decoder produces predictions that are effectively random (1/64 ≈ 1.56% baseline)

1.4 WHEN gradients flow through the entity prediction path THEN they fail to meaningfully update the entity encoder, relation encoder, or entity decoder weights

1.5 WHEN the system forms bonds between active nodes during compositional phase THEN the bond formation may not be occurring or the bonds are not being utilized for prediction

### Expected Behavior (Correct)

2.1 WHEN the system processes relational graph batches during D1 training THEN the relation prediction accuracy SHALL reach at least 50% (error < 0.50) by 20,000 steps

2.2 WHEN the D1 gate certifier evaluates `causal_prediction_error` THEN it SHALL pass because the error is below 0.50 threshold

2.3 WHEN the compositional phase processes graph batches with masked entities THEN the entity decoder SHALL produce predictions significantly better than random baseline (> 10% accuracy minimum)

2.4 WHEN gradients flow through the entity prediction path THEN they SHALL meaningfully update the entity encoder, relation encoder, and entity decoder to improve prediction accuracy over time

2.5 WHEN the system forms bonds between active nodes during compositional phase THEN the bonds SHALL be formed correctly and utilized to improve relational reasoning

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the system processes ARC grid batches during D1 training THEN the object tracking accuracy SHALL CONTINUE TO achieve 80%+ performance

3.2 WHEN the perceptive phase processes grid-based tasks THEN the grid decoder SHALL CONTINUE TO produce accurate spatial predictions

3.3 WHEN the D1 gate certifier evaluates other gates (object_tracking_accuracy, temporal_trace_persistence, vsa_binding_accuracy) THEN they SHALL CONTINUE TO pass

3.4 WHEN the system trains on mixed batches (60% ARC, 40% graph) THEN the ARC performance SHALL CONTINUE TO remain stable and not degrade

3.5 WHEN the loss computation includes multiple components (L_error, L_shape, L_vsa, L_compact) THEN the total loss SHALL CONTINUE TO decrease during training

3.6 WHEN the system uses DataParallel across 2 GPUs THEN the training SHALL CONTINUE TO run without errors and utilize both GPUs effectively
