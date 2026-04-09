import torch
from psn2.phases import PhaseController
from psn2.node import NodeBank
from psn2.vsa import normalize
import pytest

def test_phase_controller_init():
    node_bank = NodeBank(num_nodes=10, dim=512)
    controller = PhaseController(node_bank, budget=10)
    assert controller.current_phase == "A"
    assert controller.budget == 10
    
def test_phase_cycle():
    node_bank = NodeBank(num_nodes=10, dim=512)
    controller = PhaseController(node_bank, budget=10)
    
    # Needs to transition A -> B -> C -> D -> E -> F -> A
    assert controller.current_phase == "A"
    controller.step_phase()
    assert controller.current_phase == "B"
    controller.step_phase()
    assert controller.current_phase == "C"
    controller.step_phase()
    assert controller.current_phase == "D"
    controller.step_phase()
    assert controller.current_phase == "E"
    controller.step_phase()
    assert controller.current_phase == "F"
    controller.step_phase()
    assert controller.current_phase == "A"
    
def test_run_pulse():
    node_bank = NodeBank(num_nodes=10, dim=512)
    controller = PhaseController(node_bank, budget=10)
    
    # Provide a dummy input shape
    inp = normalize(torch.randn(512))
    
    # Run one full pulse
    controller.run_pulse(external_input=inp)
    
    # Budget should decrease
    assert controller.budget == 9
    assert controller.current_phase == "A" # reset to A

def test_commitment():
    node_bank = NodeBank(num_nodes=10, dim=512)
    # mock a high stability for some nodes to force commitment
    controller = PhaseController(node_bank, budget=10)
    
    # Set stability high manually
    node_bank.tau[0] = 0.95
    node_bank.tau[1] = 0.90
    
    # Step through phases A, B, C to get to D
    for _ in range(3):
        controller.step_phase()
    
    assert controller.current_phase == "D"
    controller.step_phase() # to E
    controller.step_phase() # to F
    
    # F handles commitment
    # For now, let's just make sure it doesn't crash 
    controller.execute_phase("F")
    assert controller.committed_shape is not None
