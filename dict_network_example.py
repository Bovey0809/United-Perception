import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from typing import Dict

class DictNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()

    def forward(self, x: Dict[str, torch.Tensor]):
        # Extract features from dict
        feature1 = x['feature1']  # Assuming shape (batch_size, 10)
        
        # Process through network
        x1 = self.linear1(feature1)
        x2 = self.relu(x1)
        out = self.linear2(x2)
        
        return out

def main():
    # Create model
    model = DictNetwork()
    model.eval()  # Set to eval mode for tracing
    
    # Create dummy input
    batch_size = 2
    dummy_input = {
        'feature1': torch.randn(batch_size, 10)
    }
    
    # Method 1: Direct tracing (this will fail)
    try:
        traced_model = symbolic_trace(model)
        print("Direct tracing succeeded (unexpected)")
    except Exception as e:
        print("Direct tracing failed (expected):")
        print(f"Error: {str(e)}\n")
    
    # Method 2: Create a wrapper class for tracing
    class TracingWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
            
        def forward(self, feature1):
            # Convert single input to dict format
            x = {'feature1': feature1}
            return self.model(x)
    
    # Create and trace the wrapped model
    # wrapped_model = TracingWrapper(model)
    
    # Trace with the appropriate input
    traced_wrapped_model = symbolic_trace(model)
    
    # Test the traced model
    dummy_feature1 = torch.randn(batch_size, 10)
    
    # Original model output
    original_output = model({'feature1': dummy_feature1})
    
    # Traced model output
    traced_output = traced_wrapped_model(dummy_feature1)
    
    # Verify outputs match
    print("Output shapes match:", original_output.shape == traced_output.shape)
    print("Outputs are close:", torch.allclose(original_output, traced_output))
    
    # Print the traced graph
    print("\nTraced Graph:")
    print(traced_wrapped_model.graph)
    traced_wrapped_model.print_readable()
    
    # Export the graph (optional)
    traced_wrapped_model.graph.print_tabular()
    
if __name__ == "__main__":
    main() 