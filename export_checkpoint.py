"""Export FSDP checkpoint to HuggingFace format."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_fsdp_checkpoint(checkpoint_dir):
    """Load and merge FSDP sharded checkpoint."""
    # Find all model shard files
    shard_files = sorted([
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith('model_world_size_') and f.endswith('.pt')
    ])
    
    print(f"Found {len(shard_files)} shard files: {shard_files}")
    
    # Load with DTensor support
    import torch.distributed.tensor as dtensor
    
    # Load each shard
    full_state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_file)
        print(f"Loading {shard_path}...")
        shard_dict = torch.load(shard_path, map_location='cpu', weights_only=False)
        
        # Merge into full state dict
        for key, value in shard_dict.items():
            if key not in full_state_dict:
                if hasattr(value, 'full_tensor'):
                    # DTensor - get full tensor
                    full_state_dict[key] = value.full_tensor()
                else:
                    full_state_dict[key] = value
            else:
                # Need to concatenate shards
                existing = full_state_dict[key]
                if hasattr(value, 'full_tensor'):
                    value = value.full_tensor()
                # Concatenate along the sharded dimension
                # For FSDP, this is typically dim 0
                full_state_dict[key] = torch.cat([existing, value], dim=0)
    
    return full_state_dict

def export_to_hf(checkpoint_dir, output_dir, base_model_name="Qwen/Qwen2.5-3B"):
    """Export checkpoint to HuggingFace format."""
    print(f"Loading base model from {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    print(f"Loading checkpoint from {checkpoint_dir}...")
    state_dict = load_fsdp_checkpoint(checkpoint_dir)
    
    # Remove FSDP prefixes if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common FSDP prefixes
        new_key = key
        for prefix in ['_fsdp_wrapped_module.', 'module.', '_orig_mod.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned_state_dict[new_key] = value
    
    print(f"Loading state dict into model...")
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}...")
    
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    checkpoint_dir = "/workspace/checkpoints/global_step_100/actor"
    output_dir = "/workspace/checkpoints/global_step_100/actor/huggingface"
    export_to_hf(checkpoint_dir, output_dir)
