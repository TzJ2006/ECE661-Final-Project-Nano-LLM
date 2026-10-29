"""Every local model behind one generate() call: TinyStories checkpoints (PyTorch) and ours (JAX)."""
import os

TS = ["1M", "3M", "8M", "28M", "33M", "1Layer-21M", "2Layers-33M"]  # roneneldan/TinyStories-<name>
OURS = ["baseline", "looped", "mdlm", "looped_mdlm"]


class HF:
    def __init__(self, name, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or "cpu"  # runs on the TPU VM's CPU; 1-33M models are fast enough
        self.tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        self.lm = AutoModelForCausalLM.from_pretrained(f"roneneldan/TinyStories-{name}").to(self.device).eval()

    def generate(self, prompt, max_new=200, temperature=0.0, steps=None, seed=0):
        self.torch.manual_seed(seed)
        x = self.tok(prompt, return_tensors="pt").to(self.device)
        sample = dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False)
        with self.torch.no_grad():
            y = self.lm.generate(**x, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id, **sample)
        return self.tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)


def load(name, ckpt_root=None):
    if name in TS:
        return HF(name)
    from tinylm.sample import Model
    return Model(os.path.join(ckpt_root, name))


def available(ckpt_root=None):
    """TinyStories checkpoints, plus whichever of our models have finished training."""
    ours = [m for m in OURS if ckpt_root and os.path.exists(os.path.join(ckpt_root, m, "params.msgpack"))]
    return TS + ours
