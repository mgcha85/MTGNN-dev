from safetensors.torch import save_file, load_file

def save_ckpt(model, path: str):
    save_file(model.state_dict(), path)

def load_ckpt(model, path: str, device_str: str):
    state = load_file(path, device=device_str)     # "cuda:0" or "cpu"
    model.load_state_dict(state, strict=True)
