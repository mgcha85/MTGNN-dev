from dataclasses import dataclass

@dataclass
class DataCfg:
    path: str = "./data/sm_data.txt"
    seq_in: int = 10
    seq_out: int = 36
    normalize: int = 2
    split_train: float = 0.43
    split_val: float = 0.30
    num_nodes: int = 142

@dataclass
class ModelCfg:
    gcn_true: bool = True
    buildA_true: bool = True
    gcn_depth: int = 2
    dilation_exponential: int = 2
    node_dim: int = 40
    subgraph_size: int = 20
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    layers: int = 5
    in_dim: int = 1
    out_dim: int = 36
    propalpha: float = 0.05
    tanhalpha: float = 3
    layer_norm_affline: bool = False

@dataclass
class TrainCfg:
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    clip: float = 10.0
    epochs: int = 200
    num_split: int = 1
    step_size: int = 100
    L1Loss: bool = True
    optim: str = "adam"

@dataclass
class SaveCfg:
    ckpt_path: str = "model/Bayesian/model.safetensors"
    hp_path: str = "model/Bayesian/hp.txt"

@dataclass
class Cfg:
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    save: SaveCfg = SaveCfg()
    device: str = "auto"  # "auto"|"cpu"|"cuda:0"
