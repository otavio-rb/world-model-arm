# Integração com o DreamerV3

O world model usado aqui é o [DreamerV3 (NM512/dreamerv3-torch)](https://github.com/NM512/dreamerv3-torch).
Em vez de copiar o código dele para este repositório, a integração é feita por
três arquivos pequenos aplicados sobre um clone do projeto original. O script
`setup_models.py` (na raiz) faz isso automaticamente; esta pasta documenta o que
ele aplica, caso você queira fazer manualmente.

## Arquivos de integração

| Arquivo | Destino | Função |
|---|---|---|
| `envs_ur5e.py` | `dreamerv3-torch/envs/ur5e.py` | env do UR5e para o Dreamer (observação em estado e em pixels) |
| `configs_ur5e.yaml` | anexado ao fim de `dreamerv3-torch/configs.yaml` | preset `ur5e_proprio` (observação em estado, 4 ambientes) |
| patch em `dreamer.py` | — | backend de render no Windows (`glfw`) e um ramo `ur5e` no `make_env` |

### Patch no `dreamer.py`

1. Troca da linha de GL (osmesa é Linux; no Windows usa glfw):

   ```python
   # de:
   os.environ["MUJOCO_GL"] = "osmesa"
   # para:
   os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "osmesa")
   ```

2. Novo ramo em `make_env` (antes do `else: raise NotImplementedError(suite)`):

   ```python
   elif suite == "ur5e":
       import envs.ur5e as ur5e_mod
       cls = ur5e_mod.UR5eState if task == "state" else ur5e_mod.UR5ePixel
       env = cls(size=tuple(config.size), action_repeat=config.action_repeat,
                 seed=config.seed + id)
       env = wrappers.NormalizeActions(env)
   ```

## Requisitos de ambiente

O env do Dreamer importa o env do braço (`envs/ur5e_reach.py` deste repositório).
Exporte o caminho antes de treinar/avaliar:

```bash
export WMA_ENVS=/caminho/para/world-model-arm/envs      # Windows (PowerShell): $env:WMA_ENVS="...\\envs"
```

## Treinar o world model

```bash
cd dreamerv3-torch
python dreamer.py --configs ur5e_proprio --logdir logdir/ur5e_state --steps 150000
```

O checkpoint fica em `logdir/ur5e_state/latest.pt`. Os scripts de avaliação
(`eval/dream_vs_real.py`, `viewers/split_server.py`) o carregam via
`WMA_DREAMER_CKPT` (padrão: `logdir/ur5e_state/ur5e_state_final.pt`).

## Nota sobre AMD / ROCm

Todo o treino deste trabalho rodou em GPU AMD (Radeon RX 9060 XT) via PyTorch
ROCm no Windows. A API `torch.cuda` do PyTorch mapeia para o HIP/ROCm, então o
código usa `device="cuda"` normalmente, sem alterações.
