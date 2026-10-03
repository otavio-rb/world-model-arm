# world-model-arm

> **Trabalho em andamento.** Este repositório faz parte de um TCC em
> desenvolvimento. O código, os resultados e a documentação ainda estão
> evoluindo e podem mudar — algumas tarefas (como a preensão) seguem em aberto.

Comparação entre aprendizado por reforço **model-free** (SAC) e **baseado em
modelo de mundo** (DreamerV3) no controle de braços robóticos simulados, com
foco em **eficiência amostral** — quantas interações reais com o ambiente cada
método precisa para aprender uma tarefa.

O argumento central: um modelo de mundo aprende a dinâmica do robô e treina a
política **na imaginação**, extraindo muito mais aprendizado de cada interação
real. Isso importa porque, em um robô físico, cada tentativa custa tempo,
desgaste e risco — a interação, não a computação, é o recurso caro.

Robôs: **UR5e** (6 DOF) para a tarefa de alcance e **Franka Emika Panda**
(7 DOF + garra) para a tarefa de preensão, ambos do
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
World model: [DreamerV3 (NM512/dreamerv3-torch)](https://github.com/NM512/dreamerv3-torch).
Baseline: SAC do [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

<p align="center">
  <img src="figures/ur5e_scene.png" width="300"/>
  <img src="figures/panda_scene.png" width="300"/>
</p>

---

## Resultado principal: eficiência amostral

Na tarefa de alcance com o UR5e, os dois métodos atingem o mesmo desempenho
final, mas o world model chega lá com **cerca de 2,7× menos interação real** com
o ambiente.

![Eficiência amostral](figures/sample_efficiency_ur5e.png)

| Método | Para atingir retorno 100 | Desempenho final (platô) |
|---|---|---|
| SAC (model-free) | ~152 mil passos | ~120–185 |
| DreamerV3 (world model) | ~57 mil passos | ~120–185 |

**Por quê.** O SAC só aprende com as transições que realmente experimentou. O
Dreamer treina a política dentro de rollouts **imaginados** pelo modelo de mundo
(sem tocar no simulador), com um sinal de supervisão denso (reconstruir a
observação, prever recompensa e continuidade) e gradientes analíticos através da
dinâmica aprendida. Cada passo real rende ordens de magnitude mais sinal de
aprendizado.

---

## Ver o modelo de mundo funcionando

Alcançar um ponto não demonstra o modelo de mundo por si só — o que o distingue
é a capacidade de **simular o futuro**. O experimento abaixo semeia o modelo com
5 frames reais e, a partir daí, o deixa **imaginar livre** (malha aberta, sem o
simulador), usando as mesmas ações. À esquerda, o mundo real (MuJoCo); à direita,
o que o modelo imagina.

![Sonho vs realidade](figures/dream_vs_reality.gif)

A curva de erro mostra o comportamento esperado de um bom modelo de mundo:
previsão exata enquanto ancorado nos dados reais (zona em azul), erro **limitado
e não divergente** (~10 cm) em malha aberta longa, e degradação além do horizonte
de treino (~15 passos) — o que justifica por que o Dreamer imagina em horizontes
curtos e re-ancora na realidade.

![Erro de previsão](figures/prediction_error.png)

Políticas aprendidas executando a tarefa de alcance:

<p align="center">
  <img src="figures/sac_ur5e_reach.gif" width="330"/>
  <img src="figures/dreamer_ur5e_reach.gif" width="330"/>
</p>

*(esquerda: SAC; direita: política treinada pelo world model)*

---

## Tarefa difícil: preensão (grasping)

A tarefa de **pegar um objeto** com o Franka Panda expõe o limite do RL puro. É
um problema de exploração difícil e recompensa quase esparsa: para receber a
recompensa de levantar, é preciso alinhar, fechar a garra e erguer numa sequência
precisa que a exploração aleatória quase nunca descobre.

- **SAC do zero:** 600 mil passos, **0 sucessos**. A política estaciona em "se
  aproximar do objeto" e nunca descobre a preensão.
- **Com currículo + ajuste de recompensa:** continua sem convergir (sucessivos
  ótimos locais).

A física da tarefa é válida — um controlador por **cinemática inversa (DLS)**
pega e levanta o cubo de forma confiável:

![Preensão por IK](figures/scripted_grasp_strip.png)

Conclusão (alinhada à literatura): preensão do zero requer mais do que ajuste de
recompensa — **demonstrações** (aprendizado por imitação para semear a política),
HER, ou currículos mais fortes. O controlador por IK deste repositório serve
justamente como **demonstrador** para esse próximo passo.

---

## Estrutura do repositório

```
envs/                 ambientes Gymnasium (MuJoCo)
  ur5e_reach.py         UR5e: alcançar um alvo (obs 21-d, ação 6-d)
  panda_grasp.py        Franka: pegar um cubo (obs 24-d, ação 8-d, currículo)
  assets/*.xml          cenas (incluem os modelos do Menagerie)
dreamer/              integração com o DreamerV3 (ver dreamer/README.md)
  envs_ur5e.py          env do UR5e para o Dreamer (estado e pixels)
  configs_ur5e.yaml     preset ur5e_proprio
train/                treino model-free (SAC)
  train_sac_ur5e.py     baseline no alcance, loga retorno x passos reais
  train_sac_grasp.py    SAC na preensão (loga taxa de sucesso)
eval/                 avaliação e figuras
  plot_sample_efficiency.py   gráfico SAC vs Dreamer
  dream_vs_real.py            sonho vs realidade (malha aberta) + curva de erro
  rollout_sac.py              render da política SAC
  scripted_grasp.py           preensão por cinemática inversa (demonstrador)
viewers/              visualizador web ao vivo (real vs imaginação)
figures/              figuras usadas neste README
results/              curvas de treino (para reproduzir o gráfico)
setup_models.py       baixa Menagerie + dreamerv3-torch e aplica a integração
```

---

## Como reproduzir

Pré-requisitos: Python 3.10+, `git`, e PyTorch instalado para a sua plataforma
(CPU, CUDA ou ROCm — ver `requirements.txt`).

```bash
pip install -r requirements.txt
python setup_models.py          # clona os modelos e o DreamerV3, aplica a integração
```

**Baseline model-free (SAC) no alcance:**

```bash
python train/train_sac_ur5e.py          # salva results/sac_ur5e.jsonl e results/sac_ur5e.zip
```

**World model (DreamerV3) no alcance:**

```bash
# Windows PowerShell: $env:WMA_ENVS="$PWD\envs"   |   Linux/Mac: export WMA_ENVS=$PWD/envs
cd dreamerv3-torch
python dreamer.py --configs ur5e_proprio --logdir logdir/ur5e_state --steps 150000
```

**Gerar as figuras:**

```bash
python eval/plot_sample_efficiency.py    # figures/sample_efficiency_ur5e.png
python eval/rollout_sac.py               # figures/sac_ur5e_reach.gif
python eval/dream_vs_real.py             # figures/dream_vs_reality.gif + prediction_error.png
python eval/scripted_grasp.py            # figures/scripted_grasp.gif
```

**Visualizador ao vivo (real vs imaginação):**

```bash
cd viewers
uvicorn split_server:app --port 8095     # abra http://localhost:8095
```

---

## Ambiente e hardware

Todo o treino foi feito **localmente** em GPU AMD (Radeon RX 9060 XT) usando
PyTorch ROCm no Windows. A API `torch.cuda` mapeia para o HIP/ROCm, então o
código usa `device="cuda"` sem alterações. Não há dependência de nuvem.

---

## Referências

- Hafner et al. — *Mastering Diverse Domains through World Models* (DreamerV3), 2023.
- Wu et al. — *DayDreamer: World Models for Physical Robot Learning*, 2022.
- Haarnoja et al. — *Soft Actor-Critic* (SAC), 2018.
- MuJoCo Menagerie (modelos UR5e e Franka Panda), DeepMind.
- Implementação do DreamerV3: NM512/dreamerv3-torch.

## Licença

MIT. Os modelos de robô vêm do MuJoCo Menagerie e o DreamerV3 do repositório
NM512/dreamerv3-torch, cada um sob sua própria licença (não redistribuídos aqui;
são baixados pelo `setup_models.py`).
