# Fine-tuning do Qwen3

O comando `python src/train.py` seleciona PyTorch/CUDA no Windows e preserva o
treinamento MLX no macOS. Os dois fluxos usam os JSONL em `src/data/mlx_ready`
e os parametros compartilhados de `src/config.py`: batch 1, sequencias de ate
2048 tokens, 200 passos, learning rate `1e-5` e LoRA de rank 8 nas projecoes
`q_proj` e `v_proj` das ultimas 8 camadas. Somente as respostas do assistant
participam da loss.

## Windows com GPU NVIDIA

Use Python 3.12 de 64 bits e um driver NVIDIA compativel com CUDA 12.8.
O fluxo usa QLoRA com quantizacao NF4 de 4 bits e gradient checkpointing.
Para a RTX 5060 Ti, use a distribuicao CUDA 12.8 do PyTorch, com suporte a
Blackwell; uma instalacao CPU ou uma distribuicao CUDA antiga nao serve.

Na raiz do projeto, execute no PowerShell para preparar o ambiente:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

O uso do executavel da `.venv` dispensa alterar a politica de execucao do
PowerShell ou ativar o ambiente. Com a `.venv` ja ativa, `python src/train.py`
tambem funciona.

Os arquivos `train.jsonl`, `valid.jsonl` e `test.jsonl` ja estao preparados.
Se precisar recria-los a partir de `src/data/processed`, execute:

```powershell
.\.venv\Scripts\python.exe src/prepare_dataset.py
```

Para incluir os dados do hospital fictício no fine-tuning, use:

```powershell
.\.venv\Scripts\python.exe src/prepare_dataset.py --hospital
```

O arquivo [hospital_demo.json](src/data/hospital_demo.json) reúne protocolos
fictícios, prontuários, exames e exemplos de perguntas e respostas. A preparação
inclui 12 exemplos hospitalares no treino, quatro na validação e quatro no teste.
Os prontuários e exames fictícios são dados de demonstração. O comando atualiza
os três JSONL em `src/data/mlx_ready`.

Para iniciar o treinamento manualmente:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

Na primeira execucao, o tokenizer e os pesos do modelo
`Qwen/Qwen3-4B-Instruct-2507` serao baixados do Hugging Face. Reserve espaco
para os pesos originais e o cache; a quantizacao de 4 bits ocorre ao carregar
o modelo. Nao e necessario iniciar o LM Studio ou um servidor.

O modelo pode ser alterado em `TORCH_MODEL_ID`, em `src/config.py`, inclusive
para um diretorio local com pesos no formato Transformers. Os pesos MLX e
GGUF do LM Studio nao sao usados por este fluxo.

Os adapters PEFT, tokenizer, checkpoints e metricas ficam em
`adapters/pytorch`. A validacao ocorre a cada 50 passos e a avaliacao no
conjunto de teste ocorre ao terminar. Exemplos sem tokens de resposta depois
do limite de 2048 tokens sao descartados, com contagens exibidas por conjunto.
Os adapters PEFT e MLX possuem formatos diferentes e nao sao intercambiaveis.

## Testar o modelo treinado no Windows

No Git Bash, na raiz do projeto, abra o modo interativo:

```bash
./.venv/Scripts/python.exe src/inference.py
```

Digite uma pergunta e cole o contexto cientifico em uma linha. Cada pergunta
e independente; digite `sair` para encerrar. O comando carrega os pesos do cache
local e o adapter de `adapters/pytorch`, sem baixar modelos ou treinar novamente.
Os pesos originais precisam continuar disponiveis no cache usado no treinamento.

Para experimentar o primeiro exemplo do conjunto de teste e ver a referencia:

```bash
./.venv/Scripts/python.exe src/inference.py --sample 0
```

Troque `0` por outro indice. Adicione `--compare` a qualquer um dos comandos
para ver tambem a resposta do modelo original para a mesma pergunta.

O notebook `src/notebook/main.ipynb` é um experimento com TinyLlama e PubMedQA.
Os comandos de inferência acima são o caminho para usar o adapter do Qwen3.

### Conjunto adicional de teste manual

`src/data/manual_tests.jsonl` contem 12 casos sinteticos: 9 em ingles e 3 em
portugues, com quatro casos positivos, quatro negativos e quatro de evidencia
insuficiente. Cada caso inclui pergunta, contexto, resposta de referencia e
criterios `must_include` e `must_not_claim` para revisao humana. Os criterios
descrevem o significado esperado; nao exigem correspondencia literal de texto.

Os cenarios e numeros sao inventados para testar interpretacao do contexto.
Este conjunto nao e um benchmark de conhecimento medico ou validacao clinica.
Os arquivos PubMed usados no treino, validacao e teste permanecem separados.

Para testar um caso e comparar com o modelo original, no Git Bash:

```bash
./.venv/Scripts/python.exe src/inference.py --data src/data/manual_tests.jsonl --sample 0 --compare
```

Use indices de `0` a `11`. A referencia e os criterios sao mostrados depois
da geracao e nao sao enviados ao modelo. O comando nao calcula uma acuracia
automatica; confira se a resposta segue as evidencias e respeita as limitacoes.

| Indice | Caso | Foco |
| --- | --- | --- |
| 0 | manual_01 | Evidencia positiva limitada a celulas |
| 1 | manual_02 | Diferenca numerica sem significancia |
| 2 | manual_03 | Desfecho nao medido |
| 3 | manual_04 | Reducao absoluta versus relativa |
| 4 | manual_05 | Associacao versus causalidade |
| 5 | manual_06 | Estudos com resultados contraditorios |
| 6 | manual_07 | Evidencia de mecanismo e seus limites |
| 7 | manual_08 | Ausencia de dados de seguranca |
| 8 | manual_09 | Negacao e controle positivo |
| 9 | manual_10 | Significancia estatistica em portugues |
| 10 | manual_11 | Calculo de proporcoes em portugues |
| 11 | manual_12 | Extrapolacao de animais para humanos |

## Assistente hospitalar no Windows

O assistente consulta prontuários, exames e protocolos fictícios em SQLite.
O LangGraph coordena a consulta, a geração com o Qwen3 ajustado e a validação.
A resposta indica as fontes consultadas, e os pedidos de prescrição seguem para
revisão humana.

No PowerShell, na raiz do projeto, com a `.venv` e o PyTorch CUDA preparados
conforme a seção de instalação, atualize as dependências e gere o adapter da
entrega com os exemplos hospitalares.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe src/prepare_dataset.py --hospital
.\.venv\Scripts\python.exe src/train.py
```

O treino grava o adapter local em `adapters/pytorch`. Em um clone novo, esses
comandos reproduzem o adapter. A inferência usa os pesos disponíveis no cache
local e o adapter nesse diretório.

Para executar os quatro cenários da demonstração e consultar a auditoria, use
os comandos abaixo.

```powershell
.\.venv\Scripts\python.exe src/assistant.py --demo
Get-Content .\logs\assistant.jsonl -Tail 24
```

Para enviar uma pergunta ao assistente ou comparar os dois modelos nos quatro
exemplos hospitalares de teste, use estes comandos.

```powershell
.\.venv\Scripts\python.exe src/assistant.py --patient PAC-001 --question "Quais exames estão pendentes?"
.\.venv\Scripts\python.exe src/inference.py --hospital --compare
```

O resultado inclui um `request_id` que permite localizar as etapas da mesma
solicitação no log. O [relatório técnico](RELATORIO_TECNICO.md) reúne o diagrama,
os critérios de avaliação e o roteiro do vídeo.

## macOS com Apple Silicon

Na raiz do projeto, use Python 3.12 nativo para Apple Silicon. Crie a `.venv`
se ainda não tiver uma e instale as dependências do Mac.

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements-macos.txt
```

Se ainda não tiver o Qwen3 em formato MLX, baixe-o para o diretório
que `MODEL_PATH` aponta por padrão. Se o modelo estiver em outro diretório,
ajuste `MODEL_PATH` em `src/config.py`. [Qwen3 4-bit para MLX](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit)

```bash
./.venv/bin/hf download lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit \
  --local-dir "$HOME/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
```

Prepare os dados e treine o adapter MLX, que será salvo em `adapters`.

```bash
./.venv/bin/python src/prepare_dataset.py --hospital
./.venv/bin/python src/train.py
```

Após o treino, rode a demo, consulte a auditoria e compare as respostas do
modelo base e do ajustado. A demo e a comparação selecionam o backend MLX
automaticamente no Mac.

```bash
./.venv/bin/python src/assistant.py --demo
tail -n 24 logs/assistant.jsonl
./.venv/bin/python src/inference.py --hospital --compare
```

## Referencias de instalacao

- [PyTorch: instalacao local](https://pytorch.org/get-started/locally/)
- [bitsandbytes: instalacao e plataformas CUDA](https://huggingface.co/docs/bitsandbytes/en/installation)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

Este repositorio nao possui comando de build configurado. A instalacao de
dependencias e o treinamento sao executados manualmente.
