# Cross-prompt Automated Essay Scoring

Implementação de sistemas de avaliação automática de essays usando modelos Transformer, com foco em generalização cross-prompt baseado no artigo "Improving Domain Generalization for Prompt-Aware Essay Scoring via Disentangled Representation Learning" (ACL 2023).

## Visão Geral

Este projeto implementa modelos baseados em Transformers (BERT, RoBERTa, DistilBERT, ELECTRA) para avaliar automaticamente essays. O diferencial é a avaliação **cross-prompt**: treinar o modelo em um prompt e testá-lo em outro prompt diferente, avaliando a capacidade de generalização.

## Informações do Sistema

- **GPU**: NVIDIA GeForce RTX 5070
- **VRAM**: 12GB
- **CUDA Version**: 13.0
- **Driver Version**: 581.57

### Dataset

**ASAP (Automated Student Assessment Prize)**
- 8 prompts diferentes
- Essays escritos por estudantes
- Scores atribuídos por avaliadores humanos
- Desafio: generalizar entre diferentes prompts e tipos de questões

## Estrutura do Projeto

```
.
├── data/
│   ├── raw/              # Dataset ASAP (link abaixo)
│   └── processed/        # Dados processados
├── src/
│   ├── data/            # Carregamento e pré-processamento
│   │   ├── dataset.py   # ASAP dataset handler
│   │   └── tokenizer.py # Tokenização
│   ├── models/          # Arquiteturas de modelos
│   │   └── transformer_aes.py
│   ├── utils/           # Utilidades
│   │   ├── config.py    # Configurações
│   │   └── logger.py    # Logging
│   └── evaluation/      # Métricas e avaliação
│       └── metrics.py   # QWK, Pearson, Spearman, RMSE
├── scripts/
│   ├── prepare_data.py  # Preparação dos dados
│   ├── train.py         # Script de treinamento
│   └── evaluate.py      # Script de avaliação
├── checkpoints/         # Modelos salvos
└── logs/               # Logs de treinamento
```

## Instalação

### 1. Instalar dependências com uv

```bash
# Instalar pacotes principais
uv pip install -e .

# Instalar dependências de desenvolvimento
uv pip install -e ".[dev]"
```

### 2. Adicionar o dataset ASAP

Coloque o dataset ASAP no diretório:
```
data/raw/asap/
```

### 3. Preparar os dados

```bash
python scripts/prepare_data.py \
    --input_path data/raw/asap \
    --output_path data/processed
```

## Uso

### Treinamento

```bash
python scripts/train.py \
    --model_name bert-base-uncased \
    --source_prompt 1 \
    --target_prompt 2 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --output_dir checkpoints/bert_p1_p2
```

### Avaliação

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/bert_p1_p2/best_model.pt \
    --target_prompt 2 \
    --batch_size 16 \
    --output_file results/bert_p1_p2.json
```

## Modelos Disponíveis

| Modelo | Tamanho | Parâmetros | VRAM (~) |
|--------|---------|------------|----------|
| BERT-base | 110M | 110M | ~4GB |
| RoBERTa-base | 125M | 125M | ~4.5GB |
| DistilBERT | 66M | 66M | ~2.5GB |
| ELECTRA-base | 110M | 110M | ~4GB |

## Métricas de Avaliação

- **QWK (Quadratic Weighted Kappa)**: Métrica principal do ASAP competition
- **Pearson Correlation**: Correlação linear
- **Spearman Correlation**: Correlação de rank
- **RMSE**: Root Mean Squared Error

## Cenários Cross-prompt

O projeto explora diferentes cenários:

1. **Same-prompt**: Treino e teste no mesmo prompt (baseline)
2. **Cross-prompt**: Treino em prompt A, teste em prompt B
3. **Multi-prompt**: Treino em múltiplos prompts, teste em outro
4. **Domain adaptation**: Técnicas de adaptação de domínio

## Configuração para 12 GB VRAM

```python
TrainingConfig(
    batch_size=8,
    gradient_accumulation_steps=4,  
    fp16=True,                       
    max_length=512
)
```
## Referências
- [ASAP Dataset](https://www.kaggle.com/c/asap-aes)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- Artigo base: "Improving Domain Generalization for Prompt-Aware Essay Scoring via Disentangled Representation Learning" (ACL 2023)
