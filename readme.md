# Identificação de Fraudes em Cartões de Crédito

Este projeto é um **protótipo inicial** de um sistema de detecção de anomalias/fraudes em transações de cartão de crédito. A ideia é utilizar **redes neurais autoencoders** para identificar comportamentos fora do padrão em transações financeiras.

---
## Execução

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd identify-card-fraud
```

### 2. Instalar dependências
Certifique de estar fora da pasta src e instale as dependências listadas em requirements.txt

```bash
pip install -r requirements.txt
```

### 3. Execute o sistema
A execução deve ser feita fora da pasta src/ para evitar erros de importação (ModuleNotFoundError: No module named 'src').

```bash
python -m src.main
```

### 4. Execute os testes
Para executar os testes, tambem é necessario estar fora da pasta src para evitar os mesmos erros de importação (ModuleNotFoundError: No module named 'src').

```bash
python -m pytest -v src/tests/test_fraude.py
```

---
## Observações
- O projeto ainda está em fase inicial (protótipo).
- Futuramente serão adicionadas melhorias nos autocodificadores.