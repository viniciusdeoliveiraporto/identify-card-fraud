# 💳 Detecção de Fraudes em Cartões de Crédito

Este desafio, proposto pelo [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data), disponibiliza um dataset com 284.807 transações, sendo 492 fraudes (0,172%), tornando o problema extremamente desbalanceado, assim como na vida real.

## Inteligência Artificial

🧠 Para resolver, desenvolvemos uma Rede Neural densa não supervisionada, mais especificamente um Autoencoder assimétrico, utilizando Python e TensorFlow. Ele foi treinado apenas com transações legítimas, seguindo a arquitetura 32-14-7-7-32.

![autoencoder](https://github.com/user-attachments/assets/75102ed5-1fe5-4514-9813-574e16e673ec)

🎯 Consideramos prioridade obter um bom recall, ou seja, reduzir falsos negativos e, assim, evitar considerar fraudes como não fraudes. Nosso resultado: 88%, um ótimo percentual de revocação.

![metrica](https://github.com/user-attachments/assets/0e8b3dd2-47ca-4480-b926-421e4e2f1e1d)

Além disso, nosso código obteve nota 9,08/10 na análise estática do PyLint e 100% de cobertura em testes.

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
OU
```bash
python -m src.tests.test_fraude
```

---

## Gráfico de evolução no treinamento por época
![grafico](https://github.com/user-attachments/assets/f6b04cb9-4319-47ac-9a57-ac6b22d3d9d5)
