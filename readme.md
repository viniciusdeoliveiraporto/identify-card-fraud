# 💳 Inteligência Artificial para Detecção de Fraudes em Cartões de Crédito

Este desafio, proposto pelo Kaggle, disponibiliza um dataset com 284.807 transações, sendo 492 fraudes (0,172%), tornando o problema extremamente desbalanceado, assim como na vida real.

🧠 Para resolver, desenvolvemos uma Rede Neural Autoencoder, utilizando Python e TensorFlow. Ela foi treinado apenas com transações legítimas, seguindo a arquitetura 32-14-7-7-32, inspirada no artigo do Dr. Mohammed Abdulhameed Al-Shabi.

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
