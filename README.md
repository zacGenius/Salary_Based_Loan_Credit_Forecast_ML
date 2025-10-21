# 💰 Salary Based Loan Credit Forecast (ML)

Projeto de **Machine Learning** desenvolvido por **Isaac Camilo (@zacgenius)** para prever o **limite de crédito de empréstimo** com base no **salário informado pelo cliente**, utilizando um modelo de **Regressão Linear Simples**.

---

## 🧠 Objetivo

Criar um modelo preditivo capaz de estimar o **limite de empréstimo** que pode ser oferecido a um cliente, considerando apenas o valor do seu **salário mensal**.  
Esse modelo é um exemplo introdutório de aplicação de **Machine Learning supervisionado (regressão)** para problemas financeiros.

---

## ⚙️ Tecnologias Utilizadas

- *Pandas* → leitura e manipulação de dados  
- *NumPy* → cálculos matemáticos e correlação  
- *Matplotlib / Seaborn* → visualização de dados  
- *Scikit-Learn (sklearn)* → criação e avaliação do modelo de regressão linear  
- *OpenPyXL* → leitura de arquivos Excel (`.xlsx`)

---

## 📊 Estrutura do Projeto

### 1. Leitura e exploração dos dados
```python
df = pd.read_excel('BaseDados_RegressaoLinear.xlsx', 'Plan1')
df.info()
df.describe()
````

### 2. Visualização e análise

* Gráfico de dispersão: relação entre salário e limite
* Heatmap para verificar valores nulos
* Correlação entre as variáveis

### 3. Preparação dos dados

```python
eixo_x = df.iloc[:, 0].values.reshape(-1, 1)  # Salário
eixo_y = df.iloc[:, 1].values.reshape(-1, 1)  # Limite
```

### 4. Divisão entre treino e teste

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    eixo_x, eixo_y, test_size=0.20
)
```

### 5. Criação e treino do modelo

```python
from sklearn.linear_model import LinearRegression

funcao_regressao = LinearRegression()
funcao_regressao.fit(x_train, y_train)
```

### 6. Visualização dos resultados

```python
plt.figure(figsize=(10, 5))
sns.scatterplot(x=x_train.ravel(), y=y_train.ravel())
plt.plot(x_test, funcao_regressao.predict(x_test), color='red')
plt.title('Previsão de Limite de Empréstimo com Base no Salário')
plt.xlabel('Salário')
plt.ylabel('Limite')
plt.show()
```

### 7. Avaliação do modelo

```python
from sklearn import metrics

previsoes = funcao_regressao.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, previsoes))
print('RMSE:', rmse)
```

### 8. Teste prático de previsão

```python
# Exemplo: prever o limite para um salário de R$ 1.600
print(funcao_regressao.predict([[1600]]))
# Saída aproximada: 3713.28
```

---

## 🚀 Como Executar o Projeto

1. **Clone o repositório**

   ```bash
   git clone https://github.com/zacgenius/Salary_Based_Loan_Credit_Forecast_ML.git
   cd Salary_Based_Loan_Credit_Forecast_ML
   ```

2. **Crie e ative um ambiente virtual (opcional)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o notebook**

   ```bash
   jupyter notebook emprestimo_predict.ipynb
   ```

---

## 🧾 Estrutura de Arquivos

```
Salary_Based_Loan_Credit_Forecast_ML/
│
├── emprestimo_predict.ipynb       # Notebook principal do projeto
├── BaseDados_RegressaoLinear.xlsx # Base de dados de entrada
├── README.md                      # Documentação do projeto
└── requirements.txt               # Dependências do ambiente
```

---

## 📈 Resultados

* O modelo utiliza *regressão linear simples* para encontrar a melhor relação entre salário e limite de crédito.
* O *RMSE (Root Mean Square Error)* é usado como métrica de erro para avaliar o desempenho.
* Permite prever novos valores de limite de empréstimo com base em qualquer salário informado.

---

## 👤 Autor

**Isaac Camilo**
📍 GitHub: [@zacgenius](https://github.com/zacgenius)
