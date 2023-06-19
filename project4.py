import psycopg2
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# def database_connect()
conn = psycopg2.connect(
    host = "rajje.db.elephantsql.com",
    database = "mbslvdmz",
    user = "mbslvdmz",
    password = "CcZLqLOKWz9IjsMB3U3xGEpZu8goXYB9"
)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS usedCar")
cursor.execute("""
CREATE TABLE usedCar(
  ID VARCHAR(32),
  production_year INT,
  model_release INT,
  brand VARCHAR(32),
  model_name VARCHAR(32),
  Sales_City VARCHAR(32),
  sales_area VARCHAR(32),
  mileage INT,
  displacement INT,
  natural_gas INT,
  diesel INT,
  gasoline INT,
  hybrid INT,
  LPG INT,
  Price FLOAT);
""")
conn.commit()

query = """
    COPY usedCar FROM STDIN DELIMITER ',' CSV HEADER;
"""

with open("C:/Users/jaeho/Desktop/train.csv", 'r') as f:
    cursor.copy_expert(query, f)
conn.commit()

query = "SELECT * FROM usedCar;"
cursor = conn.cursor()
cursor.execute(query)
rows = cursor.fetchall()
columns = [i[0] for i in cursor.description]
cursor.close()
conn.close()
df = pd.DataFrame(rows, columns = columns)

# 모델 생성 및 학습

df = df.drop(columns=['id','sales_city','sales_area','brand','model_name'])
x = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 2)
#선형회귀
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("학습 데이터 : ",model.score(X_train,y_train))
# print("테스트 데이터 : ", model.score(X_test,y_test))
# print(r2_score(y_test,model.predict(X_test)))
# print(model.predict(X_test[:1]))

#랜덤포레스트
model = RandomForestRegressor(random_state= 2)
model.fit(X_train, y_train)
print("학습 데이터 : ", model.score(X_train,y_train))
print("cross_val_score : ", cross_val_score(model,X_train, y_train, cv = 3))
print("테스트 데이터 : ",model.score(X_test,y_test))




from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    production_year = request.form['production_year']
    model_release = request.form['model_release']
    mileage = request.form['mileage']
    displacement = request.form['displacement']
    fuel  = request.form.get('selected_text')
    
    natural_gas = 0
    diesel = 0
    gasoline = 0
    hybrid = 0
    lpg = 0
    print("fuel : ", fuel)
    if fuel == "natural_gas":
        natural_gas = 1
    elif fuel == "diesel":
        diesel = 1
    elif fuel == "gasoline":
        gasoline = 1
    elif fuel == "hybrid":
        hybrid = 1
    elif fuel == "lpg":
        lpg = 1
    tmp = [(int(production_year), int(model_release), int(mileage), int(displacement), natural_gas, diesel, gasoline, hybrid, lpg)]
    print(tmp)
    data= pd.DataFrame(tmp, columns = ['production_year', 'model_release','mileage', 'displacement', 'natural_gas', 'diesel', 'gasoline', 'hybrid', 'lpg'])
    result = model.predict(data)
    
    return "중고차 가격: {}원".format(round(result[0]*100000))

if __name__ == '__main__':
    app.run()

