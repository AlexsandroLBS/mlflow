import warnings

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    mlflow.set_experiment(experiment_name= 'mlflow_demo')
    
    df = pd.read_csv('water_potability.csv')
    print('Loading data')
    
    colunas = df.columns
    from sklearn.impute import SimpleImputer
    preenche = SimpleImputer(strategy = "mean")
    # Substituindo os valores faltantes, pela média dos dados
    df2 = preenche.fit_transform(df)
    df = pd.DataFrame(df2, columns= colunas)
    X = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, train_size= 0.7, random_state= 1)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_treino = sc.fit_transform(X_treino)
    X_teste = sc.transform(X_teste)
    print('Completed Feature scaling')
    
    #Treinamento do modelo
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_treino, y_treino)
    print('Model Trained')
    y_pred = classifier.predict(X_teste)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_teste, y_pred)
    print(cm)

    from sklearn.metrics import accuracy_score

    model_acc = accuracy_score(y_teste,y_pred)
    print(model_acc)
    mlflow.log_metric('accuracy',model_acc)
    mlflow.sklearn.log_model(classifier, 'model')






