import opendatasets as od
iris = od.download("https://www.kaggle.com/datasets/uciml/iris")
print("Iris dataset downloaded successfully.")
print("Dataset location:", iris)
