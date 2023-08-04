
print('hhelo')
y = df['price']
X = df[['area', 'bedrooms', 'bathrooms']]
X_norm = (X - X.min()) / (X.max() - X.min())


y_norm = (y - y.min()) / (y.max() - y.min())

ones_column = np.ones(y.size).reshape(-1,1)

#X_norm = np.hstack((ones_column, X_norm))


y_n = pd.DataFrame(y_norm, columns=['price'])
y_n = y_n['price']
y_n


X_n = pd.DataFrame(X_norm, columns=['area', 'bedrooms', 'bathrooms'])
X_n = X_n[['area', 'bedrooms', 'bathrooms']]
X_n



class SimpleLinearRegression:
  def __init__(self, learning_rate : float = 0.001, threshold: float = 0.0001, n_epoch: int =10000):
    self.learning_rate = learning_rate
    self.threshold  = threshold
    self.n_epoch = n_epoch
    self.w = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

  def predict(self, X: np.array) -> np.array:
    return X @ self.w

  def update_w(self, X, y):
    m = len(y)
    h = self.predict(X)
    self.w -= self.learning_rate / m * X.T @ (h - y)


  def fit_values(self, X: np.array, y: np.array):
        last_cost = 1000000
        for i in range(self.n_epoch):
            self.update_w(X, y)
            new_cost = loss_func(self.predict(X), y)
            if last_cost - new_cost < self.threshold :
                break
            last_cost = new_cost

        return new_cost, list(self.w)


mlr = SimpleLinearRegression(learning_rate=0.001, n_epoch=100000000, threshold =0.00001)

r = mlr.fit_values(X_n, y_n)
print(r)