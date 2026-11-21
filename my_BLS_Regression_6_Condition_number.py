import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE_func
from sklearn.metrics import mean_absolute_percentage_error as MAPE_func

def percentile_clip_bounds(X, lower_percentile=0.5, upper_percentile=99.5):
    """Calculates robust clip bounds based on percentiles."""
    clip_min = np.percentile(X, lower_percentile, axis=0)
    clip_max = np.percentile(X, upper_percentile, axis=0)
    return clip_min, clip_max

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu6(x):
    return np.clip(x, 0, 6)

def pseudoinverse(A, lambd):
    # return (np.linalg.inv(A.T.dot(A) + lambd * np.eye(A.shape[1]))).dot(A.T)
    # return np.linalg.pinv(A, rcond=lambd)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_reg = S / (S**2 + lambd)
    A_pinv = Vt.T @ np.diag(S_reg) @ U.T
    return A_pinv

def soft_thresholding(a, shrink):
    # print('****  ', a.max(), a.min())
    # shrink = a.max()
    return np.sign(a) * np.maximum((abs(a) - shrink), 0)

def ADMM(A, y, shrink_coef = 0.001, iter = 50):
    
    n_features = A.shape[1]
    n_outputs = y.shape[1]
    W_k = np.zeros(shape=(n_features,n_outputs), dtype='double')
    O_k = np.zeros(shape=(n_features,n_outputs), dtype='double')
    U_k = np.zeros(shape=(n_features,n_outputs), dtype='double')

    part_0 = np.linalg.inv(A.T.dot(A) + np.eye(n_features))
    part_1 = part_0.dot(A.T).dot(y)

    for k in range(iter):
        W_k = part_1 + part_0.dot(O_k - U_k)                     # W(k+1)
        O_k = soft_thresholding(W_k + U_k, shrink_coef)          # O(k+1)
        U_k += W_k - O_k                                         # U(k+1)
        W_k = O_k
    
    return W_k

class BLS_regression:
    '''
    (self, lambd, n_features, n_ftr_groups, n_enhancements, n_enh_groups, buffer_size)
    condition = all, none, scale

    '''
    def __init__(self, lambd, shrink, n_features, n_ftr_groups, n_enhancements, n_enh_groups, buffer_size=30, activation = 'sigmoid', condition = 'all'):
        self.shrink = shrink
        self.lambd = lambd
        self.n_features = n_features
        self.n_ftr_groups = n_ftr_groups
        self.n_enhancements = n_enhancements
        self.n_enh_groups = n_enh_groups
        self.buffer_size = buffer_size
        self.ne = (buffer_size ** 2) / (2 * buffer_size)
        self.limit = int(np.ceil(np.sqrt(np.log(0.025) * -1 * buffer_size)))
        self.activation = relu6 if activation == 'relu6' else sigmoid
        self.scaler = StandardScaler()
        self.clip_min = None
        self.clip_max = None
        self.condition = condition

    def fit(self, X, y):
        #start timing
        time_start = time.time()
        
        #number of all feature nodes
        n_Zn = self.n_features * self.n_ftr_groups
        
        #generate weights for feature nodes (We)
        We = []
        for i in range(self.n_ftr_groups):
            np.random.seed(i)
            We.append(2 * np.random.randn(X.shape[1]+1,self.n_features) - 1)
        
        #generate weights for enhancement nodes (Wh)
        Wh = []
        for j in range(self.n_enh_groups):
            np.random.seed(j)
            Wh.append(2 * np.random.randn(n_Zn + 1, self.n_enhancements) - 1)

        if self.condition == 'all':
            #input scaling + clipping
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            self.clip_min, self.clip_max = percentile_clip_bounds(X_scaled)
            X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        elif self.condition == 'scale':
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

        elif self.condition == 'none':
            X_scaled = X
        
        else:
            print('Error')
            return None
        
        #add bias feature
        X_biased = np.hstack([X_scaled, 0.1 * np.ones((X.shape[0],1))])

        #define Zn
        Zn = np.zeros((X.shape[0], n_Zn))
        
        #define We
        We_sparse = []

        #for linear mapping (phi)
        win_dist = np.zeros(self.n_ftr_groups)
        win_mean = np.zeros(self.n_ftr_groups)

        #calculating Zn
        for i in range(self.n_ftr_groups):
            Zi_temp = np.dot(X_biased, We[i])   #X.W
            
            scaler = MinMaxScaler(feature_range=(-1,1))      #scale Zi to range (-1, 1)
            Zi_temp_scaled = scaler.fit_transform(Zi_temp)

            We_sparse.append(ADMM(X_biased, Zi_temp_scaled, self.shrink))   #update Wei and make it sparse, then append it to We (its learning the weights for one iteration, sparse AE?)
            Zi_new = np.dot(X_biased, We_sparse[i])   #calculate Zi with new updated sparse weights

            win_dist[i], win_mean[i]= Zi_new.max() - Zi_new.min(), Zi_new.mean()
            Zi = (Zi_new - win_mean[i]) / win_dist[i]   #linear mapping function (phi)

            Zn[:, self.n_features * i : self.n_features * (i + 1)] = Zi

        #making enhancement nodes
        Zn_biased = np.hstack([Zn, 0.1 * np.ones((X.shape[0], 1))])
        Hn = np.zeros((X.shape[0], self.n_enhancements * self.n_enh_groups))
        for j in range(self.n_enh_groups):
            Hn[:,j*self.n_enhancements : (j+1)*self.n_enhancements] = self.activation(np.dot(Zn_biased, Wh[j]))

        #making An
        An = np.hstack([Zn, Hn])

        #calculating Wn = An+ . y
        Wn = np.dot(pseudoinverse(An, self.lambd), y)

        #stop time
        fit_time = time.time() - time_start

        #calculating errors
        y_pred = np.dot(An, Wn)
        self.fit_RMSE = RMSE_func(y, y_pred)
        self.fit_MAPE = MAPE_func(y, y_pred)

        self.Wn = Wn
        self.We = We_sparse
        self.Wh = Wh
        self.win_dist = win_dist
        self.win_mean = win_mean
        self.An = An
        self.pred_y = y_pred

        self.fit_time = fit_time
        
        print(f'The condition number in {self.condition} is:\t{np.linalg.cond(An.T @ An)} and\t{np.linalg.cond((An.T @ An) + self.lambd * np.eye(An.shape[1]))}')

        return True

    def predict(self, X, y=None):
        #start timing
        time_start = time.time()

        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        X_biased = np.hstack([X_scaled, 0.1 * np.ones((X.shape[0], 1))])
        Zn = np.zeros((X.shape[0], self.n_features * self.n_ftr_groups))

        for i in range(self.n_ftr_groups):
            Zi_temp = np.dot(X_biased, self.We[i])
            Zi = (Zi_temp - self.win_mean[i]) / self.win_dist[i]
            Zn[:, self.n_features * i : self.n_features * (i+1)] = Zi

        Zn_biased = np.hstack([Zn, 0.1 * np.ones((Zn.shape[0], 1))])

        Hn = np.zeros((X.shape[0], self.n_enh_groups * self.n_enhancements))
        for j in range(self.n_enh_groups):
            Hn[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))

        An = np.hstack([Zn, Hn])

        y_pred = np.dot(An, self.Wn)

        pred_time = time.time() - time_start

        if y is not None:
            self.pred_RMSE = RMSE_func(y, y_pred)
            self.pred_MAPE = MAPE_func(y, y_pred)
        
        self.pred_time = pred_time
        self.pred_y = y_pred
        
        return y_pred


    def fit_incremental(self, Xa, y):
        time_start = time.time()

        Xa_scaled = self.scaler.transform(Xa)
        Xa_scaled = np.clip(Xa_scaled, self.clip_min, self.clip_max)

        Xa_biased = np.hstack([Xa_scaled, 0.1 * np.ones(shape=(Xa.shape[0], 1))])
        n_Zn = self.n_features * self.n_ftr_groups
        Zn_new = np.zeros((Xa.shape[0], n_Zn))

        # win_dist, win_mean= np.zeros(self.n_ftr_groups), np.zeros(self.n_ftr_groups)
        
        for i in range(self.n_ftr_groups):
            Zi_temp = np.dot(Xa_biased, self.We[i])
            # win_dist[i], win_mean[i]= Zi_temp.max() - Zi_temp.min(), Zi_temp.mean()
            # Zi = (Zi_temp - win_mean[i]) / win_dist[i]
            Zi = (Zi_temp - self.win_mean[i]) / self.win_dist[i]
            Zn_new[:, self.n_features*i:self.n_features*(i+1)] = Zi

        Zn_biased = np.hstack([Zn_new, 0.1 * np.ones((Zn_new.shape[0], 1))])

        Hn_new = np.zeros((Xa.shape[0], self.n_enh_groups * self.n_enhancements))
        for j in range(self.n_enh_groups):
            Hn_new[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))


        Ax = np.hstack([Zn_new, Hn_new]) 

        An_new = np.vstack([self.An, Ax])

        An_psi = pseudoinverse(self.An, self.lambd)
        DT = Ax @ An_psi
        CT = Ax - (DT @ self.An)
        if np.allclose(CT, 0, atol=1e-8):
            B = An_psi @ DT.T @ np.linalg.inv(np.eye(DT.shape[0]) + np.dot(DT, DT.T))
        else:
            B = pseudoinverse(CT, self.lambd)

        An_new_psi = np.hstack([An_psi - (B @ DT), B])
        
        wn = self.Wn.reshape(-1,1)
        y = y.reshape(-1,1)
        Wn_new = wn + B @ (y - Ax @ wn)

        self.fit_inc_time = time.time() - time_start

        if np.any(np.isnan(Wn_new)) or np.any(np.isinf(Wn_new)):
            print('NaN or Inf detected in Wn!')

        self.An = An_new
        self.Wn = Wn_new.reshape(len(Wn_new))
        self.pred_y = np.dot(Ax , Wn_new)
        self.inc_RMSE = RMSE_func(y, self.pred_y)
        self.inc_MAPE = MAPE_func(y, self.pred_y)
        # self.win_dist = win_dist
        # self.win_mean = win_mean
        
        return True

        
    def fit_transfer(self, X, y):
        time_start = time.time()
        Zn = np.zeros((X.shape[0], self.n_features * self.n_ftr_groups))

        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        X_Biased = np.hstack([X_scaled, 0.1 * np.ones(shape=(X.shape[0],1))])

        win_dist, win_mean= np.zeros(self.n_ftr_groups), np.zeros(self.n_ftr_groups)

        for i in range(self.n_ftr_groups):
            Zi = np.dot(X_Biased, self.We[i])
            win_dist[i], win_mean[i]= Zi.max() - Zi.min(), Zi.mean()
            Zi_scaled = (Zi - win_mean[i]) / win_dist[i]
            Zn[:, i*self.n_features: (i+1)*self.n_features] = Zi_scaled
        
        Zn_biased = np.hstack([Zn, 0.1*np.ones((Zn.shape[0], 1))])

        Hn = np.zeros((X.shape[0], self.n_enh_groups * self.n_enhancements))
        for j in range(self.n_enh_groups):
            Hn[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))

        An = np.hstack([Zn, Hn])

        #calculating Wn = An+ . y
        Wn = np.dot(pseudoinverse(An, self.lambd), y)
        # print(Zn.shape, Hn.shape, An.shape)
        #calculating errors

        self.fit_trns_time = time.time() - time_start

        if np.any(np.isnan(Wn)) or np.any(np.isinf(Wn)):
            print('NaN or Inf detected in Wn!')

        y_pred = np.dot(An, Wn)
        self.trns_RMSE = RMSE_func(y, y_pred)
        self.trns_MAPE = MAPE_func(y, y_pred)

        self.Wn = Wn
        self.An = An
        self.pred_y = y_pred

        self.win_dist = win_dist
        self.win_mean = win_mean

        return True
    
    def ks_drift_detector2(self, old_buffer, new_buffer):
        A = np.sort(old_buffer)
        B = np.sort(new_buffer)
        # print('tttt')
        # print(old_buffer, A)
        # print(new_buffer, B)
        index_A, index_B, count_A, count_B, D = 0, 0, 0, 0, 0
        test = 0
        n = len(new_buffer)
        while index_A != n and index_B != n and D != self.limit:
            if A[index_A] > B[index_B]:
                count_B += 1
                index_B += 1
                if abs(count_A - count_B) > D:
                    D += 1

            elif A[index_A] < B[index_B]:
                count_A += 1
                index_A += 1
                if abs(count_A - count_B) > D:
                    D += 1

            else:
                count_A += 1
                count_B += 1
                index_A += 1
                index_B += 1
            test += 1
        # print(test)
        return D >= self.limit

    def ks_drift_detector3(self, old_buffer, new_buffer, sensitivity):
        A = np.sort(old_buffer)
        B = np.sort(new_buffer)
        # print('tttt')
        # print(old_buffer, A)
        # print(new_buffer, B)
        index_A, index_B, count_A, count_B, D = 0, 0, 0, 0, 0
        test = 0
        n = len(new_buffer)
        while index_A != n and index_B != n and D != self.limit:
            if A[index_A] > (sensitivity + 1) * B[index_B]:
                count_B += 1
                index_B += 1
                if abs(count_A - count_B) > D:
                    D += 1

            elif A[index_A] * (sensitivity + 1) < B[index_B]:
                count_A += 1
                index_A += 1
                if abs(count_A - count_B) > D:
                    D += 1

            else:
                count_A += 1
                count_B += 1
                index_A += 1
                index_B += 1
            test += 1
        # print(test)
        return D >= self.limit


    def inc_enhancement_groups(self, n_new_enh_groups, y):
        time_start = time.time()
        Zn = self.An[:, :self.n_features * self.n_ftr_groups]
        Zn_biased = np.hstack([Zn, 0.1*np.ones((Zn.shape[0], 1))])
        
        Hnp = np.zeros((Zn.shape[0], self.n_enhancements * n_new_enh_groups))
        for i in range(n_new_enh_groups):
            Whp_temp = 2 * np.random.randn(self.n_features * self.n_ftr_groups + 1, self.n_enhancements) - 1
            self.Wh.append(Whp_temp)
            Hnp[:, i * self.n_enhancements : (i+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, Whp_temp))

        An_new = np.hstack([self.An, Hnp]) #update An

        An_psi = pseudoinverse(self.An, self.lambd)

        D = An_psi @ Hnp
        C = Hnp - (self.An @ D)

        if np.allclose(C, 0, atol=1e-8):
            BT = np.linalg.inv(np.eye(D.shape[1]) + np.dot(D.T, D)) @ D.T @ An_psi
        else:
            BT = pseudoinverse(C, self.lambd)

        An_new_psi = np.vstack([An_psi - (D @ BT), BT])

        Wn_new = np.vstack([self.Wn.reshape(-1,1) - (D @ BT @ y.reshape(-1,1)), BT @ y.reshape(-1,1)])
        
        self.inc_enh_time = time.time() - time_start

        self.An = An_new
        self.Wn = Wn_new
        self.n_enh_groups += n_new_enh_groups
        

        self.fit_RMSE = RMSE_func(y, An_new @ Wn_new)
        self.fit_MAPE = MAPE_func(y, An_new @ Wn_new)

        return True
    

    def inc_feature_groups(self, n_new_ftr_groups, n_new_enh_groups, X, y):
        time_start = time.time()
        Zn = self.An[:, :self.n_features * self.n_ftr_groups]

        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        X_biased = np.hstack([X_scaled, 0.1 * np.ones((X.shape[0],1))])
        Wep = []
        
        for i in range(n_new_ftr_groups):
            np.random.seed(i)
            Wep_temp = 2 * np.random.randn(X.shape[1] + 1, self.n_features) - 1
            Wep.append(Wep_temp)
            self.We.append(Wep_temp)
        
        Znp = np.zeros((X.shape[0], self.n_features * n_new_ftr_groups))

        win_dist_p = np.zeros(n_new_ftr_groups)
        win_mean_p = np.zeros(n_new_ftr_groups)

        for i in range(n_new_ftr_groups):
            Zip_temp = X_biased @ Wep[i]
            win_dist_p[i] = Zip_temp.max() - Zip_temp.min()
            win_mean_p[i] = Zip_temp.mean()
            Zip = (Zip_temp - win_mean_p[i]) / win_dist_p[i]
            Znp[:, i * self.n_features : (i+1) * self.n_features] = Zip

        Zn_new = np.hstack([Zn, Znp])
        Znp_biased = np.hstack([Znp, 0.1 * np.ones((Znp.shape[0], 1))])

        Hnp = np.zeros((X.shape[0], n_new_enh_groups * self.n_enhancements))
        Whp = []
        for j in range(n_new_enh_groups):
            Whp_temp = 2 * np.random.randn(Znp.shape[1]+1, self.n_enhancements) - 1
            Whp.append(Whp_temp)
            self.Wh.append(Whp_temp)
            Hnp[:, j * self.n_enhancements : (j+1) * self.n_enhancements] = self.activation(Znp_biased @ Whp_temp)

        Anp = np.hstack([Znp, Hnp])
        An_new = np.hstack([self.An, Anp])

        An_psi = pseudoinverse(self.An, self.lambd)
        D = An_psi @ Anp
        C = Anp - (self.An @ D)

        if np.allclose(C, 0, atol=1e-8):
            BT = np.linalg.inv(np.eye(D.shape[1]) + np.dot(D.T, D)) @ D.T @ An_psi
        else:
            BT = pseudoinverse(C, self.lambd)

        An_new_psi = np.vstack([An_psi - (D @ BT), BT])
        Wn_new = np.vstack([self.Wn.reshape(-1,1) - (D @ BT @ y.reshape(-1,1)), BT @ y.reshape(-1,1)])

        self.inc_ftr_time = time.time() - time_start

        self.An = An_new
        self.Wn = Wn_new
        # self.We.append(Wep)

        # Wh_new = np.zeros((self.Wh.shape[0]+Whp.shape[0]-1, self.Wh.shape[1]+Whp.shape[1]))
        # Wh_new[:self.Wh.shape[0],:self.Wh.shape[1]] = self.Wh
        # Wh_new[self.Wh.shape[0]:,self.Wh.shape[1]:] = Whp[:Whp.shape[0]-1,:]
        # self.Wh = Wh_new
        # self.Wh = np.vstack([self.Wh , Whp])
        y_pred = An_new @ Wn_new
        self.fit_RMSE = RMSE_func(y, y_pred)
        self.fit_MAPE = MAPE_func(y, y_pred)
        self.win_dist = np.hstack([self.win_dist, win_dist_p])
        self.win_mean = np.hstack([self.win_mean, win_mean_p])
        self.n_ftr_groups += n_new_ftr_groups
        self.n_enh_groups += n_new_enh_groups

        return True

    