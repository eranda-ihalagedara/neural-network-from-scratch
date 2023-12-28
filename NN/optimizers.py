import numpy as np

class rmsprop:
    def __init__(self, w_shape, b_shape, beta=0.9) -> None:
        self.beta = beta
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

    
    def get_dw_opt(self, dw):
        self.sdw = self.beta*self.sdw + (1-self.beta)*dw**2

        # For numerical stability
        epsilon = 1e-10
        return dw/np.sqrt(self.sdw + epsilon)
    

    def get_db_opt(self, db):
        self.sdb = self.beta*self.sdb + (1-self.beta)*db**2

        # For numerical stability
        epsilon = 1e-8
        return db/np.sqrt(self.sdb + epsilon)
    
class sgd:
    def __init__(self) -> None:
        pass
    def get_dw_opt(self, dw):
        return dw
    def get_db_opt(self, db):
        return db

class adam:
    def __init__(self, w_shape, b_shape, beta1=0.9, beta2=0.999) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.vdw = np.zeros(w_shape)
        self.vdb = np.zeros(b_shape)
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)


    def get_dw_opt(self, dw):
        self.vdw = self.beta1*self.vdw + (1-self.beta1)*dw
        self.sdw = self.beta2*self.sdw + (1-self.beta2)*dw**2
        # For numerical stability
        epsilon = 1e-10
        return self.vdw/np.sqrt(self.sdw + epsilon)
        return dw/np.sqrt(self.sdw + epsilon)


    def get_db_opt(self, db):
        self.vdb = self.beta1*self.vdb + (1-self.beta1)*db
        self.sdb = self.beta2*self.sdb + (1-self.beta2)*db**2
        # For numerical stability
        epsilon = 1e-10
        return self.vdb/np.sqrt(self.sdb + epsilon)


def get_optimizer(opt, w_shape, b_shape):
        if opt == 'rmsprop':
            return rmsprop(w_shape, b_shape)
        elif opt == 'adam':
            return adam(w_shape, b_shape)
        elif opt == 'sgd':
            return sgd()
        else:
            raise Exception('\'' + str(opt) + '\' optimizer not found!')