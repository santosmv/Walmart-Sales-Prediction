import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer, mean_squared_error
from sqlalchemy import create_engine, Table, MetaData, text
from datetime import datetime
import numpy as np


class Preprocessing:
    def __init__(self, store=1) -> None:
        self.df = None
        self.timeline = None
        self.store = store

    def report(self):
        sum_null = self.df.isnull().sum()
        m,n = self.df.shape
        print('Rows = %i, Columns = %i'%(m,n))
        print('Number of NULLs and NANs for each column:')
        print(sum_null)
        
        if self.df.isnull().any().sum() == 0:
            return False, 0
        else:
            return True, self.df.isnull().any().sum()
    
    def fill_nulls(self):
        any_null, n_nulls = self.report()
        if any_null:
            self.df.fillna(method='ffill', inplace=True)
        
        if self.df.isnull().any().sum() != 0:
            print('\nThere are still NULLs and NANs!\nNulls before: %i\nNulls after preprocessing: %i'%(n_nulls, self.df.isnull().any().sum()))

    def feature_engineering(self):
        self.df.Date = pd.to_datetime(self.df.Date, format='%d-%m-%Y')
        self.timeline = self.df.groupby('Store').get_group(self.store)['Date']

        self.df['Month'] = self.df.Date.dt.month
        self.df['Week'] = self.df.Date.dt.isocalendar().week
        self.df['Day'] = self.df.Date.dt.dayofyear

        features = ['Month','Week','Day','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment']
        
        X = self.df.groupby('Store').get_group(self.store)[features]
        y = self.df.groupby('Store').get_group(self.store)['Weekly_Sales']
        return X, y

    def fit_transform(self, data):
        self.df = data
        self.fill_nulls()
        X, y = self.feature_engineering()
        return X.to_numpy(), y.to_numpy(), self.timeline, self.store


class Predictor:
    def __init__(self) -> None:
        self.test_fraction = 0.25
        self.n_splits = 2
        self.n_cv = 3
        self.n_iter = 3
        self.set_id = None
        self.target_id = None
        self.X_id = None
        self.pred_id = None
        self.database_connection()

    def models(self):
        rf_reg = RandomForestRegressor()
        xgb_reg = xgb.XGBRegressor()
        svm_reg = SVR()
        return rf_reg, xgb_reg
    
    def retrieve_last_id(self, table):
        query_last_id = table.select().with_only_columns(table.c.id)
        id_counts = self.conn.execute(query_last_id).fetchall()
        
        if len(id_counts) == 0:
            id = 1
        else:
            last_id = self.conn.execute(query_last_id).fetchall()[-1][0]
            id = last_id + 1
        return id
    
    def split_data(self, X, y, timeline):
        #Splitting the data in trainning and test sets
        test_size = int(self.test_fraction * len(X))
        tss = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)
        for i, (train_index, test_index) in enumerate(tss.split(X)):
            X_train, X_test, y_train, y_test, timeline_train, timeline_test = X[train_index], X[test_index], y[train_index], y[test_index], timeline[train_index], timeline[test_index]

        #Check if there are entries in the settings_data table
        query_settings = self.table_settings.select().where((self.table_settings.c.store == self.store) & (self.table_settings.c.test_fraction == self.test_fraction) & (self.table_settings.c.n_splits == self.n_splits) & (self.table_settings.c.n_cv == self.n_cv) & (self.table_settings.c.n_iter == self.n_iter))
        result = self.conn.execute(query_settings)
        
        if len(result.fetchall()) != 0:
            #Get the settings_data id
            query_set_id = text('SELECT id FROM settings_data WHERE store=:store AND test_fraction=:test_fraction AND n_splits=:n_splits AND n_cv=:n_cv AND n_iter=:n_iter')
            result_id = self.conn.execute(query_set_id, {'store':self.store, 'test_fraction':self.test_fraction, 'n_splits':self.n_splits, 'n_cv':self.n_cv, 'n_iter':self.n_iter})
            self.set_id = result_id.fetchall()[0][0]

            return X_train, X_test, y_train, y_test, timeline_train, timeline_test

        #Generating X_data id
        self.X_id = self.retrieve_last_id(self.table_X_data)

        #Generating target_prediction id
        self.target_id = self.retrieve_last_id(self.table_target)

        #Generating settings_data id
        self.set_id = self.retrieve_last_id(self.table_settings)

        #Insert the settings of the run in the settings_data table
        date_time = datetime.now()
        query_settings = self.table_settings.insert().values(id=self.set_id,
                                                             target_id=self.target_id, 
                                                             X_id=self.X_id,
                                                             date_time=date_time, 
                                                             store=self.store, 
                                                             test_fraction=self.test_fraction, 
                                                             n_splits=self.n_splits, 
                                                             n_cv=self.n_cv, 
                                                             n_iter=self.n_iter)
        result = self.conn.execute(query_settings)
        
        X_train_test = [X_train, X_test]
        y_train_test = [y_train, y_test]
        timeline_train_test = [timeline_train, timeline_test]
        train_test = ['train','test']

        #Insert data into X_data and y_data (train and test)
        for k in range(2):
            X_data = X_train_test[k]
            y_data = y_train_test[k]
            timeline_data = timeline_train_test[k]

            for i in range(X_data.shape[0]):
                X_dict = {'id' : self.X_id,
                        'set_id' : self.set_id,
                        'target_id' : self.target_id,
                        'train_test' : train_test[k],
                        '1': X_data[i,0],
                        '2': X_data[i,1],
                        '3': X_data[i,2],
                        '4': X_data[i,3],
                        '5': X_data[i,4],
                        '6': X_data[i,5],
                        '7': X_data[i,6],
                        '8': X_data[i,7]}
                query_X_data = self.table_X_data.insert().values(X_dict)
                self.conn.execute(query_X_data)

                query_target_data = self.table_target.insert().values(id=self.target_id,
                                                                      set_id=self.set_id,
                                                                      X_id=self.X_id,
                                                                      timeline=timeline_data.iloc[i],
                                                                      train_test_prediction=train_test[k],
                                                                      y=y_data[i])
                self.conn.execute(query_target_data)


        self.conn.commit()

        return X_train, X_test, y_train, y_test, timeline_train, timeline_test
    
    def random_search(self, model):
        if type(model).__name__ == 'RandomForestRegressor':
            distributions = dict(n_estimators=[5,10,40,100,300,500,1000,2000],
                                 max_depth=[3,5,10,20,30,40,50,100],
                                 min_samples_split=[2,3,4],
                                 min_samples_leaf=[1,2,3],
                                 max_features=[1,'sqrt','log2'],
                                 bootstrap=[True,False])
        
        elif type(model).__name__ == 'XGBRegressor':
            distributions = dict(eta=[0.1,0.3,0.5,0.7,0.9],
                                 gamma=[0,0.1,0.2,1],
                                 max_depth=[3,6,7,8,9,10,20,50,100],
                                 subsample=[0.5,1],
                                 **{'lambda': [1, 0.5, 2]},
                                 tree_method=['auto','exact','approx','hist'])

        MAPE = make_scorer(mean_absolute_percentage_error)
        rn = RandomizedSearchCV(estimator=model,
                                param_distributions=distributions,
                                cv=self.n_cv,
                                scoring=MAPE,
                                n_iter=self.n_iter)
        return rn
    
    def database_connection(self):
        engine = create_engine("sqlite+pysqlite:///data/database.db")
        metadata = MetaData()
        self.conn = engine.connect()
        self.table_predictors = Table('Predictors', metadata, autoload_with=engine)
        self.table_X_data = Table('X_data', metadata, autoload_with=engine)
        self.table_target = Table('target_prediction', metadata, autoload_with=engine)
        self.table_settings = Table('settings_data', metadata, autoload_with=engine)
    
    def fit_predict(self, X, y, timeline, store):
        self.store = store

        var_names = ['X_train', 'X_test', 'y_train', 'y_test', 'timeline_train', 'timeline_test']
        model_names = ['Random_Forest', 'XGBoost']
        train_test_names = ['_train','_test']
        data_dict, y_pred_dict = {}, {}
        
        Xy_vector = self.split_data(X, y, timeline)
        
        for j, var in enumerate(Xy_vector):
            data_dict[var_names[j]] = var

        X_train, X_test, y_train, y_test = data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict['y_test']

        for i, model in enumerate(self.models()):
            rn = self.random_search(model)
            rn.fit(X_train, y_train)

            for tt in train_test_names:
                y_pred_dict[model_names[i] + tt] = rn.best_estimator_.predict(data_dict['X' + tt])
            
            mape_test = rn.best_estimator_.score(X_test, y_test)
            y_pred_test = rn.best_estimator_.predict(X_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            #inserting results into Predictors table
            config_data = {'set_id':self.set_id,
                           'name':model_names[i],
                           'score_mape_cv':rn.best_score_,
                           'score_mape_test':mape_test,
                           'score_rmse_test':rmse_test,
                           'refit_time_':rn.refit_time_
                           }
            
            config_data.update(rn.best_params_)

            query_predictor = self.table_predictors.insert().values(config_data)
            self.conn.execute(query_predictor)
            self.conn.commit()

            query_pred_id = text('SELECT id FROM Predictors ORDER BY id')
            self.pred_id = self.conn.execute(query_pred_id).fetchall()[-1][0]

            #inserting prediction data into target_prediction table
            for tt in train_test_names:
                for i_ypred in y_pred_dict[model_names[i] + tt]:
                    query_pred_data = self.table_target.insert().values(id=self.target_id,
                                                                            set_id=self.set_id,
                                                                            pred_id=self.pred_id,
                                                                            X_id=self.X_id,
                                                                            predictor=model_names[i],
                                                                            train_test_prediction='prediction' + tt,
                                                                            y=i_ypred)
                    self.conn.execute(query_pred_data)
        
        self.conn.commit()
        self.conn.close()

        return data_dict, y_pred_dict