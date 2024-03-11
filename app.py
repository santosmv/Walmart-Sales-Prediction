import plotly.graph_objects as go
import streamlit as st
from main import Preprocessing, Predictor
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
import numpy as np
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self) -> None:
        data_path = 'data/walmart-sales-dataset-of-45stores.csv'
        self.data = pd.read_csv(data_path)
        self.name_pred = None
        self.y_pred_test = None
        self.y_pred_train = None
        self.timeline_train = None
        self.timeline_test = None
        self.y_train = None
        self.y_test = None
        self.mape = None
        self.old_mape = None
        self.rmse = None
        self.old_rmse = None
        self.n_estimators, self.max_depth, self.eta, self.gamma, self.lambda_, self.tree_method, self.min_samples_split, self.min_samples_leaf, self.max_features, self.bootstrap = None,None,None,None,None,None,None,None,None,None
        self.mape_all, self.rmse_all, self.name_all = None,None,None
        self.test_frac, self.n_cv, self.n_splits = None,None,None

    def cyclic_prediction(self):
        preprocessing = Preprocessing()
        predictor = Predictor()
        X, y, timeline, store = preprocessing.fit_transform(self.data)
        predictor.fit_predict(X, y, timeline, store)
    
    def get_old_score(self, str_score, new_score):
        try:
            query_old = text(f"SELECT MIN({str_score}) FROM Predictors WHERE {str_score} > (SELECT MIN({str_score}) FROM Predictors)")
            old_score = self.conn.execute(query_old).fetchall()[0][0]
        except:
            old_score = new_score
        return old_score
    
    def data_refresh(self):
        engine = create_engine('sqlite+pysqlite:///data/database.db')
        self.conn = engine.connect()
        metadata = MetaData()

        query_best_pred = text("SELECT id, set_id, name, score_mape_test, score_rmse_test, n_estimators, max_depth, eta, gamma, lambda, tree_method, min_samples_split, min_samples_leaf, max_features, bootstrap FROM Predictors ORDER BY ABS(score_rmse_test) LIMIT 1")
        pred_id, set_id, self.name_pred, self.mape, self.rmse, self.n_estimators, self.max_depth, self.eta, self.gamma, self.lambda_, self.tree_method, self.min_samples_split, self.min_samples_leaf, self.max_features, self.bootstrap = self.conn.execute(query_best_pred).fetchall()[0]
        
        query_prediction_train = text("SELECT y FROM target_prediction WHERE pred_id = :pred_id AND train_test_prediction = 'prediction_train' AND predictor = :name_pred")
        y_pred_train = self.conn.execute(query_prediction_train, {'pred_id':pred_id, 'name_pred':self.name_pred}).fetchall()
        self.y_pred_train = np.float64(y_pred_train).ravel()
        
        query_prediction_test = text("SELECT y FROM target_prediction WHERE pred_id = :pred_id AND train_test_prediction = 'prediction_test' AND predictor = :name_pred")
        y_pred_test = self.conn.execute(query_prediction_test, {'pred_id':pred_id, 'name_pred':self.name_pred}).fetchall()
        self.y_pred_test = np.float64(y_pred_test).ravel()

        query_train = text("SELECT timeline, y FROM target_prediction WHERE train_test_prediction = 'train' AND set_id = :set_id")
        data_train = self.conn.execute(query_train, {'set_id':set_id}).fetchall()
        data_train = np.array(data_train)
        self.timeline_train, self.y_train = data_train[:,0], np.float64(data_train[:,1])

        query_test = text("SELECT timeline, y FROM target_prediction WHERE train_test_prediction = 'test' AND set_id = :set_id")
        data_test = self.conn.execute(query_test, {'set_id':set_id}).fetchall()
        data_test = np.array(data_test)
        self.timeline_test, self.y_test = data_test[:,0], np.float64(data_test[:,1])

        query_scores = text("SELECT score_mape_test, score_rmse_test, name FROM Predictors")
        data_scores = self.conn.execute(query_scores).fetchall()
        data_scores = np.array(data_scores)
        self.mape_all, self.rmse_all, self.name_all = data_scores[:,0], data_scores[:,1], data_scores[:,2]
        
        query_set = text('SELECT test_fraction, n_cv, n_splits FROM settings_data WHERE id = :set_id')
        self.test_frac, self.n_cv, self.n_splits = self.conn.execute(query_set, {'set_id':set_id}).fetchall()[0]

        query_table = text("SELECT * FROM Predictors ORDER BY score_rmse_test LIMIT 20")
        self.table = self.conn.execute(query_table).fetchall()

        table_predictors = Table('Predictors', metadata, autoload_with=engine)
        self.column_names_predictor = table_predictors.columns.keys()

        self.old_mape = self.get_old_score('score_mape_test', self.mape)
        self.old_rmse = self.get_old_score('score_rmse_test', self.rmse)
    
    def auto_correlation_plot(self):
        fig = tsaplots.plot_acf(self.y_train, lags=60, color='r', vlines_kwargs={'color':'k'})
        plt.xlabel('Lag at k', color='grey')
        plt.ylabel('Autocorrelation Coefficient', color='grey')
        return fig

    def prediction_plot(self):
        layout = go.Layout(height=600)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=self.timeline_train, y=self.y_train, mode='lines+markers', line_color='limegreen', name='Train'))
        fig.add_trace(go.Scatter(x=self.timeline_test, y=self.y_test, mode='lines+markers', line_color='red', name='Test'))
        fig.add_trace(go.Scatter(x=self.timeline_train, y=self.y_pred_train, mode='lines+markers', line_color='yellow', name='Prediction (train)', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=self.timeline_test, y=self.y_pred_test, mode='lines+markers', line_color='darkorange', name='Prediction (test)'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Sales ($)", font={'size':32}, legend=dict(x=0.8,y=0.95))
        return fig

    def scores_plot(self):
        layout = go.Layout(height=600)
        # fig_scores = go.Figure(layout=layout, layout_yaxis_range=[0, float(max(self.rmse_all)) + 2e4])
        fig_scores = go.Figure(layout=layout)
        marker_colors = ['blue' if val == 'XGBoost' else 'red' if val == 'Random_Forest' else None for val in self.name_all if val == 'XGBoost' or val == 'Random_Forest']
        fig_scores.add_trace(go.Scatter(x=np.array(self.mape_all, dtype=np.float64) * 100, y=self.rmse_all, mode='markers', marker=dict(color=marker_colors, size=12), showlegend=False))
        fig_scores.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='XGBoost', marker=dict(color='blue', size=12)))
        fig_scores.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Random Forest', marker=dict(color='red', size=12)))        
        fig_scores.update_layout(xaxis_title="MAPE (%)", yaxis_title="RMSE", font={'size':32})
        return fig_scores

    def app(self):
        # self.cyclic_prediction()
        self.data_refresh()

        st.set_page_config(layout="wide")

        # SIDEBAR
        sidebar = st.sidebar.empty()
        # stores = [1,2]
        # store = st.sidebar.selectbox('Select the Walmart store:', stores)

        # TITLE
        st.title('Analysing Walmart Sales')
        st.markdown('This dashboard provides a comprehensive view of the forecasting analysis for Walmart Inc sales. The data can be found in [Walmart Sales Dataset of 45 stores](https://www.kaggle.com/datasets/varsharam/walmart-sales-dataset-of-45stores/data).')

        # METRICS OF THE BEST MODEL
        st.header('Best Model')
        c1, c2, c3 = st.columns(3)
        percent_mape = self.mape / self.old_mape - 1
        percent_rmse = 1 - self.rmse / self.old_rmse
        c1.metric('RMSE', '%.1f'%self.rmse, '%.2f%%'%percent_rmse)
        c2.metric('MAPE', '%.4f%%'%(self.mape * 100), '%.2f%%'%percent_mape)
        c3.metric('Best Estimator', self.name_pred)
        
        # PLOT PREDICTION
        pred_plot = st.empty()
        pred_fig = self.prediction_plot()
        pred_plot = st.plotly_chart(pred_fig, use_container_width=True)
        
        # MODEL METRICS
        st.slider("train/test split", 0, 100, (75, 100))
        # Row A
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(r'$n$ estimators', self.n_estimators)
        c2.metric('max depth', self.max_depth)
        c3.metric('eta', self.eta)
        c4.metric('gamma', self.gamma)
        c5.metric('lambda', self.lambda_)
        # Row B
        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric('tree method', self.tree_method)
        c7.metric('min samples split', self.min_samples_split)
        c8.metric('min samples leaf', self.min_samples_leaf)
        c9.metric('max features', self.max_features)
        c10.metric('bootstrap', self.bootstrap)
        st.divider()

        # CONFIGURATION METRICS
        st.header('Configuration metrics')
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric('Number of trainings', len(self.mape_all))
        c2.metric('Data points train', len(self.y_train))
        c3.metric('Data points test', len(self.y_test))
        c4.metric('CV batches', self.n_cv)
        c5.metric('Splits (train/test)', self.n_splits)
        c6.metric('Test fraction', '%i%%'%(self.test_frac * 100))
        st.metric('Models used', ', '.join(set(self.name_all)))

        # TABLE BEST MODELS
        table_preds = pd.DataFrame(self.table, columns=(self.column_names_predictor))
        st.table(table_preds)

        # PLOT SCORES
        st.header('RMSE vs MAPE scores')
        scores_fig = self.scores_plot()
        st.plotly_chart(scores_fig, use_container_width=True)

        #BUTTON
        button = st.sidebar.button('Update data')
        place_msg = st.sidebar.empty()
        if button:
            place_msg.write(":grey[Updating...]")
            # self.cyclic_prediction()
            self.data_refresh()
            place_msg.write('')
        
        
dash = Dashboard()
dash.app()