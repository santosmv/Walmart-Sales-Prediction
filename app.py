import plotly.graph_objects as go
import streamlit as st
from main import Preprocessing, Predictor
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from time import sleep

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
    
    def cyclic_prediction(self):
        preprocessing = Preprocessing()
        predictor = Predictor()
        X, y, timeline, store = preprocessing.fit_transform(self.data)
        predictor.fit_predict(X, y, timeline, store)
    
    def get_old_score(self, str_score, new_score):
        try:
            query_old = text(f"SELECT MAX({str_score}) AS second FROM Predictors WHERE {str_score} < (SELECT MAX({str_score}) FROM Predictors)")
            old_score = self.conn.execute(query_old).fetchall()[0][0]
        except:
            old_score = new_score
        return old_score
    
    def data_refresh(self):
        engine = create_engine('sqlite+pysqlite:///data/database.db')
        self.conn = engine.connect()

        query_best_pred = text("SELECT id, set_id, name, score_mape_test, score_rmse_test FROM Predictors WHERE score_mape_test = (SELECT MAX(score_mape_test) FROM Predictors)")
        pred_id, set_id, self.name_pred, self.mape, self.rmse = self.conn.execute(query_best_pred).fetchall()[0]
        
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

        self.old_mape = self.get_old_score('score_mape_test', self.mape)
        self.old_rmse = self.get_old_score('score_rmse_test', self.rmse)

    def prediction_plot(self):
        layout = go.Layout(height=600)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=self.timeline_train, y=self.y_train, mode='lines+markers', line_color='limegreen', name='Train'))
        fig.add_trace(go.Scatter(x=self.timeline_test, y=self.y_test, mode='lines+markers', line_color='red', name='Test'))
        fig.add_trace(go.Scatter(x=self.timeline_train, y=self.y_pred_train, mode='lines+markers', line_color='yellow', name='Prediction (train)'))
        fig.add_trace(go.Scatter(x=self.timeline_test, y=self.y_pred_test, mode='lines+markers', line_color='darkorange', name='Prediction (test)'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Sales ($)", font={'size':32})
        return fig

    def app(self):
        self.cyclic_prediction()
        self.data_refresh()

        st.set_page_config(layout="wide")

        # SIDEBAR
        sidebar = st.sidebar.empty()
        # stores = [1,2]
        # store = st.sidebar.selectbox('Select the Walmart store:', stores)

        button = st.sidebar.button('Make new prediction')
        place_msg = st.sidebar.empty()
        
        if button:
            place_msg.write(":grey[New prediction in course...]")

            self.cyclic_prediction()
            self.data_refresh()

            place_msg.write('')

        # FRONT
        pred_plot = st.empty()

        # Row A
        # st.markdown('### Metrics')
        c1, c2, c3 = st.columns(3)

        percent_mape = self.mape / self.old_mape - 1
        percent_rmse = 1 - self.rmse / self.old_rmse
        c1.metric('MAPE', '%.2f%%'%(self.mape * 100), '%.2f%%'%percent_mape)
        c2.metric('RMSE', '%.1f'%self.rmse, '%.2f%%'%percent_rmse)
        c3.metric('Best Estimator', self.name_pred)
        
        # Plot prediction
        pred_fig = self.prediction_plot()
        pred_plot = st.plotly_chart(pred_fig, use_container_width=True)

        
        
dash = Dashboard()
dash.app()