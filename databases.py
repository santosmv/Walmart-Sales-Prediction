from sqlalchemy import create_engine, Table, Column, String, Integer, Boolean, MetaData, Float, Date, DateTime
import os

def create_table_predictors():
    engine = create_engine("sqlite+pysqlite:///data/database.db")
    metadata = MetaData()

    predictors = Table('Predictors', metadata,
                  Column('id', Integer, primary_key=True, autoincrement=True),
                  Column('set_id', Integer),
                  Column('name', String),
                  Column('score_mape_cv', Float),
                  Column('score_mape_test', Float),
                  Column('score_rmse_test', Float),
                  Column('refit_time_', Float),
                  Column('n_estimators', Integer),
                  Column('max_depth', Integer),
                  Column('min_samples_split', Integer),
                  Column('min_samples_leaf', Integer),
                  Column('max_features', String),
                  Column('bootstrap', Boolean),
                  Column('eta', Float),
                  Column('gamma', Float),
                  Column('subsample', Float),
                  Column('lambda', Float),
                  Column('tree_method', String)
                  )

    metadata.create_all(engine)

def create_table_X_data():
    engine = create_engine("sqlite+pysqlite:///data/database.db")
    metadata = MetaData()

    table_X = Table('X_data', metadata,
            Column('id', Integer),
            Column('set_id', Integer),
            Column('target_id', Integer),
            Column('train_test', String),
            Column('1', Float),
            Column('2', Float),
            Column('3', Float),
            Column('4', Float),
            Column('5', Float),
            Column('6', Float),
            Column('7', Float),
            Column('8', Float)
            )

    metadata.create_all(engine)

def create_table_target_prediction():
    engine = create_engine("sqlite+pysqlite:///data/database.db")
    metadata = MetaData()

    table_target = Table('target_prediction', metadata,
                  Column('id', Integer),
                  Column('set_id', Integer),
                  Column('pred_id', Integer), #only for predictions
                  Column('X_id', Integer),
                  Column('predictor', String), #only for predictions
                  Column('timeline', Date),
                  Column('train_test_prediction', String),
                  Column('y', Float)
                  )

    metadata.create_all(engine)

def create_table_settings_data():
    engine = create_engine("sqlite+pysqlite:///data/database.db")
    metadata = MetaData()

    table_settings = Table('settings_data', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('target_id', Integer),
                    Column('X_id', Integer),
                    Column('date_time', DateTime),
                    Column('store', Integer),
                    Column('test_fraction', Float),
                    Column('n_splits', Integer),
                    Column('n_cv', Integer),
                    Column('n_iter', Integer)
                    )

    metadata.create_all(bind=engine)

def drop_all_tables():
    engine = create_engine("sqlite+pysqlite:///data/database.db")
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)
    
    tables = metadata.tables.keys()

    if len(tables) == 0:
        print('All tables were successfully dropped!')
    else:
        print('Error when dropping tables: %i tables remaining!'%len(tables))
        exit()

if __name__ == '__main__':
    path = 'data/database.db'
    check_file = os.path.isfile(path)

    if check_file:
        asw = False
        # while asw != 'y' and asw != 'n':
        #     asw = input("Are you sure? All tables and data will be deleted. (y/n): ")

        if asw == 'n':
            print('Exit...')
            exit()
        else:
            drop_all_tables()
    
    print('Creating new tables...')
    create_table_predictors()
    create_table_X_data()
    create_table_target_prediction()
    create_table_settings_data()
    print('Tables were successfully created!')