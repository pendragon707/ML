import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline
from heamy.dataset import Dataset

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

from sklearn.tree import export_graphviz
from IPython.display import Image
from io import StringIO
import graphviz
import pydotplus

def get_png_tree(tree_model_param, feature_names_param):
    dot_data = StringIO()
    export_graphviz(tree_model_param, out_file=dot_data, feature_names=feature_names_param, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph.create_png()

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/forestfires.csv', sep=",", nrows=500)
    return data

class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, loc=0.05, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(loc, a-0.05, str(round(b,3)), color='white')
        st.pyplot(fig)

@st.cache
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    data_out = data_out.drop(['day'], axis = 1)
    data_out = pd.concat([data_out, pd.get_dummies(data_out['month'], columns='month', drop_first=True)],axis=1)
    data_out.drop('month', axis=1, inplace=True)
    data_out['logarea'] = [math.log(data_out['area'][i]) if data_out['area'][i] > 0 else 0 for i in range(len(data_out['area']))]
    data_out = data_out.drop(['area'], axis = 1)
    min_max_scaler = MinMaxScaler()
    data_out[:] = min_max_scaler.fit_transform(data_out)
#    new_cols = []
#    for i in range(len(data.columns)):
#        col = data.columns[i]
#        new_col_name = col + '_scaled'
#        new_cols.append(new_col_name)
#        data_out[new_col_name] = data_out[:,i]
#    return data_out[new_cols], data_out['logarea']
    temp_y = data_out['logarea']
    temp_X = data_out.drop(['logarea'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, train_size=0.25, random_state=1)
    return data[:], X_train, X_test, y_train, y_test

# Модели
models_list = ['LinR', 'SVR', 'Tree', 'RF']
clas_models = {'LinR': LinearRegression(),
               'SVR': SVR(kernel='rbf', gamma=0.001, C=1000.0),
               'Tree': DecisionTreeRegressor(),
               'RF': RandomForestRegressor(n_estimators=15, oob_score=True, random_state=10)
#               'GB':GradientBoostingClassifier()}
               }

@st.cache(suppress_st_warning=True)
def print_models(models_select, metrics, X_train, X_test, y_train, y_test):
    current_models_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        y_pred = model.predict(X_test)

        RMSE = mean_squared_error(y_test, y_pred, squared=False)
        MAE = mean_absolute_error(y_test, y_pred)
        R2_Score = r2_score(y_test, y_pred)
        MedAE = median_absolute_error(y_test, y_pred)

        metrics.add('RMSE', model_name, RMSE)
        metrics.add('MAE', model_name, MAE)
        metrics.add('R2 Score', model_name, R2_Score)
        metrics.add('Median AE', model_name, MedAE)

        st.write('{} \t RMSE={}, MAE={}, R2={}, Median={}'.format(model_name, round(RMSE, 3), round(MAE, 3), round(R2_Score, 3), round(MedAE, 3)))

#    for metric in metrics.df['metric'].unique():
#        metrics.plot('Метрика: ' + metric, metric, figsize=(7, len(models_select)))
#        metrics.plot('Метрика: ' + metric, metric, figsize=(7, 5))

    metrics.plot('Метрика: ' + 'RMSE', 'RMSE', ascending=False, figsize=(7, 3))
    metrics.plot('Метрика: ' + 'MAE', 'MAE', ascending=False, figsize=(7, 3))
    metrics.plot('Метрика: ' + 'R2 Score', 'R2 Score', loc = -0.1, ascending=True, figsize=(7, 3))
    metrics.plot('Метрика: ' + 'Median AE', 'Median AE', ascending=False, figsize=(7, 3))

@st.cache(suppress_st_warning=True)
def print_tree(X_train, y_train, columns):
    tree = clas_models['Tree']
    tree.fit(X_train, y_train)
    st.image(get_png_tree(tree, list(X_train.columns)))

#######################

st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

data = load_data()
data, X_train, X_test, y_train, y_test = preprocess_data(data)
if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

# # Подбор гиперпараметра
# n_range_list = list(range(1,allowed_knn,step_slider))
# n_range = np.array(n_range_list)
# st.write('Возможные значения соседей - {}'.format(n_range))
# tuned_parameters = [{'n_neighbors': n_range}]


st.subheader('Оценка качества модели')
metrics = MetricLogger()
if len(models_select)>0:
    print_models(models_select, metrics, X_train, X_test, y_train, y_test)

if 'Tree' in models_select:
    if st.checkbox('Показать дерево решений'):
        print_tree(X_train, y_train, list(data.columns))

#st.write('Значения accuracy для отдельных фолдов')
#st.bar_chart(scores)
#st.write('Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores)))

# clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=cv_slider, scoring='roc_auc')
# clf_gs.fit(data_X, data_y)
# 
# st.subheader('Оценка качества модели')
# 
# st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))
# 
# # Изменение качества на тестовой выборке в зависимости от К-соседей
# fig1 = plt.figure(figsize=(7,5))
# ax = plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
# st.pyplot(fig1)
