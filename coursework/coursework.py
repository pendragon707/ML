import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

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
        array_labels, array_metric = self.get_data_for_metric(metric,     ascending)
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
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('weatherAUS.csv', parse_dates=['Date'])
    return data


@st.cache
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data = data_in.copy()
    data = data.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis = 1)
    data = data.dropna(axis=0, how='any')
    data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x== 'Yes' else 0)
    data = data.drop(['Date', 'WindDir3pm','WindDir9am', 'WindGustDir', 'Location'], axis = 1)
    min_max_scaler = MinMaxScaler()
    data[:] = min_max_scaler.fit_transform(data)
    temp_y = data['RainTomorrow']
    temp_X = data[['Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'RainToday']]

    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, train_size=0.005, random_state=1)
    return data[:], X_train, X_test, y_train, y_test

# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'ET', 'GB']
clas_models = {'LogR': LogisticRegression(),
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'ET':ExtraTreesClassifier(),
               'GB':GradientBoostingClassifier()}

@st.cache(suppress_st_warning=True)
def print_models(models_select, X_train, X_test, y_train, y_test, clasMetricLogger):
    current_models_list = []
    roc_auc_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:,1]

        precision = precision_score(y_test.values, Y_pred)
        recall = recall_score(y_test.values, Y_pred)
        f1 = f1_score(y_test.values, Y_pred)
        roc_auc = roc_auc_score(y_test.values, Y_pred_proba)

        clasMetricLogger.add('precision', model_name, precision)
        clasMetricLogger.add('recall', model_name, recall)
        clasMetricLogger.add('f1', model_name, f1)
        clasMetricLogger.add('roc_auc', model_name, roc_auc)

        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        #Отрисовка ROC-кривых 
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
        plot_confusion_matrix(model, X_test, y_test.values, ax=ax[1],
                        display_labels=['0','1'],
                        cmap=plt.cm.Blues, normalize='true')
        fig.suptitle(model_name)
        st.pyplot(fig)

    if len(roc_auc_list)>0:
        temp_d = {'roc-auc': roc_auc_list}
        temp_df = pd.DataFrame(data=temp_d, index=current_models_list)
        st.bar_chart(temp_df)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

data = load_data()

if st.checkbox('Показать пропуски в данных'):
    fig1, ax = plt.subplots(figsize=(10,5))
    cols = data.columns
    colours = ['#235AB5', '#E8B41E']
    sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))
    st.pyplot(fig1)
    st.write('Желтый - пропущенные данные')

data, X_train, X_test, y_train, y_test = preprocess_data(data)

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

st.header('Оценка качества моделей')
metrics = MetricLogger()
print_models(models_select, X_train, X_test, y_train, y_test, metrics)

clas_metrics = metrics.df['metric'].unique()
for metric in clas_metrics:
    metrics.plot('Метрика: ' + metric, metric, figsize=(7, 6))
