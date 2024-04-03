import time

from data_preprocess.telecom_analyzer import analyzer
import datetime
import numpy as np
import optuna
import xgboost
import lightgbm
import catboost
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from lazypredict.Supervised import LazyRegressor
import pandas as pd
from sklearn.feature_selection import RFECV ,SelectFromModel,SequentialFeatureSelector
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_validate,learning_curve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'

class prevalue_model:
    def __init__(self,models,data_type):
        self.models = models
        self.data_type = data_type
        self.tele_analyzer = analyzer()
        self.data = self.load_data()

    def load_data(self):
        def get_value_level(value):
            if value < one_level:
                value_level = 0
            elif value >= one_level and value < two_level:
                value_level = 1
            elif value >= two_level:
                value_level = 2
            return value_level
        if self.data_type =='origin':
            data = self.tele_analyzer.data
        elif self.data_type == 'factor_load':
            data = self.tele_analyzer.factor_analyzer('get_data')
        one_level = data['user_values'].quantile(0.3)
        two_level = data['user_values'].quantile(0.8)
        data['value_level'] = data['user_values'].apply(lambda x: get_value_level(x))
        return data


    def init_model(self,model,all_run=None):
        target = self.data['user_values']
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        train_data = self.data[xgb_feas]
        scorer1 = make_scorer(mean_squared_error)
        scorer2 = make_scorer(r2_score)
        score_name1 = '均方根误差'
        score_name2 = 'R2'
        if all_run:
            model_name    = list()
            train_r2_list = list()
            test_r2_list  = list()
            train_rmse_list = list()
            test_rmse_list  = list()
            token_times_list = list()
            for k,v in self.models.items():
                print(k)
                model_name.append(k)
                model = v.get('model')
                start_t = time.time()
                print(f'{k} Start {datetime.datetime.now()}')
                scores1 = cross_validate(model, train_data, target, cv=5, n_jobs=-1, scoring=scorer1, return_train_score=True)
                scores2 = cross_validate(model, train_data, target, cv=5, n_jobs=-1, scoring=scorer2, return_train_score=True)
                train_rmse = round(np.mean(np.sqrt(scores1['train_score'])),4)
                test_rmse = round(np.mean(np.sqrt(scores1['test_score'])),4)
                train_r2  = round(np.mean(scores2['train_score']),4)
                test_r2  = round(np.mean(scores2['test_score']),4)
                train_r2_list.append(train_r2)
                test_r2_list.append(test_r2)
                train_rmse_list.append(train_rmse)
                test_rmse_list.append(test_rmse)
                print(
                    f"训练集{score_name1} {train_rmse}|测试集{score_name1} {test_rmse}")
                print(
                    f"训练集{score_name2} {train_r2}|测试集{score_name2} {test_r2}")
                end_t = time.time()
                print(f'{k} End {datetime.datetime.now()}')
                token_times = round(end_t - start_t,4)
                token_times_list.append(token_times)
                print()
            save_df = pd.DataFrame()
            save_df['Model'] = model_name
            save_df['train_R2'] = train_r2_list
            save_df['test_R2'] = test_r2_list
            save_df['train_RMSE'] = train_rmse_list
            save_df['test_RMSE'] = test_rmse_list
            save_df['Token_times'] = token_times_list
            save_df.to_csv('tmp/regressor_diff.csv',index=False)
            print('file save Succ!!!')


        else:
            print(datetime.datetime.now())
            scores1 =cross_validate(model,train_data,target,cv=5,n_jobs=-1,scoring=scorer1,return_train_score=True)
            scores2 =cross_validate(model,train_data,target,cv=5,n_jobs=-1,scoring=scorer2,return_train_score=True)
            print(f"训练集{score_name1} {np.mean(np.sqrt(scores1['train_score']))}|测试集{score_name1} {np.mean(np.sqrt(scores1['test_score']))}")
            print(f"训练集{score_name2} {np.mean(scores2['train_score'])}|测试集{score_name2} {np.mean(scores2['test_score'])}")
            print(datetime.datetime.now())

        # xgb.fit(train_data, target)
        # importance = xgb.feature_importances_
        # indices = np.argsort(importance)[::-1]
        # print("特征重要性排名:")
        # fealist = []
        # feascore = []
        # top_feas = list()
        # for t in range(len(train_data.columns)):
        #     top_feas.append(train_data.columns[indices[t]])
        #     print("%d. 特征名：%s，重要性得分：%f" % (t + 1, train_data.columns[indices[t]], importance[indices[t]]))
        #     fealist.append(train_data.columns[indices[t]])
        #     feascore.append(importance[indices[t]])

    def para_adjustment(self,model, score,n_trials):
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        train_data = self.data[xgb_feas]
        target = self.data['user_values']
        print(train_data.shape)
        print(target.shape)
        # if isinstance(model, xgboost.XGBRegressor):
        #     model_name = 'xgb'
        #     print(model_name)
        #     train = data[xgb_feas]
        # elif isinstance(model, catboost.CatBoostRegressor):
        #     model_name = 'cat'
        #     print(model_name)
        #     train = data[cat_feas]
        # elif isinstance(model, ExtraTreesRegressor):
        #     model_name = 'etr'
        #     print(model_name)
        #     train = data[etr_feas]

        # if isinstance(model,KNeighborsRegressor):
        #     train = MinMaxScaler().fit_transform(data[knn_feas])

        if score == 'mse':
            def custom_mse(y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred, squared=False)
                return mse
            scorer = make_scorer(custom_mse)
            score_direction = 'minimize'
        elif score == 'r2':
            scorer = make_scorer(r2_score)
            score_direction = 'maximize'

        def objective(trial):
            params = {
                # 'num_leaves': trial.suggest_int('num_leaves', 15,35),
                # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 12,22),

                # 'min_samples_split': trial.suggest_int('min_samples_split', 2,7),
                # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,7),

                'n_estimators': trial.suggest_int('n_estimators', 80, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight':trial.suggest_int('min_child_weight', 1, 8),
                # 'subsample':trial.suggest_float('subsample',0.9,1),
                # 'colsample_bytree':trial.suggest_float('colsample_bytree',0.8,1),
                # 'learning_rate': trial.suggest_float('learning_rate', 0.09, 0.1),

                # 'n_neighbors':trial.suggest_int('n_neighbors',5,20),
                # 'weights':trial.suggest_categorical('weights',['uniform','distance']),
                # 'metric':trial.suggest_categorical('metric',['euclidean','manhattan','chebyshev','minkowski','hamming']),
                # 'algorithm':trial.suggest_categorical('algorithm',['auto','brute','ball_tree'])

                # 'iterations':trial.suggest_int('iterations', 200, 600),
                # 'depth':trial.suggest_int('depth', 3, 10),
                # 'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.8,1),
                # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
                # 'subsample':trial.suggest_float('subsample', 0.8,1),
                # 'learning_rate': trial.suggest_float('learning_rate', 0.01,0.08),
            }
            model.set_params(**params)
            scores = cross_validate(model, train_data, target, cv=5, scoring=scorer, n_jobs=-1)  # cv表示交叉验证的折数
            return round(scores['test_score'].mean(), 4)

        study = optuna.create_study(direction=score_direction)
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_score = study.best_value
        print("Best parameters: ", best_params)
        print("Best score: ", best_score)


    def sfs(self, cv):
        data = self.data
        xgb = xgboost.XGBRegressor(random_state=30)
        target = data['user_values']
        del_cols = ['user_values']
        train_data = data.drop(columns=del_cols)
        print(f'Start{datetime.datetime.now()}')
        sfs = SequentialFeatureSelector(estimator=xgb, n_features_to_select=25,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)  # backward
        sfs.fit(train_data, target)
        print(f'End{datetime.datetime.now()}')
        # 获取选择的特征的索引
        selected_features_idx = sfs.get_support()
        sele_feas = train_data.columns[selected_features_idx]
        print(sele_feas)

        a = 'neg_mean_squared_error'
        b = 'r2'
        scorer = make_scorer(mean_squared_error)
        cv_scores =cross_validate(xgb,train_data,target,cv=cv,n_jobs=-1,scoring=scorer,return_train_score=True)
        print(
            f"训练集 {np.mean(np.sqrt(cv_scores['train_score']))}|测试集 {np.mean(np.sqrt(cv_scores['test_score']))}")

    def learning_curve_show(self,model,score):
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        train_data = self.data[xgb_feas]
        target = self.data['user_values']

        if score == 'mse':
            def custom_mse(y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred, squared=False)
                return mse
            scorer = make_scorer(custom_mse)
            title_type = 'RMSE'
        elif score == 'r2':
            scorer = make_scorer(r2_score)
            title_type = 'R2'
        train_sizes, train_scores, test_scores = learning_curve(model, train_data, target,
                                                                cv=5,scoring=scorer)
        print(f'end...{datetime.datetime.now()}')

        # 将均方误差取负值，因为 learning_curve 使用负的评分
        train_scores = train_scores
        test_scores = test_scores

        # 计算平均值和标准差
        train_mean = np.round(np.mean(train_scores, axis=1), 6)
        train_std = np.round(np.std(train_scores, axis=1), 6)
        test_mean = np.round(np.mean(test_scores, axis=1), 6)
        test_std = np.round(np.std(test_scores, axis=1), 6)

        # 绘制学习曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='orange', marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.1)
        for i, j in zip(train_sizes, train_mean):
            plt.text(i, j, str(j), ha='center', va='bottom')
        for i, j in zip(train_sizes, test_mean):
            plt.text(i, j, str(j), ha='center', va='bottom')
        # 设置图表属性
        plt.title(f'Learning Curve for Regression Model ({title_type})')
        plt.xlabel('Number of Training Examples')
        plt.ylabel(title_type)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def lazy_re(self):
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        train_data = self.data[xgb_feas]
        target = self.data['user_values']
        train = MinMaxScaler().fit_transform(train_data)
        X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=30, test_size=0.2)
        LazyR = LazyRegressor()
        models, predictions = LazyR.fit(X_train, X_test, y_train, y_test)
        result_df = pd.concat([models, predictions], axis=1)
        result_df.to_csv('tmp/lazy_regressor_results.csv')

    def bagging_models(self):
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        data = self.data[xgb_feas]
        target = self.data['user_values']
        cv = KFold(n_splits=5, shuffle=True, random_state=30)
        for train_index, test_index in cv.split(data):
            train_data = data[train_index]
            test_data = data[test_index]
            train_tar = target[train_index]
            test_tar = target[test_index]


    def run(self):
        self.init_model(model = self.models.get('knn')['model'],all_run=True)
        self.bagging_models()
        # self.sfs(cv=5)
        # self.para_adjustment(score='mse',n_trials=150)
        # self.lazy_re()
        # self.learning_curve_show(model = self.models.get('xgb')['model'],score='mse')


def init_models():
    model_dic = dict()
    number = 3
    xgb = xgboost.XGBRegressor()
    etr = ExtraTreesRegressor()
    lgb = lightgbm.LGBMRegressor()
    rfr = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    knn = KNeighborsRegressor()
    lr  = LinearRegression()

    xgb_params = {'n_estimators': 131, 'max_depth': 8, 'min_child_weight': 8,'random_state': 30}
    etr_params = {'random_state':30}
    lgb_params = {'random_state':30}
    rfr_params = {'random_state':30}
    gbr_params = {'random_state':30}

    xgb.set_params(**xgb_params)
    etr.set_params(**etr_params)
    lgb.set_params(**lgb_params)
    rfr.set_params(**rfr_params)
    gbr.set_params(**gbr_params)

    model_dic['xgb'] =dict()
    model_dic['xgb']['number']=number
    model_dic['xgb']['model'] = xgb
    model_dic['etr'] =dict()
    model_dic['etr']['number']=number
    model_dic['etr']['model'] = etr
    model_dic['lgb'] = dict()
    model_dic['lgb']['number'] = number
    model_dic['lgb']['model'] = lgb
    model_dic['rfr'] = dict()
    model_dic['rfr']['number'] = number
    model_dic['rfr']['model'] = rfr
    model_dic['gbr'] = dict()
    model_dic['gbr']['number'] = number
    model_dic['gbr']['model'] = gbr
    model_dic['knn'] = dict()
    model_dic['knn']['number'] = number
    model_dic['knn']['model'] = knn
    model_dic['lr'] = dict()
    model_dic['lr']['number'] = number
    model_dic['lr']['model'] = lr
    return model_dic


if __name__ == '__main__':
    models = init_models()
    data_type = 'factor_load'  # origin factor_load:载入因子得分
    pre_model = prevalue_model(models =models, data_type=data_type)
    pre_model.run()
