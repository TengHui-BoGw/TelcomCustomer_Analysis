import json
import random
import time
from data_preprocess.telecom_analyzer import analyzer
import datetime
import numpy as np
import optuna
import xgboost
import lightgbm
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
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
    def __init__(self,data_type,verbosity):
        self.verbosity = verbosity
        self.models = self.init_models()
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

    def init_models(self):
        model_dic = dict()
        xgb = xgboost.XGBRegressor()
        lgb = lightgbm.LGBMRegressor()
        gbr = GradientBoostingRegressor()
        bgr = BaggingRegressor()

        xgb_params = {'n_estimators': 131, 'max_depth': 8, 'min_child_weight': 8,
                      'colsample_bytree': 0.9401840831296924, 'learning_rate': 0.10317074040093721, 'random_state': 30}
        lgb_params = {'random_state': 30, 'verbosity': -1, 'num_leaves': 49, 'min_child_samples': 20,
                      'n_estimators': 389, 'max_depth': 13}
        gbr_params = {'random_state': 30, 'n_estimators': 288, 'max_depth': 10}
        bgr_params = {'random_state': 30,'n_estimators': 97, 'max_samples': 0.9937681661021245, 'max_features': 0.9498173759873925}

        xgb.set_params(**xgb_params)
        lgb.set_params(**lgb_params)
        gbr.set_params(**gbr_params)
        bgr.set_params(**bgr_params)

        model_dic['xgb'] = dict()
        model_dic['xgb']['model'] = xgb
        model_dic['lgb'] = dict()
        model_dic['lgb']['model'] = lgb
        model_dic['bgr'] = dict()
        model_dic['bgr']['model'] = bgr
        model_dic['gbr'] = dict()
        model_dic['gbr']['model'] = gbr
        return model_dic

    def run_model(self,model,all_run=None):
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
        kf = KFold(n_splits=5, shuffle=True, random_state=30)
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
            scores1 =cross_validate(model,train_data,target,cv=kf,n_jobs=-1,scoring=scorer1,return_train_score=True)
            scores2 =cross_validate(model,train_data,target,cv=kf,n_jobs=-1,scoring=scorer2,return_train_score=True)
            print(f"训练集{score_name1} {np.mean(np.sqrt(scores1['train_score']))}|测试集{score_name1} {np.mean(np.sqrt(scores1['test_score']))}")
            print(f"训练集{score_name2} {np.mean(scores2['train_score'])}|测试集{score_name2} {np.mean(scores2['test_score'])}")
            print(datetime.datetime.now())
            f
            model.fit(train_data, target)
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            print("特征重要性排名:")
            fealist = []
            feascore = []
            top_feas = list()
            for t in range(len(train_data.columns)):
                top_feas.append(train_data.columns[indices[t]])
                print("%d. 特征名：%s，重要性得分：%f" % (t + 1, train_data.columns[indices[t]], importance[indices[t]]))
                fealist.append(train_data.columns[indices[t]])
                feascore.append(importance[indices[t]])

    def para_adjustment(self,model,score,n_trials):
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        weights_dic = dict()
        weights_dic['xgb'] = 0.2
        weights_dic['lgb'] = 0.2
        weights_dic['bgr'] = 0.25
        weights_dic['gbr'] = 0.35

        train_data = self.data[xgb_feas]
        target = self.data['user_values']
        print(train_data.shape)
        print(target.shape)
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
                # lgb
                # 'num_leaves': trial.suggest_int('num_leaves', 20,50),
                # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 22),
                # 'n_estimators': trial.suggest_int('n_estimators', 80, 400),
                # 'max_depth': trial.suggest_int('max_depth', 3, 15),

                # xgb
                # 'n_estimators': trial.suggest_int('n_estimators', 80, 200),
                # 'max_depth': trial.suggest_int('max_depth', 3, 8),
                # 'min_child_weight':trial.suggest_int('min_child_weight', 1, 8),
                # 'colsample_bytree':trial.suggest_float('colsample_bytree',0.8,1),
                # 'learning_rate': trial.suggest_float('learning_rate', 0.09,0.11),

                # gbr
                # 'n_estimators': trial.suggest_int('n_estimators', 80, 300),
                # 'max_depth': trial.suggest_int('max_depth', 3, 10),
                # 'min_samples_split':trial.suggest_int('min_samples_split', 1, 6),

                # bgr
                # 'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                # 'max_samples': trial.suggest_float('max_samples', 0.8, 1),
                # 'max_features': trial.suggest_float('max_features', 0.75, 1),

                # bagging
            }
            min_samples = trial.suggest_float('min_samples', 0.95, 1),
            min_features = trial.suggest_float('min_features', 0.93, 1),
            xgb_num = trial.suggest_int('xgb_num', 1, 5),
            lgb_num = trial.suggest_int('lgb_num', 1, 5),
            gbr_num = trial.suggest_int('gbr_num', 1, 5),
            bgr_num = trial.suggest_int('bgr_num', 1, 5),
            model.set_params(**params)
            rmse = self.bagging_models(min_samples=min_samples, min_features=min_features, xgb_num=xgb_num,
                                lgb_num=lgb_num, gbr_num=gbr_num, bgr_num=bgr_num, weights=weights_dic)
            # scores = cross_validate(model, train_data, target, cv=5, scoring=scorer, n_jobs=-1)  # cv表示交叉验证的折数
            # return round(scores['test_score'].mean(), 4)
            return rmse

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

    def bagging_models(self,min_samples,min_features,xgb_num,lgb_num,bgr_num,gbr_num,weights=None):
        if isinstance(min_samples,tuple):
            min_samples = min_samples[0]
        if isinstance(min_features,tuple):
            min_features = min_features[0]
        if isinstance(xgb_num,tuple):
            xgb_num = xgb_num[0]
        if isinstance(lgb_num,tuple):
            lgb_num = lgb_num[0]
        if isinstance(bgr_num,tuple):
            bgr_num = bgr_num[0]
        if isinstance(gbr_num,tuple):
            gbr_num = gbr_num[0]
        if weights is not None and len(list(self.models.keys()))!=len(list(weights.keys())):
            raise Exception('weights and models donot match!!!')
        xgb_feas = ['region', 'totalemployed_months', 'activeusers_family', 'credit_rating',
                    'phonenetwork', 'newphoneuser', 'phone_usedays', 'phoneprice',
                    'useminutes', 'over_useminutes', 'over_cost', 'overdata_cost',
                    'roaming_callcounts', 'useminutes_percentchange_before_threemonth',
                    'cost_percentchange_before_threemonth', 'complete_usedata_counts',
                    'customerservice_callcounts', 'customerservice_useminutes',
                    'inAndout_callcounts_PVC', 'incomplete_minutes_PVC', 'callcounts_NPVC',
                    'forward_callcounts', 'wait_callcounts', 'user_spend_limit',
                    'value_level']
        need_feas = xgb_feas + ['user_values']
        data = self.data[need_feas]
        scores_dic = dict() #存储 训练过程中 基学习器的训练效果
        for k1,v1 in self.models.items():
            scores_dic[k1] = dict()
            if k1=='xgb':
                v1['number'] = xgb_num
            if k1=='lgb':
                v1['number'] = lgb_num
            if k1=='bgr':
                v1['number'] = bgr_num
            if k1=='gbr':
                v1['number'] = gbr_num

            for i in range(5):
                i = i + 1
                scores_dic[k1][i] = dict()
                scores_dic[k1][i]['train_r2'] = list()
                scores_dic[k1][i]['train_rmse'] = list()
                scores_dic[k1][i]['test_r2'] = list()
                scores_dic[k1][i]['test_rmse'] = list()

        cv = KFold(n_splits=5, shuffle=True, random_state=30)
        all_test_r2 = list()
        all_test_rmse = list()
        all_train_r2 = list()
        all_train_rmse = list()
        cnt = 1
        for train_index, test_index in cv.split(data):
            print(f'第{cnt}轮开始{datetime.datetime.now()}')
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            test_models_pre = dict() # 存储验证集的预测结果
            train_models_pre = dict()# 存储训练集的预测结果
            for p1, u1 in self.models.items():
                test_models_pre[p1] = list()
                train_models_pre[p1] = list()
            for k2,v2 in self.models.items():
                for t in range(v2.get('number')):
                    n_iter = t + 1
                    model = v2.get('model')
                    # print(f'开始训练{k2}{n_iter}模型{datetime.datetime.now()}')
                    if n_iter== 1:
                        train_samples_num = train_data.shape[0] * 1
                        train_feas_num = round(len(xgb_feas) * 1)
                    else:
                        train_samples_num = train_data.shape[0] * min_samples
                        train_feas_num = round(len(xgb_feas) * min_features)
                    seed = 18 + n_iter * 5
                    random.seed(seed)
                    feas = random.sample(xgb_feas, random.choice(range(train_feas_num,len(xgb_feas)+1)))
                    sample_data = train_data.sample(random.choice(range(int(train_samples_num),train_data.shape[0]+1)),random_state=seed)
                    pre_train = train_data[feas]
                    train = sample_data[feas]
                    train_target = sample_data['user_values']
                    test = test_data[feas]
                    test_target = test_data['user_values']
                    model.fit(train,train_target)

                    # 训练集和验证机进行预测
                    test_y_pred = model.predict(test)
                    train_y_pred = model.predict(train)
                    all_train_pred = model.predict(pre_train)

                    test_models_pre[k2].append(list(test_y_pred))
                    train_models_pre[k2].append(list(all_train_pred))

                    test_r2 = r2_score(test_target, test_y_pred)
                    test_rmse = np.sqrt(mean_squared_error(test_target, test_y_pred))
                    train_r2 = r2_score(train_target, train_y_pred)
                    train_rmse = np.sqrt(mean_squared_error(train_target, train_y_pred))
                    scores_dic[k2][cnt]['train_r2'].append(round(train_r2,4))
                    scores_dic[k2][cnt]['train_rmse'].append(round(train_rmse,4))
                    scores_dic[k2][cnt]['test_r2'].append(round(test_r2,4))
                    scores_dic[k2][cnt]['test_rmse'].append(round(test_rmse,4))
                    # print(f"===========================train-r2:{round(train_r2,4)} train-rmse:{round(train_rmse,4)}")
                    # print(f"===========================test-r2:{round(test_r2,4)} test-rmse:{round(test_rmse,4)}")
                    # print(f'结束训练{k2}{n_iter}模型{datetime.datetime.now()}')
                    # print()

            test_all = list()
            train_all = list()
            for p2,u2 in test_models_pre.items():
                if weights is not None:
                    weight = weights[p2]
                else:
                    weight = 1/len(list(self.models.keys()))
                tmp_arrays = np.array(u2)
                test_mean = np.mean(tmp_arrays,axis=0)
                test_r2_ = r2_score(test_mean, test_data['user_values'])
                test_rmse_ = np.sqrt(mean_squared_error(test_mean, test_data['user_values']))
                if self.verbosity:
                    print(f'=================={p2} test r2:{test_r2_}')
                    print(f'=================={p2} test rmse:{test_rmse_}')
                test_all.append(test_mean * weight)
            print()
            for p3, u3 in train_models_pre.items():
                if weights is not None:
                    weight = weights[p3]
                else:
                    weight = 1/len(list(self.models.keys()))
                tmp_arrays = np.array(u3)
                train_mean = np.mean(tmp_arrays, axis=0)
                train_r2_ = r2_score(train_mean, train_data['user_values'])
                train_rmse_ = np.sqrt(mean_squared_error(train_mean, train_data['user_values']))
                if self.verbosity:
                    print(f'=================={p3} train r2:{train_r2_}')
                    print(f'=================={p3} train rmse:{train_rmse_}')
                train_all.append(train_mean * weight)

            tmp_arrays1 = np.array(test_all)
            tmp_arrays2 = np.array(train_all)
            test_pre_tmp = np.sum(tmp_arrays1, axis=0)
            train_pre_tmp = np.sum(tmp_arrays2, axis=0)
            test_r2_ = r2_score(test_pre_tmp, test_data['user_values'])
            test_rmse_ = np.sqrt(mean_squared_error(test_pre_tmp, test_data['user_values']))
            train_r2_ = r2_score(train_pre_tmp, train_data['user_values'])
            train_rmse_ = np.sqrt(mean_squared_error(train_pre_tmp, train_data['user_values']))
            all_test_r2.append(test_r2_)
            all_test_rmse.append(test_rmse_)
            all_train_r2.append(train_r2_)
            all_train_rmse.append(train_rmse_)
            print(f'================== ALLtest r2:{test_r2_}')
            print(f'================== ALLtest rmse:{test_rmse_}')
            print(f'================== ALLtrain r2:{train_r2_}')
            print(f'================== ALLtrain rmse:{train_rmse_}')
            print(f'第{cnt}轮结束{datetime.datetime.now()}')
            print()
            cnt = cnt + 1
        print(f'训练集 R2:{np.mean(all_train_r2)},RMSE:{np.mean(all_train_rmse)}')
        print(f'验证集 R2:{np.mean(all_test_r2)},RMSE:{np.mean(all_test_rmse)}')

        # 存储 训练过程中 基学习器的训练效果
        train_r2_scores = list()
        train_rmse_scores = list()
        test_r2_scores = list()
        test_rmse_scores = list()
        for k3,v3 in scores_dic.items():
            for k4,v4 in v3.items():
                train_r2_mean = np.mean(v4['train_r2'])
                train_rmse_mean = np.mean(v4['train_rmse'])
                test_r2_mean = np.mean(v4['test_r2'])
                test_rmse_mean = np.mean(v4['test_rmse'])
                train_r2_scores.append(train_r2_mean)
                train_rmse_scores.append(train_rmse_mean)
                test_r2_scores.append(test_r2_mean)
                test_rmse_scores.append(test_rmse_mean)
        with open('res/res.json', 'w') as json_file:
            json.dump(scores_dic, json_file,indent=4, ensure_ascii=False)
        print('保存文件成功!!! 查看目录res/res.json 查看模型训练详情')
        return np.mean(all_test_rmse)


    def run(self):
        # self.run_model(model = self.models.get('xgb')['model'],all_run=False)

        # weights_dic = dict()
        # weights_dic['xgb'] = 0.2
        # weights_dic['lgb'] = 0.2
        # weights_dic['bgr'] = 0.25
        # weights_dic['gbr'] = 0.35
        # self.bagging_models(min_samples=0.95,min_features=1,xgb_num=3,
        #                     lgb_num=3,gbr_num=3,bgr_num=3,weights=weights_dic)

        # self.sfs(cv=5)
        self.para_adjustment(model=self.models.get('bgr')['model'],score='mse',n_trials=10)
        # self.lazy_re()
        # self.learning_curve_show(model = self.models.get('xgb')['model'],score='mse')



if __name__ == '__main__':
    data_type = 'factor_load'  # origin factor_load: 载入因子得分
    pre_model = prevalue_model(data_type=data_type,verbosity=False)
    pre_model.run()
