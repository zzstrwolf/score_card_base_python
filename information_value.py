#coding:utf-8

import numpy as np
import pandas as pd
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import seaborn as sns

class WOE(object):
    #class method
    @staticmethod
    def regroup(df,column,split_points):
        for i in range(len(split_points)-1):
            df[column][(df[column]>=split_points[i]) & (df[column]<=split_points[i+1])] = '%s-%s' % (split_points[i],split_points[i+1])
        df[column] = df[column].astype(np.str_)
    
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20
        self.var_iv_df = pd.DataFrame()
        self.woe_dicts = {}
        self.X_bin = pd.DataFrame()
        self.X_woe = pd.DataFrame()
        #经过woe替代的数据集
    
    #find columns including missing values
    def find_na_column(self,df):
        miss_columns = []
        for column in df:
            if sum(pd.isnull(df[column])) > 0:
                miss_columns.append(column)
        return miss_columns
    
    #return columns which NA rate > 0
    def NA_rate(self,df):
        NA_df = (df.isnull().sum()/df.shape[0]).reset_index()
        NA_df.columns = ['variables','NA_rate']
        return NA_df[NA_df['NA_rate']>0].sort_values(by='NA_rate',ascending=False)
        
    def woe(self, X, y, event=1, category_cols = []):
        #are there any columns including missing values
        miss_columns = self.find_na_column(X)
        if len(miss_columns) > 0:
            raise Exception('there are some columns (%s) including missing values' % ', '.join(miss_columns))
            
        #检查y是否是二分类变量，如果不是，内部raise错误
        self.check_target_binary(y)
        self.X_bin = self.feature_discretion(X, category_cols)

        res_woe = []
        res_iv = []
        #我在外部组合了一个IV和variable name的dataframe，我现在在内部完成
        #同样，内部也可以组合一个woe的dict
        for i in range(0, self.X_bin.shape[-1]):
            x = self.X_bin.iloc[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        #var_iv_df构造
        self.var_iv_df = pd.DataFrame({'variable':X.columns,'IV':res_iv})
        self.var_iv_df.sort_values('IV',ascending = 0,inplace = True)
        #woe_dict构造
        self.woe_dicts = dict(zip(X.columns, res_woe))
        return self

    def woe_single_x(self, x, y, event=1):
        self.check_target_binary(y)
        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = x.unique()
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[x == x1]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                #woe1 = math.log(rate_event / rate_non_event)
                woe1 = math.log(rate_non_event / rate_event)
            woe_dict[x1] = (woe1,event_count+non_event_count,round(event_count*1.0/(event_count+non_event_count),3))
            #iv += (rate_event - rate_non_event) * woe1
            iv += (rate_non_event - rate_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res
        
    def woe_replace_(self):
        self.X_woe = self.X_bin.copy()
        for column in self.X_woe.columns:
            for k in self.woe_dicts[column].keys():
            #遍历这个column的woe_dict的每一个key
                self.X_woe[column][self.X_woe[column] == k] = self.woe_dicts[column][k][0]
        self.X_woe = self.X_woe.astype(np.float32)
        return self

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X, category_cols = []):
        temp_df = pd.DataFrame()
        for i in range(0, X.shape[-1]):
            x = X.iloc[:, i]
            #x_type = type_of_target(x)
            #不用他的方法来判断,用set的方式，值大于10个全部认为是连续变量
            if x.name in category_cols:
                temp_df.loc[:,x.name] = x
            elif len(set(x)) > 10:
                x1 = self.discrete(x)
                temp_df.loc[:,x1.name] = x1
            else:
                temp_df.loc[:,x.name] = x
        return temp_df

    def discrete(self, x, bin=5):
        #res = np.array([0] * x.shape[-1], dtype=int)
        #暂时修改成以前错误的引用赋值形式，为了得到WOE，那我实际应用的时候也要用<=的WOE方式
        x_copy = pd.Series.copy(x)
        x_copy = x_copy.astype(str)
        #x_copy = x_copy.astype(np.str_)
        #x_copy = x
        x_gt0 = x[x>=0]
        #if x.name == 'TD_PLTF_CNT_1M':
            #bin = 5
            #x_gt0 = x[(x>=0) & (x<=24)]
        
        for i in range(bin):
            point1 = stats.scoreatpercentile(x_gt0, i * (100.0/bin))
            point2 = stats.scoreatpercentile(x_gt0, (i + 1) * (100.0/bin))
            x1 = x[(x >= point1) & (x <= point2)]
            mask = np.in1d(x, x1)
            #x_copy[mask] = i + 1
            x_copy[mask] = '%s-%s' % (point1,point2)
            #x_copy[mask] = point1
            #print x_copy[mask]
            #print x
        #print x
        return x_copy
        
    def grade(self, x, bin=5):
        #res = np.array([0] * x.shape[-1], dtype=int)
        #暂时修改成以前错误的引用赋值形式，为了得到WOE，那我实际应用的时候也要用<=的WOE方式
        x_copy = np.copy(x)
        #x_copy = x_copy.astype(str)
        #x_copy = x_copy.astype(np.str_)
        #x_copy = x
        x_gt0 = x[x>=0]
        
        for i in range(bin):
            point1 = stats.scoreatpercentile(x_gt0, i * (100.0/bin))
            point2 = stats.scoreatpercentile(x_gt0, (i + 1) * (100.0/bin))
            x1 = x[(x >= point1) & (x <= point2)]
            mask = np.in1d(x, x1)
            #x_copy[mask] = i + 1
            x_copy[mask] = i + 1
            #x_copy[mask] = point1
            #print x_copy[mask]
            #print x
            print point1,point2
        #print x
        return x_copy

        
    def bin(self, x, points):
        x_copy = np.copy(x)
        for i in range(len(points)-1):
            x1 = x[np.where((x >= points[i]) & (x <= points[i+1]))]
            mask = np.in1d(x, x1)
            x_copy[mask] = i + 1
        return x_copy
    
    def sort_dict(self,item):
        if item[0].split('-')[0] == '':
            return float(item[0])
        else:
            return float(item[0].split('-')[0])
    
    #输出某个字段的woe值
    def print_woe(self,column):
        #print column,type(self.woe_dicts[column].items()[0][0])
        if type(self.woe_dicts[column].items()[0][0]) == str:
            for i in sorted(self.woe_dicts[column].items(), key = self.sort_dict):
                print i
        else:
            for i in sorted(self.woe_dicts[column].items(),key = lambda item:item[0]):
                print i
            
    def plot_br_chart(self,column):
        if type(self.woe_dicts[column].items()[0][0]) == str:
            woe_lists = sorted(self.woe_dicts[column].items(), key = self.sort_dict)
        else:
            woe_lists = sorted(self.woe_dicts[column].items(),key = lambda item:item[0])
        tick_label = [i[0] for i in woe_lists]
        counts = [i[1][1] for i in woe_lists]
        br_data = [i[1][2] for i in woe_lists]
        x = range(len(counts))
        fig, ax1 = plt.subplots(figsize=(12,8))
        my_palette = sns.color_palette(n_colors=100)
        sns.barplot(x,counts,ax=ax1,palette=sns.husl_palette(n_colors=20,l=.7))
        plt.xticks(x,tick_label,rotation = 30,fontsize=12)
        plt.title(column,fontsize=18)
        ax1.set_ylabel('count',fontsize=15)
        ax1.tick_params('y',labelsize=12)
        #ax1.bar(x,counts,tick_label = tick_label,color = 'y',align = 'center')
        #ax1.bar(x,counts,color = 'y',align = 'center')
        
        ax2 = ax1.twinx()
        ax2.plot(x,br_data,color='black')
        ax2.set_ylabel('bad rate',fontsize=15)
        ax2.tick_params('y',labelsize=12)
        plot_margin = 0.25
        x0, x1, y0, y1 = ax1.axis()
        ax1.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - 0,
              y1 * 1.1))
        plt.show()
        
    def save_br_chart(self, column, path):
        if type(self.woe_dicts[column].items()[0][0]) == str:
            woe_lists = sorted(self.woe_dicts[column].items(), key = self.sort_dict)
        else:
            woe_lists = sorted(self.woe_dicts[column].items(),key = lambda item:item[0])
        tick_label = [i[0] for i in woe_lists]
        counts = [i[1][1] for i in woe_lists]
        br_data = [i[1][2] for i in woe_lists]
        x = range(len(counts))
        fig, ax1 = plt.subplots(figsize=(12,8))
        my_palette = sns.color_palette(n_colors=100)
        sns.barplot(x,counts,ax=ax1,palette=sns.husl_palette(n_colors=20,l=.7))
        plt.xticks(x,tick_label,rotation = 30,fontsize=12)
        plt.title(column,fontsize=18)
        ax1.set_ylabel('count',fontsize=15)
        ax1.tick_params('y',labelsize=12)
        ax2 = ax1.twinx()
        ax2.plot(x,br_data,color='black')
        ax2.set_ylabel('bad rate',fontsize=15)
        ax2.tick_params('y',labelsize=12)
        plot_margin = 0.25
        x0, x1, y0, y1 = ax1.axis()
        ax1.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - 0,
              y1 * 1.1))
        plt.savefig(path)
        

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max
