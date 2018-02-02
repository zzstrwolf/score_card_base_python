#coding:utf-8

import numpy as np
import pandas as pd
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from IPython.display import display,HTML
#MySQL shell style table dependences
#from prettytable import PrettyTable
#from prettytable import MSWORD_FRIENDLY

class scorecard(object):
    @staticmethod
    def regroup(df,column,split_points):
        for i in range(len(split_points)-1):
            df[column][(df[column]>=split_points[i]) & (df[column]<=split_points[i+1])] = '%s-%s' % (split_points[i],split_points[i+1])
        df[column] = df[column].astype(np.str_)
        
    def __init__(self,woe_min = -2,woe_max = 2):
        self._WOE_MIN = woe_min
        self._WOE_MAX = woe_max
        self.var_iv_df = pd.DataFrame()
        self.woe_hash = {}
        self.X_bin = pd.DataFrame()
        self.X_woe = pd.DataFrame()
        
    def check_variables(self,X):
        #Are there any columns including missing values
        miss_columns = self.find_NA_column(X)
        if len(miss_columns) > 0:
            raise Exception('there are some columns (%s) including missing values' % ', '.join(miss_columns))
    
    #Find columns including missing values
    def find_NA_column(self, df):
        miss_columns = []
        for column in df:
            if sum(pd.isnull(df[column])) > 0:
                miss_columns.append(column)
        return miss_columns
    
    #return columns which NA rate > 0
    def NA_rate(self, df):
        NA_df = (df.isnull().sum()/df.shape[0]).reset_index()
        NA_df.columns = ['variables','NA_rate']
        return NA_df[NA_df['NA_rate']>0].sort_values(by='NA_rate',ascending=False)
        
    def check_target(self, y):
        y_type = type_of_target(y)
        if y_type != 'binary':
            raise ValueError('Label type must be binary')
            
    def discrete(self, x, bin=5,miss_value=-2):
        x_copy = pd.Series.copy(x)
        x_copy = x_copy.astype(str)
        x_gt0 = x[x!=miss_value]
        #x_gt0 = x[x>=0]
        for i in range(bin):
            point1 = stats.scoreatpercentile(x_gt0, i * (100.0/bin))
            point2 = stats.scoreatpercentile(x_gt0, (i + 1) * (100.0/bin))
            mask = (x >= point1) & (x <= point2)
            x_copy[mask] = '%s--%s' % (point1,point2)
        mask = x==miss_value
        x_copy[mask] = '%s--%s' % (miss_value,miss_value)
        return x_copy
            
    def feature_discrete(self, X, category_cols = [], discrete_bins = {}):
        temp_df = pd.DataFrame()
        for i in range(0, X.shape[-1]):
            x = X.iloc[:, i]
            if x.name in category_cols:
                temp_df.loc[:,x.name] = x
            elif len(set(x)) > 10:
                if x.name in discrete_bins.keys():
                    bin = discrete_bins[x.name]
                else:
                    bin = 5
                x_discreted = self.discrete(x,bin)
                temp_df.loc[:,x.name] = x_discreted
            else:
                temp_df.loc[:,x.name] = x
        return temp_df
        
    def count_binary(self, y, event = 1):
        event_count = (y == event).sum()
        non_event_count = y.shape[-1] - event_count
        return event_count, non_event_count

    def woe_solo(self, x, y, event  = 1):
        event_total, non_event_total = self.count_binary(y, event = event)
        x_unique = x.unique()
        woe_dict = {}
        iv = 0
        for value in x_unique:
            y_group = y[x == value]
            event_count, non_event_count = self.count_binary(y_group, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe_group = self._WOE_MAX
            elif rate_non_event == 0:
                woe_group = self._WOE_MIN
            else:
                woe_group = round(math.log(rate_non_event / rate_event),4)
            woe_dict[value] = (woe_group,event_count+non_event_count,round(event_count*1.0/(event_count+non_event_count),3))
            iv += (rate_non_event - rate_event) * woe_group
        return woe_dict, iv
        
    def woe(self, X, y, event = 1, category_cols = [], discrete_bins = {}):
        #check type of X
        if isinstance(X,pd.DataFrame):
            pass
        elif isinstance(X,pd.Series):
            X = pd.DataFrame(X)
        else:
            raise Exception('Require X to be DataFrame or Series')
        #check X
        self.check_variables(X)
        #check Y
        self.check_target(y)
        
        self.X_bin = self.feature_discrete(X, category_cols, discrete_bins)
        
        all_woe = []
        all_iv = []
        
        for i in range(0, self.X_bin.shape[-1]):
            x = self.X_bin.iloc[:, i]
            woe_dict, iv = self.woe_solo(x, y, event)
            all_woe.append(woe_dict)
            all_iv.append(iv)
        
        self.var_iv_df = pd.DataFrame({'variable':X.columns,'IV':all_iv})
        self.var_iv_df.sort_values('IV',ascending = 0,inplace = True)
        self.woe_hash = dict(zip(X.columns, all_woe))
        return self
        
    def print_var_iv(self, min_iv=0.03):
        print self.var_iv_df[self.var_iv_df['IV'] >= min_iv]
        
    def print_single_iv(self, column):
        print '%s IV : %.4f' % (column,self.var_iv_df[self.var_iv_df['variable'] == column].iloc[0,0])
        
    def woe_replace(self, var_list = None):
        if var_list is None:
            var_list = self.X_bin.columns
        self.X_woe = self.X_bin.loc[:,var_list].copy()
        for column in var_list:
            for k in self.woe_hash[column].keys():
                self.X_woe[column][self.X_woe[column] == k] = self.woe_hash[column][k][0]
        self.X_woe = self.X_woe.astype(np.float32)
        return self
    
    #build dataset of woe for training model
    def woe_dataset(self,var_list):
        return pd.concat([self.X_bin.loc[:,var_list], self.y], axis=1)
        
    def grade(self, x, bin=5):
        x_copy = pd.Series.copy(x)
        #x_gt0 = x[x>=0]
        x_gt0 = x
        for i in range(bin):
            point1 = stats.scoreatpercentile(x_gt0, i * (100.0/bin))
            point2 = stats.scoreatpercentile(x_gt0, (i + 1) * (100.0/bin))
            mask = (x >= point1) & (x <= point2)
            x_copy[mask] = i + 1
            print point1,point2
        return x_copy
    
    def sort_dict(self,item):
        if item[0].split('--')[0] == '':
            return float(item[0])
        else:
            return float(item[0].split('--')[0])
    
    def print_woe(self,column):
        if type(self.woe_hash[column].items()[0][0]) == str:
            for i in sorted(self.woe_hash[column].items(), key = lambda item:item[0]):
                print i
        else:
            for i in sorted(self.woe_hash[column].items(), key = lambda item:item[0]):
                print i
                
    def print_woe_table(self,column):
        self.print_single_iv(column)
        bin_woe_df = pd.DataFrame(columns=['bins','woe','volume','bad rate'])
        if type(self.woe_hash[column].items()[0][0]) == str:
            for i in sorted(self.woe_hash[column].items(), key = self.sort_dict):
                bin_woe_df.loc[bin_woe_df.shape[0]] = {'bins':i[0],'woe':i[1][0],'volume':i[1][1],'bad rate':i[1][2]}
        else:
            for i in sorted(self.woe_hash[column].items(),key = lambda item:item[0]):
                bin_woe_df.loc[bin_woe_df.shape[0]] = {'bins':i[0],'woe':i[1][0],'volume':i[1][1],'bad rate':i[1][2]}
        styler = bin_woe_df.style
        styler.set_table_styles([
                               {'selector': '.row_heading',
                                'props': [('display', 'none')]},
                               {'selector': '.col_heading',
                                'props': [('text-align', 'center')]},
                               {'selector': '.blank.level0',
                                'props': [('display', 'none')]},         
                               {'selector': 'td:nth-child(2)',
                                'props': [('text-align', 'center')]}, 
                               {'selector': 'td:nth-child(3)',
                                'props': [('text-align', 'right')]},
                               {'selector': 'td:nth-child(4)',
                                'props': [('text-align', 'center')]},
                               {'selector': 'td:nth-child(5)',
                                'props': [('text-align', 'left')]}
                               ])
        styler = styler.format({'woe':'{:.4f}','bad rate':'{:.4f}'})
        display(styler)
                
    def print_woe_table1(self,column):
        bin_woe_df = pd.DataFrame(columns=['bins','woe','volume','bad rate'])
        if type(self.woe_hash[column].items()[0][0]) == str:
            for i in sorted(self.woe_hash[column].items(), key = self.sort_dict):
                bin_woe_df.loc[bin_woe_df.shape[0]] = {'bins':i[0],'woe':i[1][0],'volume':i[1][1],'bad rate':i[1][2]}
        else:
            for i in sorted(self.woe_hash[column].items(),key = lambda item:item[0]):
                bin_woe_df.loc[bin_woe_df.shape[0]] = {'bins':i[0],'woe':i[1][0],'volume':i[1][1],'bad rate':i[1][2]}
        print bin_woe_df['woe'].dtype
        display(bin_woe_df)
    
    #MySQL shell style table. Require module "prettytable"
    def print_woe_table2(self,column):
        self.print_single_iv(column)
        bin_woe_tabel = PrettyTable(["bins", "woe", "volume", "bad rate"])
        bin_woe_tabel.align["volume"] = "l"
        bin_woe_tabel.align["bad rate"] = "l"
        
        if type(self.woe_hash[column].items()[0][0]) == str:
            for i in sorted(self.woe_hash[column].items(), key = self.sort_dict):
                bin_woe_tabel.add_row([i[0],i[1][0],i[1][1],i[1][2]])
        else:
            for i in sorted(self.woe_hash[column].items(),key = lambda item:item[0]):
                bin_woe_tabel.add_row([i[0],i[1][0],i[1][1],i[1][2]])
        print bin_woe_tabel
            
    def plot_br_chart(self,column):
        if type(self.woe_hash[column].items()[0][0]) == str:
            woe_lists = sorted(self.woe_hash[column].items(), key = self.sort_dict)
        else:
            woe_lists = sorted(self.woe_hash[column].items(),key = lambda item:item[0])
        sns.set_style(rc={"axes.facecolor": "#EAEAF2",
                "axes.edgecolor": "#EAEAF2",
               "axes.linewidth": 1,
                "grid.color": "white",})
        #sns.set_style('dark')
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
        ax1.tick_params('y',direction='in',length=6, width=0.5, labelsize=12)
        
        ax2 = ax1.twinx()
        ax2.plot(x,br_data,color='black')
        ax2.set_ylabel('bad rate',fontsize=15)
        ax2.tick_params('y',direction='in',length=6, width=0.5, labelsize=12)
        plot_margin = 0.25
        x0, x1, y0, y1 = ax1.axis()
        ax1.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - 0,
              y1 * 1.1))
              
        ax1.yaxis.grid(False)
        ax2.yaxis.grid(False)
        plt.show()
        
    def save_br_chart(self, column, path):
        if type(self.woe_hash[column].items()[0][0]) == str:
            woe_lists = sorted(self.woe_hash[column].items(), key = self.sort_dict)
        else:
            woe_lists = sorted(self.woe_hash[column].items(),key = lambda item:item[0])
        sns.set_style(rc={"axes.facecolor": "#EAEAF2",
                "axes.edgecolor": "#EAEAF2",
                "axes.linewidth": 1,
                "grid.color": "white",})
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
        ax1.tick_params('y',direction='in',length=6, width=0.5, labelsize=12)
        
        ax2 = ax1.twinx()
        ax2.plot(x,br_data,color='black')
        ax2.set_ylabel('bad rate',fontsize=15)
        ax2.tick_params('y',direction='in',length=6, width=0.5, labelsize=12)
        plot_margin = 0.25
        x0, x1, y0, y1 = ax1.axis()
        ax1.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - 0,
              y1 * 1.1))
        ax1.yaxis.grid(False)
        ax2.yaxis.grid(False)
        plt.savefig(path)
        
    def woe_br_chart(self,column):
        self.print_woe_table(column)
        self.plot_br_chart(column)
        
    def woe_br_chart_bat(self, min_iv = 0.03):
        for i in self.var_iv_df.variable[self.var_iv_df.IV >= min_iv]:
            self.woe_br_chart(i)
    
    def encode_test(self, X_test):
        X_test_c = X_test.loc[:,self.X_woe.columns.tolist()].copy()
        woe_hash_copy = deepcopy(self.woe_hash)
        for column in self.X_woe.columns:
            if type(woe_hash_copy[column].keys()[0]) == str:
                for k in woe_hash_copy[column].keys():
                    if k.split('-')[0] == '':
                        X_test_c[column][X_test[column] == float(k)] = woe_hash_copy[column][k][0]
                        woe_hash_copy[column].pop(k)
                for i,item in enumerate(sorted(woe_hash_copy[column].items(), key = self.sort_dict)):
                    if i == 0:
                        X_test_c[column][(float(item[0].split('-')[1]) >= X_test[column]) & (X_test[column] >= 0)] = woe_hash_copy[column][item[0]][0]
                    elif i == len(woe_hash_copy[column]) - 1:
                        X_test_c[column][X_test[column] >= float(item[0].split('-')[0])] = woe_hash_copy[column][item[0]][0]
                    else:
                        X_test_c[column][(float(item[0].split('-')[1]) >= X_test[column]) & (X_test[column] >= float(item[0].split('-')[0]))] = woe_hash_copy[column][item[0]][0]
            else:
                for j in woe_hash_copy[column].keys():
                    X_test_c[column][X_test[column] == j] = woe_hash_copy[column][j][0]
        return X_test_c
    
    def X_woe_heatmap(self):
        corrmat = self.X_woe.corr(method='pearson')
        plt.figure(figsize=(10,10))
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.title('Important Variables Correlation', fontsize=15)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.show()
