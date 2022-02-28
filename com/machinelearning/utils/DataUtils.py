import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

"""
@Time : 2022/2/28 14:11 
@Author : yhf
@desc : 数据处理工具类
"""

class DataUtils:

    """
    @desc : 数据量大时可用来减小内存开销。
    @df : dataFrame类型
    @return : dataFrame类型
    """
    @staticmethod
    def reduce_mem_usage(df):
        start_mem = df.memory_usage().sum() / 1024 ** 2
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return df
    """
    @desc : 划分训练和测试集,并生成训练集和测试集文件,若是数据集偏小且维度丰富,则易产生抽样偏差,建议使用分层抽样
    @df : dataFrame类型
    @return : train_df:训练集,test_df:测试集
    """
    @staticmethod
    def split_train_test(df):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv("train.scv")
        test_df.to_csv("test.scv")
        return train_df, test_df

    """
    @desc : 分层抽样划分训练和测试集,并生成训练集和测试集文件
    @df : dataFrame类型
    @dim : 维度列名,即那一列为分层列,string类型
    @return : strat_train_df: 训练集, strat_test_df:测试集
    """
    @staticmethod
    def stratified_shuffle_split(df, dim):
        sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sp.split(df, df[dim]):
            strat_train_df = df.loc[train_index]
            strat_test_df = df.loc[test_index]
        print("完整集抽样比例\n")
        print(str(df[dim].value_counts() / len(df))+"\n")
        print("训练集抽样比例\n")
        print(str(strat_train_df[dim].value_counts() / len(strat_train_df))+"\n")
        print("测试集抽样比例\n")
        print(str(strat_test_df[dim].value_counts() / len(strat_test_df)) + "\n")
        # 保存文件
        strat_train_df.to_csv("strat_train.scv")
        strat_test_df.to_csv("strat_test.scv")
        return strat_train_df, strat_test_df