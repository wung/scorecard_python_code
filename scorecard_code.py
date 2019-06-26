# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:48:27 2018

@author: XiaSiYang
"""
import numpy as np
from math import log
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,learning_curve
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.multiclass import type_of_target
import pickle
from sklearn.metrics import roc_curve,auc,confusion_matrix
import itertools
import os
from patsy import dmatrices
from patsy import dmatrix
import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 0})
pd.set_option('precision',4) #设置小数点后面4位，默认是6位
plt.rcParams['font.sans-serif'] = ['simHei'] #指定默认字体,防止乱码
plt.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.serif'] = ['SimHei']
font = {'family':'SimHei'}
plt.rc(*('font',),**font)


## 数据预处理
class DataProcess:
    """
    About:数据预处理相关函数功能集
    """
    
    def __init__(self,logging_file,dep='y'): #默认是y
        """
        about:初始参数
        :param dep: str,the label of y;
        """
        self.dep = dep
        self.logging_file = logging_file
    
    def str_var_check(self,df):
        """
        about:移除df中的字符型变量
        :param df: DataFrame
        :return: DataFrame and 字符型变量名列表
        """
        col_type = pd.DataFrame(df.dtypes) #获取每个变量的类型,返回Series
        col_type.columns = ['type'] #增加column名称
        save_col = list(col_type.loc[(col_type['type']== 'float64')|(col_type['type'] == 'int64')].index)
        #得到是浮点型、数值型的列名
        rm_col = [x for x in df.columns if x not in save_col] #得到df不含浮点型和数值的变量名
        save_df = df[save_col] #得到浮点型和数值的变量数据
        return (save_df,rm_col)
    
    def get_varname(self,df):
        """
        about:获取特征变量名
        :param df:DataFrame
        :return DataFrame
        """
        if self.dep in df.columns:
            dfClearnVar = df.drop(self.dep,1)
        else:
            dfClearnVar = df
        var_namelist = dfClearnVar.T
        var_namelist['varname'] = var_namelist.index
        var_namelist = var_namelist['varname'].reset_index()
        var_namelist = var_namelist.drop('index',1)
        return var_namelist
    
    def calcConcentric(self,df):
        """
        about:集中度计算
        :calcuate the concentration rate
        :mydat,pd.DataFrame
        """
        sumCount = len(df.index)
        colsDf = pd.DataFrame({'tmp':['tmp',np.nan,np.nan]})
        for col in self.notempty_var_list: #此处的self.notempty_var_list在下面的var_status定义
            print(col)
            valueCountDict = {}
            colDat = df.loc[:,col]
            colValueCounts = pd.value_counts(colDat).sort_values(ascending=False) #按照colDat的值计数，再排序，按照大到小
            concentElement = colValueCounts.index[0] #得到计算最大频率的值
            valueCountDict[col] = [concentElement,colValueCounts.iloc[0],colValueCounts.iloc[0]* 1.0 / sumCount]# 得到值、出现频率、出现占总数据比
            colDf = pd.DataFrame(valueCountDict)
            colsDf = colsDf.join(colDf) #join 方法添加
        #通过循环得到所有变量的取值的频率、占比
        colsDf = (colsDf.rename(index={0:'concentricElement',1:'concentricCount',2:'concentricRate'})).drop('tmp',axis=1) #替换index的值，同时删除多余的tmp
        return colsDf
        
    def var_status(self,df):
        """
        about:变量描述性统计
        :param df:
        :return: Dataframe
        """
        self.logging_file.info('正在进行变量描述统计...')
        CleanVar = df.drop(self.dep,1)
        describe = CleanVar.describe().T
        self.notempty_var_list = list(describe.loc[describe['count'] > 0].index) #得到变量个数大于0的变量名
        sample_num = int(describe['count'].max())
        describe['varname'] = describe.index
        describe.rename(columns={'count':'num','mean':'mean_v','std':'std_v','max':'max_v','min':'min_v'},inplace=True)
        describe['saturation'] = describe['num'] / sample_num
        describe = describe.drop(['25%','50%','75%'],1) #删除多余的列
        describe = describe.reset_index(drop=True) #重设index，且放弃之前的index
        describe['index'] = describe.index
        self.logging_file.info('正在计算变量集中度...')
        Concent = self.calcConcentric(df)
        concentricRate = pd.DataFrame(Concent.T['concentricRate'])
        concentricRate['index'] = concentricRate.index
        self.logging_file.info('正在计算变量IV值...')
        mywoeiv = Woe_iv(df,dep=self.dep,event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None)
        mywoeiv.woe_iv_vars()
        iv = mywoeiv.iv_dict
        iv = pd.DataFrame(pd.Series(iv,name='iv'))
        iv['index'] = iv.index
        var_des = pd.merge(describe,concentricRate,how='inner',left_on=['varname'],right_on=['index'])
        var_des = pd.merge(var_des,iv,how='inner',left_on =['varname'],right_on=['index'])
        var_des = var_des[['varname','num','mean_v','std_v','min_v','max_v','saturation','concentricRate','iv']]
        self.logging_file.info('描述统计完毕...')
        return var_des
    
    def var_filter(self,df,varstatus,min_saturation=0.01,max_concentricRate=0.98,min_iv=0.01):
        """
        about:变量初筛
        :param df:
        :param varstatus: dataframe.var_status函数的输出结果
        :param min_saturation:float 变量筛选值饱和度下限
        :param max_concentricRate: folat 变量筛选值集中度上限
        :param min_iv: float 变量筛选之IV值下限
        :return: DataFrame 这里nan会变成false
        """
        var_selected = list(varstatus['varname'].loc[(varstatus['saturation'] >= min_saturation) & (varstatus['concentricRate'] <= max_concentricRate)
        & (varstatus['iv'] >= min_iv)]) #依据要求得到符合条件的变量名
        var_selected.insert(0,self.dep) #插入y
        df_selected = df[var_selected]
        return df_selected #返回筛选后的数据
    
    def var_corr_delete(self,df_WoeDone_select,var_desc_woe,corr_limit=0.95):
        """
        about:剔除相关系数高的变量
        :param df_WoeDone_select: 变量初删后返回的值
        :param var_desc_woe: 变量描述统计后返回的值
        :param corr_limit:
        :return:
        """
        deleted_vars = []
        high_IV_var = list((df_WoeDone_select.drop(self.dep,axis=1)).columns)
        for i in high_IV_var:
            if i in deleted_vars:
                continue
            for j in high_IV_var:
                if not i == j:
                    if not j in deleted_vars:
                        roh = np.corrcoef(df_WoeDone_select[i],df_WoeDone_select[j])[(0,1)] #比较每两个之间的相关系数，最后删除IV值较小的数
                        if abs(roh) > corr_limit:
                            x1_IV = var_desc_woe.iv.loc[var_desc_woe.varname == i].values[0] #这个就是IV值
                            y1_IV = var_desc_woe.iv.loc[var_desc_woe.varname == j].values[0]
                            if x1_IV > y1_IV:
                                deleted_vars.append(j)
                                self.logging_file.info('变量' + i + '和' + '变量' + j + '高度相关,相关系数达到' + str(abs(roh)) + ',已删除' + j)
                            else:
                                deleted_vars.append(i)
                                self.logging_file.info('变量' + i + '和' + '变量' + j + '高度相关,相关系数达到' + str(abs(roh)) + ',已删除' + i)
        df_corr_select = df_WoeDone_select.drop(deleted_vars,axis=1) #删除选中的变量
        self.logging_file.info('已对相关系数达到' + str(corr_limit) + '以上的变量进行筛选，剔除的变量列表如下' + str(deleted_vars))
        return df_corr_select
    

class Plot_vars:
    """
    about:风险曲线图
    cut_points_bring:单变量,针对连续变量离散化，目标是均分xx组，但当某一值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group:单变量，离散化连续变量，并生产一个groupby数据:
    nodis_group:不需要离散化的变量，生产一个groupby数据:
    plot_var:单变量绘图
    plot_vars:多变量绘图
    """
    
    def __init__(self,mydat,dep='y',nodiscol= None,plotcol=None,disnums=5,file_name=None):
        """
        abount:变量风险图
        :param model_name:
        :param mydat:DataFrame,包含X,y的数据集
        :param dep: str,the label of y
        :param nodiscol: list,defalut None,当这个变量有数据时，会默认这里的变量不离散，且只画nodis_group,
        其余的变量都需要离散化，且只画dis_group.当这个变量为空时，系统回去计算各变量的不同数值的数量，若小于15，则认为不需要离散，直接丢到nodiscol中
        :param disnums: int,连续变量需要离散的组数
        """
        self.mydat = mydat
        self.dep = dep
        self.plotcol = plotcol #这个是制定多个变量，批量跑多个变量
        self.nodiscol = nodiscol
        self.disnums = disnums
        self.file_path = os.getcwd()
        self.file_name = file_name
        self.col_cut_points = {}
        self.col_notnull_count = {}
        for i in self.mydat.columns:
            if i != self.dep:
                self.col_cut_points[i] = []
        for i in self.mydat.columns:
            if i != self.dep:
                col_notnull = len(self.mydat[i][pd.notnull(self.mydat[i])].index)
                self.col_notnull_count[i] = col_notnull
        
        if self.nodiscol is None:
            nodiscol_tmp = []
            for i in self.mydat.columns:
                if i != self.dep:
                    col_cat_num = len(set(self.mydat[i][pd.notnull(self.mydat[i])]))
                    if col_cat_num < 5: #非空的数据分类小于5个
                        nodiscol_tmp.append(i)
            if len(nodiscol_tmp) > 0:
                self.nodiscol = nodiscol_tmp
        if self.file_name is not None:
            self.New_Path = self.file_path + '\\' + self.file_name + '\\'
            if not os.path.exists(self.New_Path):
                os.makedirs(self.New_Path) #增加新的文件夹
        
    def cut_point_bring(self,col_order,col):
        """
        about:分割函数
        :param col_order:DataFrame,非null得数据集，包含y，按变量值顺序排列
        :param col:str 变量名
        :return:
        """
        PCount = len(col_order.index)
        min_group_num = self.col_notnull_count[col] / self.disnums #特定变量的非null数据量除以默认分组5组
        disnums = int(PCount / min_group_num)   #数据集的数量除以最小分组的数量
        if PCount /self.col_notnull_count[col] >= 1 /self.disnums:         
            if disnums > 0 :
                n_cut = int(PCount /disnums)
                cut_point = col_order[col].iloc[n_cut - 1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point) #得到切点值
                        break #这里为了解决分割点的值多个相当，因此一直分到不相等，才退出
                self.cut_point_bring(col_order[n_cut:],col) #这里是递归函数,用剩下的数据继续跑
                
    
    def dis_group(self,col):
        """
        abount:连续性变量分组
        :param col:str,变量名称
        :return:
        """
        dis_col_data_notnull = self.mydat.loc[(pd.notnull(self.mydat[col]),[self.dep,col])]
        Oreder_P = dis_col_data_notnull.sort_values(by=[col],ascending=True)
        self.cut_point_bring(Oreder_P,col)
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min()) #得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col]) #加入变量的切点值
        dis_col_data = self.mydat.loc[:,[self.dep,col]]
        dis_col_data['group'] = np.nan
        for i in range(len(dis_col_cuts) - 1): #这里开始依据分组的切点，改变group的值
            if i == 0:
                dis_col_data.loc[dis_col_data[col] <= dis_col_cuts[i+1],['group']] = i
            elif i == len(dis_col_cuts) - 2:
                dis_col_data.loc[dis_col_data[col] > dis_col_cuts[i],['group']] = i
            else:
                dis_col_data.loc[(dis_col_data[col] > dis_col_cuts[i]) & (dis_col_data[col] <= dis_col_cuts[i+1]),['group']] = i
        dis_col_data[col] = dis_col_data['group']
        dis_col_bins = []
        dis_col_bins.append('nan')
        dis_col_bins.extend(['(%s,%s]' % (dis_col_cuts[i],dis_col_cuts[i+1]) for i in range(len(dis_col_cuts) - 1)])
        dis_col = dis_col_data.fillna(-1) 
        col_group = (dis_col.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'}) #按照切点分组，按dep计数
        avg_risk = self.mydat[self.dep].sum() /self.mydat[self.dep].count() #平均风险度
        col_group['totalrate'] = col_group['count'] / col_group['count'].sum() #占比totalrate
        col_group['badrate'] = col_group['bad_num'] / col_group['count'] #坏客户占比badrate
        col_group['lift'] = col_group['badrate'] / avg_risk #风险倍数即lift
        if -1 in list(col_group[col]):
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        col_group[col] = col_group[col].astype(np.float) #转换数据类型
        col_group = col_group.sort_values([col],ascending=True)
        col_group = pd.DataFrame(col_group,columns=[col,'bins','totalrate','badrate','lift'])
        return col_group  

    def nodis_group(self,col):
        """
        aount:离散型变量分组，主要处理null值
        :param col:str 变量名称
        :return:
        """
        nodis_col_data = self.mydat.loc[:,[self.dep,col]]
        is_na = pd.isnull(nodis_col_data[col]).sum() > 0 #判断是否有null
        col_group = (nodis_col_data.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])
        if is_na:
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count] #添加到最后一行
        avg_risk = self.mydat[self.dep].sum() /self.mydat[self.dep].count()
        col_group['totalrate'] = col_group['count'] / col_group['count'].sum()
        col_group['badrate'] = col_group['bad_num'] /col_group['count']
        col_group['lift'] = col_group['badrate'] / avg_risk #风险倍数
        if is_na:
            bins = col_group[col][:len(col_group.index) - 1]
            bins[len(col_group.index)] = 'nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index) - 1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        col_group[col] = col_group[col].astype(np.float)
        col_group = col_group.sort_values([col],ascending=True)
        col_group = pd.DataFrame(col_group,columns=[col,'bins','totalrate','badrate','lift'])
        return col_group
    
    def plot_var(self,col):
        """
        about:单变量泡泡图
        :param col: str 变量名称
        :return:
        """
        if self.nodiscol is not None:
            if col in self.nodiscol:
                col_group = self.nodis_group(col)
            else:
                col_group = self.dis_group(col)
        fix,ax1 = plt.subplots(figsize=(8,6))
        ax1.bar(x=list(col_group[col]),height=col_group['totalrate'],width=0.6,align='center') #bar是柱形图
        for a,b in zip(col_group[col],col_group['totalrate']):
            ax1.text(a,b+0.005,'%.4f' %b,ha='center',va='bottom',fontsize=8) #给柱形图加上文字标示
            
        ax1.set_xlabel(col)
        ax1.set_ylabel('percent',fontsize=12)
        ax1.set_xlim([-2,max(col_group[col])+2]) #设置刻度长度
        ax1.set_ylim([0,max(col_group['totalrate'])+0.3])
        ax1.grid(False) #不显示网格
        ax2 = ax1.twinx() #在原有的图像上添加坐标轴，类似双轴
        ax2.plot(list(col_group[col]),list(col_group['lift']),'-ro',color='red')
        for a,b in zip(col_group[col],col_group['lift']):
            ax2.text(a,b+0.05,'%.4f' % b,ha='center',va='bottom',fontsize=7)
            
        ax2.set_ylabel('lift',fontsize=12)
        ax2.set_ylim([0,max(col_group['lift']) + 0.5])
        ax2.grid(False)
        plt.title(col)
        the_table = plt.table(cellText=col_group.round(4).values,colWidths=[0.12]*len(col_group.columns), #round(4)保留4位小数
        rowLabels=col_group.index,colLabels=col_group.columns,loc=1,cellLoc='center') #loc=1显示在右对齐.loc=2显示在左对齐.colWidths是每列表格的长度占比，5列，每列0.2，就占满
        the_table.auto_set_font_size(False) #自动设置字体大小
        the_table.set_fontsize(10) #设置表格的大小
        if self.file_path is not None:
            plt.savefig(self.New_Path+col+'.pdf',dpi=600)
        # plt.show()
        
    def plot_vars(self):
        """
        about:批量跑多个变量，如果未制定多列或者单个变量，将会跑所有的变量
        """
        if self.plotcol is not None: #这个是制定多个变量，批量跑多个变量
            for col in self.plotcol:
                self.plot_var(col)
        else:
            cols = self.mydat.columns
            for col in cols:
                print(col)
                if col != self.dep:
                    self.plot_var(col)


class BestBin:
    """
    about:分箱类函数
    """

    def __init__(self, data_in, dep='y', method=4, group_max=3, per_limit=0.05, min_count=0.05, logging_file=None):
        """
        about:参数设置
        :param data_in:DataFrame 包含X,y的数据集
        :param dep:str,the label of y;
        :param method:int,defaut 4,分箱指标选取：1 基尼系数，2 信息熵，3 皮尔逊卡方统计量、4IV值
        :param group_max: int,defalut 3, 最大分箱组数
        :param per_limit: float,小切割范围占比
        """
        self.data_in = data_in
        self.dep = dep
        self.logging_file = logging_file
        if self.dep in self.data_in.columns:
            self.CleanVar = self.data_in.drop(self.dep, 1)  # 删除Y
        else:
            self.CleanVar = self.data_in
        self.method = method
        self.group_max = group_max
        self.per_limit = per_limit
        self.min_count = min_count
        self.file_path = os.getcwd()
        self.bin_num = int(1 / per_limit)
        self.nrows = data_in.shape[0]
        self.col_cut_points = {}
        self.col_notnull_count = {}
        for i in self.CleanVar.columns:
            self.col_cut_points[i] = []
            col_notnull = len(self.data_in[i][pd.notnull(self.data_in[i])].index)
            self.col_notnull_count[i] = col_notnull

    def cut_points_bring(self, col_order, col):
        """
        about:分割函数
        :param col_order:DataFrame,非null的数据集，包含y，按变量值顺序排列
        :param col: str 变量名
        :return:
        """
        PCount = len(col_order.index)  # 得到数据长度
        min_group_num = self.col_notnull_count[col] * self.per_limit  # 得到最小分组数量，这个是不变的
        disnums = int(PCount / min_group_num)  # 得到分组数，每次都会变化
        if PCount / self.col_notnull_count[col] >= self.per_limit * 2:  # 剩余的数量/总的数量 大于等于最小分割的2呗，留下最后一份
            if disnums > 0:  # 最后分组要大于0
                n_cut = int(PCount / disnums)
                cut_point = col_order[col].iloc[n_cut - 1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point)
                        break
                self.cut_points_bring(col_order[n_cut:], col)  # 递归调用自己

    def groupbycount(self, input_data, col):
        """
        abount:分组
        :param col:str 变量名称
        :param input_data:DataFrame 输入数据
        :return:
        """
        dis_col_data_notnull = self.data_in.loc[(pd.notnull(self.data_in[col]), [self.dep, col])]  # 得到变量函和Y的数据
        Oreder_P = dis_col_data_notnull.sort_values(by=[col], ascending=True)  # 将所有的数据排序
        self.cut_points_bring(Oreder_P, col)  # 调用分割函数,按照
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min())  # 得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col])  # 加入变量的切点值
        dis_col_cuts.append(dis_col_data_notnull[col].max())  # 加入变量最大值
        input_data['group'] = input_data[col]
        for i in range(len(dis_col_cuts) - 1):  # 这里开始依据分组的切点，改变group的值
            if i == 0:  # 第一组
                input_data.loc[input_data[col] <= dis_col_cuts[i + 1], ['group']] = i + 1
            elif i == len(dis_col_cuts) - 2:  # 最后一组
                input_data.loc[input_data[col] > dis_col_cuts[i], ['group']] = i + 1
            else:
                input_data.loc[
                    (input_data[col] > dis_col_cuts[i]) & (input_data[col] <= dis_col_cuts[i + 1]), ['group']] = i + 1

        return (input_data, dis_col_cuts)  # 返回分组值，分组切点

    def bin_con_var(self, data_in_stable, indep_var, group_now):
        """
        about:分箱
        :param data_in_stable:输入数据集
        :param indep_var:自变量
        :param group_now:分箱切点映射表,也是那个最大分组数
        :return:
        """
        data_in = data_in_stable.copy()  # 复制
        data_in, bin_cut = self.groupbycount(data_in, indep_var)  # 调用分组函数
        Mbins = len(bin_cut) - 1
        Temp_DS = data_in.loc[:, [self.dep, indep_var, 'group']]
        Temp_DS['group'].fillna(0, inplace=True)  # 将null值用0来填充-未实施此步骤
        temp_blimits = pd.DataFrame({'Bin_LowerLimit': [], 'Bin_UpperLimit': [], 'group': []})
        for i in range(Mbins):  # 得到变量分组后的上下限值
            temp_blimits.loc[(i, ['Bin_LowerLimit'])] = bin_cut[i]
            temp_blimits.loc[(i, ['Bin_UpperLimit'])] = bin_cut[i + 1]
            temp_blimits.loc[(i, ['group'])] = i + 1

        grouped = Temp_DS.loc[:, [self.dep, 'group']].groupby(['group'])  # 分组
        g1 = grouped.sum().rename(columns={self.dep: 'y_sum'})
        g2 = grouped.count().rename(columns={self.dep: 'count'})
        g3 = pd.merge(g1, g2, left_index=True, right_index=True)
        g3['good_sum'] = g3['count'] - g3['y_sum']
        g3['group'] = g3.index
        g3['PDVI'] = g3.index
        g3.rename(columns={'group': 'id'}, inplace=True)
        temp_cont = pd.merge(g3, temp_blimits, how='left', on='group')  # 得到分组统计的数据
        for i in range(Mbins):
            mx = temp_cont.loc[(temp_cont['group'] == i + 1, 'group')].values
            if mx:
                Ni1 = temp_cont['good_sum'].loc[temp_cont['group'] == i + 1].values[0]
                Ni2 = temp_cont['y_sum'].loc[temp_cont['group'] == i + 1].values[0]
                count = temp_cont['count'].loc[temp_cont['group'] == i + 1].values[0]
                bin_lower = temp_cont['Bin_LowerLimit'].loc[temp_cont['group'] == i + 1].values[0]
                bin_upper = temp_cont['Bin_UpperLimit'].loc[temp_cont['group'] == i + 1].values[0]
                if i == Mbins - 1:  # 如果i是最后一位
                    i1 = temp_cont['group'].loc[temp_cont['group'] < Mbins].values.max()  # 最后一位就取前面中最大的
                else:
                    i1 = temp_cont['group'].loc[temp_cont['group'] > i + 1].values.min()  # 其他的就取后面中最小的
                if Ni1 == 0 or Ni2 == 0 or count == 0:
                    # 如果有一组好客户、或者坏客户、或者总数为0，就把本组的所有人数加到后一组中，如果是最后一组，就加到上一组中，同时改正分组的阈值，再删除当组的数据
                    temp_cont['good_sum'].loc[temp_cont['group'] == i1] = \
                    temp_cont['good_sum'].loc[temp_cont['group'] == i1].values[0] + Ni1
                    temp_cont['y_sum'].loc[temp_cont['group'] == i1] = \
                    temp_cont['y_sum'].loc[temp_cont['group'] == i1].values[0] + Ni2
                    temp_cont['count'].loc[temp_cont['group'] == i1] = \
                    temp_cont['count'].loc[temp_cont['group'] == i1].values[0] + count
                    if i < Mbins - 1:
                        temp_cont['Bin_LowerLimit'].loc[temp_cont['group'] == i1] = bin_lower
                    else:
                        temp_cont['Bin_UpperLimit'].loc[temp_cont['group'] == i1] = bin_upper
                    delete_indexs = list(temp_cont.loc[temp_cont['group'] == i + 1].index)
                    temp_cont = temp_cont.drop(delete_indexs)

        temp_cont['new_index'] = (temp_cont.reset_index(drop=True)).index + 1
        temp_cont['var'] = temp_cont['group']
        temp_cont['group'] = 1
        Nbins = 1
        while Nbins < group_now:  # 这会一直分组
            Temp_Splits = self.CandSplits(temp_cont)
            temp_cont = Temp_Splits
            Nbins = Nbins + 1

        temp_cont.rename(columns={'var': 'OldBin'}, inplace=True)
        ## 这个地方需要改下吧
        temp_Map1 = temp_cont.drop(['good_sum', 'PDVI', 'new_index'], axis=1)
        temp_Map1 = temp_Map1.sort_values(by=['group'])
        min_group = temp_Map1['group'].min()
        max_group = temp_Map1['group'].max()
        lmin = temp_Map1['Bin_LowerLimit'].min()
        notnull = temp_Map1.loc[temp_Map1['Bin_LowerLimit'] > lmin - 10]
        var_map = pd.DataFrame(
            {'group': [], 'LowerLimit': [], 'UpperLimit': [], 'total': [], 'bad': [], 'good': [], 'risk': []})
        for i in range(min_group, max_group + 1):
            ll = notnull['Bin_LowerLimit'].loc[notnull['group'] == i].min()
            uu = notnull['Bin_UpperLimit'].loc[notnull['group'] == i].max()
            total = notnull['count'].loc[notnull['group'] == i].sum()
            bad = notnull['y_sum'].loc[notnull['group'] == i].sum()
            good = total - bad

            if total > 0:
                risk = bad * 1.0 / (total + 0.0001)
                var_map = var_map.append(
                    {'group': i, 'LowerLimit': ll, 'UpperLimit': uu, 'total': total, 'bad': bad, 'good': good,
                     'risk': risk}, ignore_index=True)

        null_group = temp_Map1['group'].loc[temp_Map1['Bin_LowerLimit'].isnull()]
        if null_group.any():
            temp_Map_null = temp_Map1.loc[temp_Map1['Bin_LowerLimit'].isnull()]
            ll = temp_Map_null['Bin_LowerLimit'].min()
            uu = temp_Map_null['Bin_UpperLimit'].max()
            total = temp_Map_null['count'].sum()
            bad = temp_Map_null['y_sum'].sum()
            good = total - bad
            i = temp_Map_null['group'].max()
            risk = bad * 1.0 / total
            var_map = var_map.append(
                {'group': i, 'LowerLimit': ll, 'UpperLimit': uu, 'total': total, 'bad': bad, 'good': good,
                 'risk': risk}, ignore_index=True)
        var_map = var_map.sort_values(by=['LowerLimit', 'UpperLimit'])
        var_map['newgroup'] = var_map.reset_index().index + 1
        var_map = var_map.reset_index()
        var_map = var_map.drop('index', 1)
        if null_group.any():
            ng = var_map['group'].loc[var_map['LowerLimit'].isnull()].max()
            notnull = var_map.loc[var_map['LowerLimit'].notnull()]
            cng = var_map['group'].loc[var_map['group'] == ng].count()
            if cng > 1:
                var_map['newgroup'].loc[var_map['LowerLimit'].isnull()] = notnull['newgroup'].loc[
                    notnull['group'] == ng].max()
                var_map['group'] = var_map['newgroup']
            else:
                var_map['group'] = var_map['newgroup']
                var_map['group'].loc[var_map['LowerLimit'].isnull()] = 0
        else:
            var_map['group'] = var_map['newgroup']
        var_map = var_map.drop('newgroup', 1)
        var_map['totalpct'] = var_map['total'] * 1.0 / var_map['total'].sum()
        var_map['badpct'] = var_map['bad'] * 1.0 / var_map['bad'].sum()
        var_map['goodpct'] = var_map['good'] * 1.0 / var_map['good'].sum()
        var_map['woe'] = np.log(var_map['badpct'] / var_map['goodpct'])
        var_map['iv'] = (var_map['badpct'] - var_map['goodpct']) * var_map['woe']
        var_map['IV'] = var_map['iv'].sum()
        var_map['var_name'] = indep_var
        return var_map

    def df_bin_con_var(self):
        """
        about:分箱
        :return
        """
        ToBin = self.CleanVar.T  # 删除后的Y转置
        ToBin['varname'] = ToBin.index  # 得到变量名
        for i in ToBin['varname']:
            self.logging_file.info(i)
            varnum = self.data_in.loc[:, [self.dep, i]].groupby([i])  # 以变量分组,groupby 自动删除null计数
            vcount = len(varnum.count().index)  # 得到分组的个数
            if vcount <= 2:
                self.data_in.loc[:, i + '_g'] = self.data_in[i]
                self.data_in.loc[self.data_in[i + '_g'].isnull(), [i + '_g']] = -1  # 将_g变量中所有的null变为-1
            else:
                var_group_map = self.bin_con_var(self.data_in, i, self.group_max)  # 开始分组,得到分组值
                mincount = var_group_map['totalpct'].min()  # 得到占比最小
                group_now = len(var_group_map.group.unique())  # 得到分组数
                while mincount < self.min_count and group_now > 3:  # 判断最小分组占比小于设定的比率，且现在分组大于2组，则一直分下去
                    group_now = group_now - 1
                    var_group_map = self.bin_con_var(self.data_in, i, group_now)
                    mincount = var_group_map['total'].min() / var_group_map['total'].sum()

                self.ApplyMap(self.data_in, i, var_group_map)  #
                print(var_group_map)
                if self.file_path is not None:
                    self.New_Path = self.file_path + '\\bestbin\\'
                    if not os.path.exists(self.New_Path):
                        os.makedirs(self.New_Path)
                var_group_map.to_excel(self.New_Path + i + '.xlsx', index=False)

    def CandSplits(self, BinDS):
        """
        about:Generate all candidate splits from currentBins and select the best new bins,first we sort the dataset OldBins by PDVI and Bin
        :param BinDS
        :return
        """
        BinDS.sort_values(by=['group', 'PDVI'], inplace=True)  # 先排序
        Bmax = BinDS['group'].values.max()  # 取分组最大
        m = []
        names = locals()  # 返回当前的所有局部变量
        for i in range(Bmax):  # 当两组之后的，就把数据集合给分割成不同的部分
            names['x%s' % i] = BinDS.loc[BinDS['group'] == i + 1]
            m.append(BinDS['group'].loc[BinDS['group'] == i + 1].count())  # 不同部分的组别个数，大于1下面就分割，否则不分割

        temp_allVals = pd.DataFrame({'BinToSplit': [], 'DatasetName': [], 'Value': []})
        for i in range(Bmax):  # 对每个组别
            if m[i] > 1:  # 如果组别的个数大于1
                testx = self.BestSplit(names['x%s' % i], i)  # 调用函数
                names['temp_trysplit%s' % i] = testx
                names['temp_trysplit%s' % i].loc[names['temp_trysplit%s' % i]['Split'] == 1, ['group']] = Bmax + 1
                d_indexs = list(BinDS.loc[BinDS['group'] == i + 1].index)
                names['temp_main%s' % i] = BinDS.drop(d_indexs)
                names['temp_main%s' % i] = pd.concat([names['temp_main%s' % i], names['temp_trysplit%s' % i]])
                Value = self.GValue(names['temp_main%s' % i])  # 这里是求按照上述分组后的值
                temp_allVals = temp_allVals.append({'BinToSplit': i, 'DatasetName': 'temp_main%s' % i, 'Value': Value},
                                                   ignore_index=True)
                # 这里是对分了两组后的数据，再此分组，加入其的Value值，下面来判断哪个是最大的，那下次返回的数据就是最大的一个的分组，这样1分为2，2分为3，不会到4

        ifsplit = temp_allVals['BinToSplit'].max()
        if ifsplit >= 0:
            temp_allVals = temp_allVals.sort_values(by=['Value'], ascending=False)
            bin_i = int(temp_allVals['BinToSplit'][0])
            NewBins = names['temp_main%s' % bin_i].drop('Split', 1)
        else:
            NewBins = BinDS
        return NewBins

    def BestSplit(self, BinDs, BinNo):
        """
        about:
        :param BinDs
        :param BinNo
        :return:
        """
        mb = BinDs['group'].loc[BinDs['group'] == BinNo + 1].count()
        BestValue = 0
        BestI = 1
        for i in range(mb - 1):  # 重复循环，计算出最大的Value值
            Value = self.CalcMerit(BinDs, i + 1)  # 再次调用函数,以i+1为分割点，计算相应两部分的值
            if BestValue < Value:
                BestValue = Value
                BestI = i + 1
        BinDs.loc[:, 'Split'] = 0
        BinDs.loc[BinDs['new_index'] <= BestI, ['Split']] = 1  # 可以找到最佳的二分点
        BinDs = BinDs.drop('new_index', 1)
        BinDs = BinDs.sort_values(by=['Split', 'PDVI'], ascending=True)
        BinDs['testindex'] = (BinDs.reset_index(drop=True)).index
        BinDs['new_index'] = BinDs['testindex'] + 1
        BinDs.loc[BinDs['Split'] == 1, ['new_index']] = BinDs['new_index'].loc[BinDs['Split'] == 1] - \
                                                        BinDs['Split'].loc[BinDs['Split'] == 0].count()
        NewBinDs = BinDs.drop('testindex', 1)
        return NewBinDs

    def CalcMerit(self, BinDs, ix):
        """
        about:
        :param BinDs
        :param ix
        :return
        """
        n_11 = BinDs['good_sum'].loc[BinDs['new_index'] <= ix].sum()
        n_21 = BinDs['good_sum'].loc[BinDs['new_index'] > ix].sum()
        n_12 = BinDs['y_sum'].loc[BinDs['new_index'] <= ix].sum()
        n_22 = BinDs['y_sum'].loc[BinDs['new_index'] > ix].sum()
        n_1s = BinDs['count'].loc[BinDs['new_index'] <= ix].sum()
        n_2s = BinDs['count'].loc[BinDs['new_index'] > ix].sum()
        n_s1 = BinDs['good_sum'].sum()
        n_s2 = BinDs['y_sum'].sum()
        if self.method == 1:  # 基尼系数
            N = n_1s + n_2s
            G1 = 1 - (n_11 * n_11 + n_12 * n_12) / (n_1s * n_1s)  # 1-好人占比平方-坏人占比平方
            G2 = 1 - (n_21 * n_21 + n_22 * n_22) / (n_2s * n_2s)  # 同上
            G = 1 - (n_s1 * n_s1 + n_s2 * n_s2) / (N * N)  # 好坏占总人数的G值
            Gr = 1 - (n_1s * G1 + n_2s * G2) / (N * G)  # 这里将原来分之后的/分之前的G值，再被1减去，反应改变后的变化
            M_value = Gr

        if self.method == 2:  # 信息熵
            N = n_1s + n_2s
            E1 = -(n_11 / n_1s * log(n_11 / n_1s) + n_12 / n_1s * log(n_12 / n_1s)) / log(2)  # 这里用换底公式
            E2 = -(n_21 / n_2s * log(n_21 / n_2s) + n_22 / n_2s * log(n_22 / n_2s)) / log(2)
            E = -(n_s1 / N * log(n_s1 / N) + n_s2 / N * log(n_s2 / N)) / log(2)
            Er = 1 - (n_1s * E1 + n_2s * E2) / (N * E)  # 此处为了方便统一，都是用分组前和分组后的比值来体现
            M_value = Er

        if self.method == 3:  # 皮尔逊卡方统计量
            N = n_1s + n_2s
            m_11 = n_1s * n_s1 / N  # 计算好客户预期值，在区间1
            m_12 = n_1s * n_s2 / N  # 计算坏客户预期值，在区间1
            m_21 = n_2s * n_s1 / N  # 计算好客户预期值，在区间2
            m_22 = n_2s * n_s2 / N  # 计算坏客户预期值，在区间2
            X2 = (n_11 - m_11) * (n_11 - m_11) / m_11 + (n_12 - m_12) * (n_12 - m_12) / m_12 + (n_21 - m_21) * (
                        n_21 - m_21) / \
                 m_21 + (n_22 - m_22) * (n_22 - m_22) / m_22  # 实际减去预期**2之和
            M_value = X2

        if self.method == 4:  # IV值
            IV = (n_11 / n_s1 - n_12 / n_s2) * log(n_11 * n_s2 / (n_12 * n_s1 + 0.01) + 1e-05) + (n_21 / n_s1 - n_22 / n_s2) * log(
                n_21 * n_s2 / (n_22 * n_s1 + 0.01) + 1e-05)  # 两个的IV值相加
            M_value = IV

        return M_value

    def GValue(self, BinDs):
        """
        about:
        :param BinDs
        :return
        """
        R = BinDs['group'].max()
        N = BinDs['count'].sum()
        nnames = locals()
        for i in range(R):
            nnames['N_1%s' % i] = BinDs['good_sum'].loc[BinDs['group'] == i + 1].sum()
            nnames['N_2%s' % i] = BinDs['y_sum'].loc[BinDs['group'] == i + 1].sum()
            nnames['N_s%s' % i] = BinDs['count'].loc[BinDs['group'] == i + 1].sum()
            N_s_1 = BinDs['good_sum'].sum()
            N_s_2 = BinDs['y_sum'].sum()

        if self.method == 1:
            aa = locals()
            for i in range(R):
                aa['G_%s' % i] = 0
                aa['G_%s' % i] = aa['G_%s' % i] + nnames['N_1%s' % i] * nnames['N_1%s' % i] + nnames['N_2%s' % i] * \
                                 nnames['N_2%s' % i] * \
                                 nnames['N_2%s' % i]
                aa['G_%s' % i] = 1 - aa['G_%s' % i] / (nnames['N_s%s' % i] * nnames['N_s%s' % i])

            G = N_s_1 * N_s_1 + N_s_2 * N_s_2
            G = 1 - G / (N * N)
            Gr = 0
            for i in range(R):
                Gr = Gr + nnames['N_s%s' % i] * aa['G_%s' % i] / N

            M_Value = 1 - Gr / G

        if self.method == 2:
            for i in range(R):
                if nnames['N_1%s' % i] == 0 or nnames['N_1%s' % i] == '' or nnames['N_2%s' % i] == 0 or nnames['N_2%s' % i] == '':
                    M_Value = ''
                    return

            nnames['E_%s' % i] = 0
            for i in range(R):
                nnames['E_%s' % i] = nnames['E_%s' % i] - nnames['N_1%s' % i] / nnames['N_s%s' % i] * log(nnames['N_1%s' % i] / \
                    nnames['N_s%s' % i])
                nnames['E_%s' % i] = nnames['E_%s' % i] - nnames['N_2%s' % i] / nnames['N_s%s' % i] * log(
                    nnames['N_2%s' % i] / \
                    nnames['N_s%s' % i])
                nnames['E_%s' % i] = nnames['E_%s' % i] / log(2)

            E = 0
            E = E - N_s_1 / N * log(N_s_1 / N) - N_s_2 / N * log(N_s_2 / N)
            E = E / log(2)
            Er = 0
            for i in range(R):
                Er = Er + nnames['N_s%s' % i] * nnames['E_%s' % i] / N

            M_Value = 1 - Er / E

        if self.method == 3:
            N = N_s_1 + N_s_2
            X2 = 0
            for i in range(R):
                nnames['m_1%s' % i] = nnames['N_s%s' % i] * N_s_1 / N
                X2 = X2 + (nnames['N_1%s' % i] - nnames['m_1%s' % i]) * (nnames['N_1%s' % i] - nnames['m_1%s' % i]) / \
                     nnames['m_1%s' % i]
                nnames['m_2%s' % i] = nnames['N_s%s' % i] * N_s_2 / N
                X2 = X2 + (nnames['N_2%s' % i] - nnames['m_2%s' % i]) * (nnames['N_2%s' % i] - nnames['m_2%s' % i]) / \
                     nnames['m_2%s' % i]
                nnames['m_2%s' % i]

            M_Value = X2

        if self.method == 4:
            IV = 0
            for i in range(R):
                if nnames['N_1%s' % i] == 0 or nnames['N_1%s' % i] == '' or nnames['N_2%s' % i] == 0 or nnames['N_2%s' % i] == '' \
                        or N_s_1 == 0 or N_s_1 == '' or N_s_2 == 0 or N_s_2 == '':
                    M_Value = ''
                    return
            for i in range(R):
                IV = IV + (nnames['N_1%s' % i] / N_s_1 - nnames['N_2%s' % i] / N_s_2) * log(nnames['N_1%s' % i] * N_s_2 / \
                    (nnames['N_2%s' % i] * N_s_1))
            M_Value = IV

        return M_Value

    def ApplyMap(self, DSin, VarX, DSVapMap):
        """
        about:分箱组数替换 Dataframe
        :param Dsin 原始数据
        :param VarX 需要更新的变量名 str
        :param DSVapMap 读取的需要更新的分组数据 Dataframe
        :return
        """
        null_g = DSVapMap['group'].loc[DSVapMap['LowerLimit'].isnull()].max()  # 判断是否有null的group,有的话，把值改成group
        DSin.loc[:, VarX + '_g'] = 0
        if null_g > 0:
            DSin.loc[DSin[VarX].isnull(), [VarX + '_g']] = int(null_g)  # 有的话，全部是最大
        lmin = DSVapMap['LowerLimit'].min()
        nnull = DSVapMap.loc[DSVapMap['LowerLimit'] > lmin - 10]  # 这里计算数值分类是在10组之外的要用到分组，10组之内的不用替换
        nnull = nnull.sort_values(by=['LowerLimit'], ascending=True)
        nnull = nnull.reset_index(drop=True)
        mm = nnull['group'].count()
        for i in range(mm):  # 开始替换数据
            ll = nnull['LowerLimit'][i]
            uu = nnull['UpperLimit'][i]
            gg = nnull['group'][i]
            if i == 0:
                DSin.loc[DSin[VarX] <= uu, [VarX + '_g']] = gg
            elif i == mm - 1 and uu > ll:
                DSin.loc[DSin[VarX] > ll, [VarX + '_g']] = gg
            elif i == mm - 1:
                DSin.loc[DSin[VarX] >= ll, [VarX + '_g']] = uu == ll and gg
            else:
                DSin.loc[(DSin[VarX] > ll) & (DSin[VarX] <= uu), [VarX + '_g']] = gg

class Woe_iv:
    """
    about: woe iv 类计算函数
    check_target_binary:检查是否是二分类问题
    target_count:计算好坏样本数目
    cut points_bring:单变量，针对连续变量离散化，目标是均分xx组，但当某一数值重复过高时会适当调整，最后产出的是分割点，不包含首尾
    dis_group: 单变量，离散化连续变量，并生产一个groupby数据
    nodis_group:不需要离散化的变量，生产一个groupby数据
    woe_iv_var:单变量，计算各段的woe值和iv
    woe_iv_vars:多变量，计算多变量的woe和iv
    apply_woe_replace:将数据集中的分段替换成对应的woe值
    一般应大于0.02，默认选IV大于0.1的变量进模型，但具体要结合实际。如果IV大于0.5，就是过预测（over-predicting）变量
    AUC在 0.5～0.7时有较低准确性， 0.7～0.8时有一定准确性, 0.8~0.9则高，AUC在0.9以上时有非常高准确性。AUC=0.5时，说明诊断方法完全不起作用，无诊断价值
    psi 判断：index <= 0.1，无差异；0.1< index <= 0.25，需进一步判断；0.25 <= index，有显著位移，模型需调整。
    """
    def __init__(self,mydat,dep='y',event=1,nodiscol=None,ivcol=None,disnums=20,X_woe_dict=None):
        """
        about:初始化参数设置
        :param mydat:DataFrame 输入的数据集，包含y
        :param dep: str,the label of y
        :param event: int y中bad的标签
        :param nodiscol: list,defalut
        None,不需要离散的变量名，当这个变量有数据时，会默认这里的变量不离散，且只跑nodis_group,其余的变量都需要离散化，且只跑
        dis_group。当这个变量为空时，系统会去计算各变量的不同数值的数量，若小于15，则认为不需要离散，直接丢到nodiscol中
        :param ivcol: list 需要计算woe,iv的变量名，该变量不为None时，只跑这些变量，否则跑全体变量
        :param disnums: int 连续变量离散化的组数
        :param X_woe_dict: dict 每个变量每段的woe值，这个变量主要是为了将来数据集中的分段替换成对应的woe值
        即输入的数据集已经超过离散化分段处理，只需要woe化而已
        """
        self.mydat = mydat
        self.event = event
        self.nodiscol = nodiscol
        self.ivcol = ivcol
        self.disnums = disnums
        self._WOE_MIN = -20
        self._WOE_MAX = 20
        self.dep = dep
        self.col_cut_points = {}
        self.col_notnull_count = {}
        self._data_new = self.mydat.copy(deep=True)
        self.X_woe_dict = X_woe_dict
        self.iv_dict = None

        for i in self.mydat.columns:
            if i != self.dep:
                self.col_cut_points[i] = []
                
        for i in self.mydat.columns:
            if i != self.dep:
                col_notnull = len(self.mydat[i][pd.notnull(self.mydat[i])].index)
                self.col_notnull_count[i] = col_notnull
                
        if self.nodiscol is None:
            nodiscol_tmp = []
            for i in self.mydat.columns:
                if i != self.dep:
                    col_cat_num = len(set(self.mydat[i][pd.notnull(self.mydat[i])]))
                    if col_cat_num < 20:
                        nodiscol_tmp.append(i)
            
            if len(nodiscol_tmp) > 0:
                self.nodiscol = nodiscol_tmp
                
    def check_target_binary(self,y):
        """
        about: 检测因变量是否为二元变量
        :param y: the target variable,series type
        :return:
        """
        y_type = type_of_target(y) #检测数据是不是二分类数据
        if y_type not in ('binary',):
            raise ValueError('Label tyoe must be binary!!!')
        
    def target_count(self,y):
        """
        about：计算Y值得数量
        :param y: the target variable,series type
        :return: 0,1的数量 
        """
        y_count = y.value_counts()
        if self.event not in y_count.index:
            event_count = 0
        else:
            event_count = y_count[self.event]
        non_event_count = len(y) - event_count
        return (event_count,non_event_count) #返回好坏客户的数量
    
    def cut_points_bring(self,col_order,col):
        """
        about:分割函数
        :param col_order: DataFrame 非null的数据集，包含y，按变量值顺序排列
        :param col: str 变量名
        :return 
        """
        PCount = len(col_order.index)
        min_group_num = self.col_notnull_count[col] /self.disnums
        disnums = int(PCount/min_group_num)
        if PCount / self.col_notnull_count[col] >= 1 / self.disnums:
            if disnums >0 :
                n_cut = int(PCount / disnums)
                cut_point = col_order[col].iloc[n_cut-1]
                for i in col_order[col].iloc[n_cut:]:
                    if i == cut_point:
                        n_cut += 1
                    else:
                        self.col_cut_points[col].append(cut_point)
                        break
                self.cut_points_bring(col_order[n_cut:],col)
    
    def dis_group(self,col):
        """
        abount:连续型变量分组
        :param col:str 变量名称
        :return:
        """
        dis_col_data_notnull = self.mydat.loc[(pd.notnull(self.mydat[col]),[self.dep,col])]
        Oreder_P = dis_col_data_notnull.sort_values(by=[col],ascending=True)
        self.cut_points_bring(Oreder_P,col)
        dis_col_cuts = []
        dis_col_cuts.append(dis_col_data_notnull[col].min()) #得到变量的最小值
        dis_col_cuts.extend(self.col_cut_points[col]) #加入变量的切点值
        dis_col_cuts.append(dis_col_data_notnull[col].max()) #得到变量的最大值
        dis_col_data = self.mydat.loc[:,[self.dep,col]]
        dis_col_data['group'] = np.nan
        for i in range(len(dis_col_cuts) - 1): #这里开始依据分组的切点，改变group的值
            if i == 0:
                dis_col_data.loc[dis_col_data[col] <= dis_col_cuts[i+1],['group']] = i
            elif i == len(dis_col_cuts) - 2:
                dis_col_data.loc[dis_col_data[col] > dis_col_cuts[i],['group']] = i
            else:
                dis_col_data.loc[(dis_col_data[col] > dis_col_cuts[i]) & (dis_col_data[col] <= dis_col_cuts[i+1]),['group']] = i
        dis_col_data[col] = dis_col_data['group']
        dis_col_bins = []
        dis_col_bins.append('nan')
        dis_col_bins.extend(['(%s,%s]' % (dis_col_cuts[i],dis_col_cuts[i+1]) for i in range(len(dis_col_cuts) - 1)])
        dis_col = dis_col_data.fillna(-1) 
        col_group = (dis_col.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'}) #按照切点分组，按dep计数       
        col_group['per'] = col_group['count'] / col_group['count'].sum() #占比
        col_group['good_num'] = col_group['count'] - col_group['bad_num'] #好客户数量
        if -1 in list(col_group[col]):
            col_group['bins'] = dis_col_bins
        else:
            col_group['bins'] = dis_col_bins[1:]
        for i in range(len(dis_col_cuts) - 1):
            if i == 0:
                self._data_new.loc[self.mydat[col] <= dis_col_cuts[i + 1],[col]] = dis_col_bins[i + 1]
            else:
                if i == len(dis_col_cuts) - 2:
                    self._data_new.loc[self.mydat[col] > dis_col_cuts[i],[col]] = dis_col_bins[i + 1]
                else:
                    self._data_new.loc[(self.mydat[col] > dis_col_cuts[i]) & (self.mydat[col] <= dis_col_cuts[i + 1]),[col]] = dis_col_bins[i + 1]
        self._data_new[col].fillna(value='nan',inplace=True)
        return col_group
       
    def nodis_group(self,col):
        """
        about:离散型变量分组
        :param col: str 变量名称
        :return:
        """
        nodis_col_data = self.mydat.loc[:,[self.dep,col]]
        is_na = pd.isnull(nodis_col_data[col]).sum() > 0 #判断是否有null
        col_group = (nodis_col_data.groupby([col],as_index=False))[self.dep].agg({'count':'count','bad_num':'sum'})
        col_group = pd.DataFrame(col_group,columns=[col,'bad_num','count'])
        if is_na:
            y_na_count = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].count()
            y_na_sum = nodis_col_data[pd.isnull(nodis_col_data[col])][self.dep].sum()
            col_group.loc[len(col_group.index),:] = [-1,y_na_sum,y_na_count] #添加到最后一行
        col_group['per'] = col_group['count'] / col_group['count'].sum()
        col_group['good_num'] = col_group['count'] - col_group['bad_num']
        if is_na:
            bins = col_group[col][:len(col_group.index) - 1]
            bins.loc[len(bins.index)] = 'nan'
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index) - 1))
            col_labels.append(-1)
            col_group[col] = col_labels
        else:
            bins = col_group[col]
            col_group['bins'] = bins
            col_labels = list(range(len(col_group.index)))
            col_group[col] = col_labels
        return col_group
    
    def woe_iv_var(self,col,adj=1):
        """
        about:单变量woe分布及iv值计算
        :param col: str 变量名称
        :param adj: float 分母为0时的异常调整
        :return:
        """
        self.check_target_binary(self.mydat[self.dep]) #检查目标值是否时二分类
        event_count,non_event_count = self.target_count(self.mydat[self.dep]) #计算坏、好客户数量
        if self.nodiscol is not None:
            if col in self.nodiscol:
                col_group = self.nodis_group(col)
            else:
                col_group = self.dis_group(col)
        x_woe_dict = {}
        iv = 0
        for cat in col_group['bins']:
            cat_event_count = col_group.loc[(col_group.loc[:,'bins'] == cat,'bad_num')].iloc[0] #分组中坏客户的数量
            cat_non_event_count = col_group.loc[(col_group.loc[:,'bins'] == cat,'good_num')].iloc[0] #分组中好客户的数量
            rate_event = cat_event_count * 1.0 / event_count #本组的坏客户/总的坏客户
            rate_non_event = cat_non_event_count * 1.0 / non_event_count #本周的好客户/总的好客户
            if rate_non_event == 0:#这个是让分子、分母都不为0
                woe1 = np.log((cat_event_count * 1.0 + adj) / event_count / ((cat_non_event_count * 1.0 + adj) / non_event_count)) #本组坏客户占比除以好客户占比
            else:
                if rate_event == 0:
                    woe1 = np.log((cat_event_count * 1.0 +adj) / event_count / ((cat_non_event_count * 1.0 + adj) / non_event_count))
                else:
                    woe1 = np.log(rate_event / rate_non_event)
            x_woe_dict[cat] = woe1
            iv += abs((rate_event - rate_non_event) * woe1)        
        return (x_woe_dict,iv)
    
    def woe_iv_vars(self,adj=1):
        """
        about:多变量woe分布及IV值计算
        :param adj:folat,分母为0时的异常调整
        :return:
        """
        X_woe_dict = {}
        iv_dict = {}
        if self.ivcol is not None: #ivcol这里也是制定批量处理的
            for col in self.ivcol:
                print(col)
                x_woe_dict,iv = self.woe_iv_var(col,adj)
                X_woe_dict[col] = x_woe_dict
                iv_dict[col] = iv
        else:
            for col in self.mydat.columns: #这里时全部处理
                print(col)
                if col != self.dep:
                    x_woe_dict,iv = self.woe_iv_var(col,adj)
                    X_woe_dict[col] = x_woe_dict
                    iv_dict[col] = iv
                    
        
        self.X_woe_dict = X_woe_dict #储存IV和WOE值
        self.iv_dict = iv_dict
        
    def apply_woe_replace(self):
        """
        about:变量woe值替换
        :return：
        """
        for col in self.X_woe_dict.keys():#这个格式是字典中带字典
            for binn in self.X_woe_dict[col].keys():
                self._data_new.loc[(self._data_new.loc[:,col] == binn,col)] = self.X_woe_dict[col][binn]
                
        self._data_new = self._data_new.astype(np.float64) #得到_data_new是所有分箱后的woe值
        
    def woe_dict_save(self,obj,woe_loc):
        """
        about:woe 字典保存
        :param obj:
        :param woe_loc:保存路径及文件名
        :return:
        """
        with open(woe_loc,'wb') as f:
            pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL) #将obj对象保存在f中，HIGHEST_PROTOCOL是整型，最高协议版本
         
    def woe_dict_load(self,woe_loc):
        """
        about: woe 字典读取
        :param woe_loc:读取路径及文件名
        :return:
        """
        with open(woe_loc,'rb') as f:
            return pickle.load(f)
        

### 逐步回归方法筛选变量
class StepWise:
    def __init__(self,X,y,logging_file,start_from =[],direction = 'FORWARD/BACKWARD',lrt=True,lrt_threshold=0.05):
        self.X = X
        self.y = y
        self.logging_file =logging_file
        self.start_from = start_from
        self.direction = direction
        self.lrt = lrt
        self.lrt_threshold = lrt_threshold

    def _likelihood_ration_test(self, ll_r, ll_f, lrt_threshold):
        test_statistics = (-2 * ll_r) - (-2 * ll_f)
        p_value = 1 - chi2(df=1).cbf(test_statistics)
        return p_value <= lrt_threshold

    def _forward(self,dataset, current_score, best_new_score, remaining, selected, result_dict, reduced_loglikelihood):
        self.logging_file.info('While loop begining current_score:%' % current_score)
        self.logging_file.info('While loop begining best_new_score:%' % best_new_score)
        current_score = best_new_score
        aics_with_candidates = {}
        p_values_ok_to_add = []
        # 选择最好的变量 to add
        for candidate in remaining:
            formula = "{}~{}".format('y', '+'.join(selected + [candidate]))
            mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            ## 只有新加指标的coefficient>0 或者 加上之后其他指标coefficient 也<0
            if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0:
                aics_with_candidates[candidate] = mod1.aic
                full_loglikehood = mod1.llf
                if self.lrt:
                    p_values_ok = self._likelihood_ration_test(reduced_loglikelihood, full_loglikehood,
                                                               self.lrt_threshold)
                    if p_values_ok:
                        p_values_ok_to_add.append(candidate)
        # 只有通过likelihood ratio test 变量才会进行AIC 进行选择
        candidate_scores = pd.Series(aics_with_candidates)
        if self.lrt:
            candidate_scores = candidate_scores.loc[p_values_ok_to_add]

        # 有变量的pvalues 显著 reject the reduced model and need to add the variable
        if not candidate_scores.empty:
            best = candidate_scores[candidate_scores == candidate_scores.min()]
            best_new_score = best.iloc[0]
            best_candidate = best.index.values[0]
        else:
            return None
        # 当加上变量的模型的AIC 比当前模型的小时，选择加上变量的模型
        if current_score > best_new_score:
            self.logging_file.info("Best Variable to Add :%s" % best_candidate)
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            improvement_gained = current_score - best_new_score
            result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'FORWARD'}
            formula = "{}~{}".format('y', "+".join(selected))
            mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            ## loglikelihood of the reduced model
            reduced_loglikelihood = mod2.llf
            self.logging_file.info("FORWAD Step：AIC=%s" % mod2.aic)
            self.logging_file.info(mod2.summary())
            return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
        else:
            return None

    def _backward(self,dataset, current_score, best_new_score, remaining, selected, result_dict,
                  reduced_loglikelihood):
        self.logging_file.info("While loop begining current_score:%s" % current_score)
        self.logging_file.info("While loop begining best_new_score:%s" % best_new_score)
        current_score = best_new_score
        aics_with_candidates = {}
        p_values_ok_to_delete = []
        for candidate in selected:
            put_in_model = [i for i in selected if i != candidate]
            formula = "{}~{}".format('y', '+'.join(put_in_model))
            mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0:
                aics_with_candidates[candidate] = mod1.aic
                reduced_reduced_loglikelihood = mod1.aic
                if self.lrt:
                    p_values_rejected = self._likelihood_ration_test(reduced_reduced_loglikelihood,
                                                                     reduced_loglikelihood, self.lrt_threshold)
                    if not p_values_rejected:
                        p_values_ok_to_delete.append(candidate)
        # 只有通过likelihood ratio test 的变量 才会进行AIC 比较进行选择
        candidate_scores = pd.Series(aics_with_candidates)
        if self.lrt:
            candidate_scores = candidate_scores.loc[p_values_ok_to_delete]
        # 有变量Pvalues 不显著 没有reject the reduced model then need to delete the variable
        if not candidate_scores.empty:
            best = candidate_scores[candidate_scores == candidate_scores.min()]
            best_new_score = best.iloc[0]
            best_candidate = best.index.values[0]
        else:
            return None

        # 当减去变量的模型AIC 比当前模型的小时，选择减去变量的模型
        if current_score > best_new_score:
            self.logging_file.info('Best Variable to Delete:%s' % best_candidate)
            remaining.append(best_candidate)
            selected.remove(best_candidate)
            improvement_gained = current_score - best_new_score
            result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'BACKWARD'}
            formula = "{}~{}".format('y', '+'.join(selected))
            mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            reduced_loglikelihood = mod2.llf
            self.logging_file.info('BACKWARD Step： AIC=%s' % mod2.aic)
            return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
        else:
            return None

    def stepwise(self):
        # 因为之后的建模的function 不接受变量名含"." 或者 "-" 所以更换名字
        new_indexs = []
        for col in self.X.columns:
            new_col = 'header_'+col.replace('.','_').replac('-','_')
            new_indexs.append(new_col)
        self.dataset = self.X.copy()
        self.dataset.columns = new_indexs
        self.dataset.loc[:,'y'] = self.y
        mapping = pd.DataFrame({'new_var_code':new_indexs,'var_code':list(self.X.columns)})
        new_start_from = mapping.loc[mapping.var_code.isin(self.start_from),'new_var_code'].tolist()
        self.logging_file.info(new_start_from)
        self.logging_file.info(mapping)

        if self.direction == "FORWARD/BACKWARD":
            remaining = set(new_indexs)
            selected = []
            result_dict = {}

            if len(self.start_from) == 0:
                init_model = smf.glm(formula = "y~1",data = self.dataset,family =sm.families.Binomial()).fit()
            else:
                formula = "{}~{}".format('y','+'.join(new_start_from))
                init_model = smf.glm(formula = formula,data = self.dataset,family = sm.families.Binomial()).fit()
                remaining = remaining - set(new_start_from)
                selected  = new_start_from
            current_score,best_new_score  = init_model.aic,init_model.aic
            reduced_loglikelihood = init_model.llf
            remaining = list(remaining)

            while current_score >= best_new_score:
                if remaining:
                    fwd_result = self._forward(self.dataset,current_score,best_new_score,remaining,selected,\
                                          result_dict,reduced_loglikelihood)
                    if fwd_result != None:
                        current_score = fwd_result[0]
                        best_new_score = fwd_result[1]
                        result_dict = fwd_result[2]
                        selected = fwd_result[3]
                        remaining  = fwd_result[4]
                        reduced_loglikelihood = fwd_result[5]
                    else:
                        self.logging_file.info('FORWARD complete and no variable selected to add')
                else:
                    fwd_result = None
                    self.logging_file.info("FORWARD has no more remaining variables to add")
                if len(selected) == 1:
                    continue
                elif len(selected) >1:
                    bkd_result = self._backward(self.dataset,current_score,best_new_score,remaining,selected,result_dict,reduced_loglikelihood)
                    if bkd_result != None:
                        current_score = bkd_result[0]
                        best_new_score = bkd_result[1]
                        result_dict = bkd_result[2]
                        selected = bkd_result[3]
                        remaining = bkd_result[4]
                        reduced_loglikelihood = bkd_result[5]
                    else:
                        self.logging_file.info('BACKWARD completed and no variable selected to delete')
                else:
                    bkd_result = None
                    self.logging_file.info("BACKWARD has no more selected variables to delete")
                if not fwd_result and not bkd_result:
                    break
        # results
        result_aic  = pd.DataFrame(result_dict).transpose().reset_index().rename(columns={'index':'new_var_code'})
        if selected:
            formula = "{}~{}".format('y','+'.join(selected))
            model = smf.glm(formula=formula,data = self.dataset,family = sm.families.Binomial()).fit()
            self.logging_file.info("===========================STEPWISE FINAL MODEL==============================")
            self.logging_file.info(model.summary())
            pvalue = model.pvalues
            self.logging_file.info('Remove insignificant variables and those with negative coefficients')
            # 即使不显著 也要保留start_from的变量
            pvalue = pvalue.drop(['Intercept']+new_start_from)
            while pvalue.max() > 0.05 or sum(model.params.loc[model.params.index != 'Intercept'] < 0) >0:
                while pvalue.max() > 0.05:
                    pvalue = pvalue.drop(pvalue.idxmax())
                    if len(pvalue) == 0:
                        break
                    formula  = "{}~{}".format('y','+'.join(pvalue.index))
                    model = smf.glm(formula = formula,data =self.dataset,family = sm.families.Binomial()).fit()
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    self.logging_file.info('Max p-values:%s,variable:%s'%(pvalue.max(),pvalue.idxmax()))
                while sum(model.params.loc[model.params.index != 'Intercept']<0)>0:
                    coefs = model.params.loc[model.params.index != 'Intercept'].copy()
                    negative_coef_variables = list(coefs.loc[coefs<0].index)
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    negative_coef_pvalue = pvalue.loc[negative_coef_variables]
                    self.logging_file.info("Max p_values:%s,variable:%s"%(negative_coef_pvalue.max(),negative_coef_pvalue.idxmax()))
                    pvalue = pvalue.drop(negative_coef_pvalue.idxmax())
                    formula = "{}~{}".format('y','+'.join(pvalue.index))
                    model = smf.glm(formula=formula,data = self.dataset,family = sm.families.Binomial()).fit()

            self.logging_file.info('SIGNIFICANT MODEL')
            self.logging_file.info(model.summary())

            final_selected = model.pvalues.drop('Intercept').index
        else:
            final_selected = selected
        result_aic = result_aic.loc[result_aic.new_var_code.isin(final_selected)]
        imp_df = pd.merge(mapping,result_aic,how='left',on='new_var_code')
        imp_df.loc[:,'final_selected'] = np.where(imp_df.new_var_code.isin(final_selected),1,0)
        imp_df = imp_df.drop('new_var_code',axis=1)
        return imp_df



class Lr_model:
    """
    about:IR相关
    """
    def __init__(self,logging_file):
        self.logging_file = logging_file
        
    def check_same_var(self,x):
        var_corr = x.corr(method='pearson',min_periods=1)
        same_var = []
        for i in x.columns:
            if var_corr[i].loc[var_corr[i] == 1].count() > 1:
                same_var.append(i)
                
        if len(same_var) > 0:
            for i in range(len(same_var)):
                self.logging_file.info('完全相同的变量如下,请进行删除：' + same_var[i])
                
        else:
            self.logging_file.info('无完全相同的变量，请进行后续操作')
            
    def in_model_var(self,in_x,model,m_type):
        if m_type == 'l1':#如果是l1则直接使用回归系数
            weight = pd.DataFrame(model.coef_).T #.coef_是回归系数,intercept_是截距
        else:
            if m_type == 'l2':#如果是l2则是回归系数的转置
                weight = pd.DataFrame(model.coef_).T
            else:
                print('type error')
        weight['index'] = weight.index
        var_namelist = in_x.T #得到X的转置
        var_namelist['varname'] = var_namelist.index
        varname = var_namelist['varname'].reset_index()
        varname['index'] = varname.index #得到变量的名
        model_tmp = pd.merge(varname,weight,how='inner',on=['index']) #将变量和回归系数结合起来
        model_var = model_tmp.drop('index',1)
        model_var.columns = ['var','coef'] #修改columns名
        var = list(model_var['var'].loc[abs(model_var['coef']) > 0]) #得到回归系数大于0的变量
        return var
        
    def lr_model_iter(self,x,y,dep,p_max=0.05,alpha=0.1,penalty='l2',method=None,
                      intercept=True,normalize=False,criterion='bic',
                      p_value_enter=0.05,f_pvalue_enter=0.05,
                      direction='both',show_step=True,criterion_enter=None,
                      criterion_remove=None,max_iter=200,lasso_penalty = "l1"):
        model = LogisticRegression(C=alpha,penalty=penalty,class_weight='balanced',max_iter=100,random_state=1)
        #建立模型，用L1线性正则化.penalty是正则化选择参数，solver是优化算法选择参数。L1向量中各元素绝对值的和，作用是产生少量的特征，
        #而其他特征都是0，常用于特征选择；L2向量中各个元素平方之和再开根号，作用是选择较多的特征，使他们都趋近于0。
        #C值的目标函数约束条件：s.t.||w||1<C，默认值是0，C值越小，则正则化强度越大。
        #n_jobs表示bagging并行的任务数
        model.fit(x,y) #开始训练模型
        select_var = self.in_model_var(x,model,'l2') #调用函数，得到回归系数大于0的变量
        xx = x[select_var] #取相应的变量
        lr_model = LogisticRegression(penalty='l2',class_weight='balanced',max_iter=100,random_state=1)
        #采用l2正则化，其实是用l1正则化选取变量，用l2正则化拿到回归系数
        lr_model.fit(xx,y)
        modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep,lasso_penalty="l1") #调用类
        modelstat.all_stats_l2() #调用类的方法
        model_output = modelstat.var_stats  #得到结果
        p = model_output['pvalue'].max() #最大的P值
        c = model_output['coef'].min()  #最小的回归系数
        while p >= p_max or c < 0:
            if p >= p_max:#如果P值较大
                max_var = model_output['var'].loc[model_output['pvalue'] == p].values[0] #得到自变量
                x = x.drop(max_var,axis=1)
                self.logging_file.info('删除P值最高的变量' + max_var + 'P值为' + str(p))
                model.fit(x,y)
                select_var = self.in_model_var(x,model,'l2')
                xx = x[select_var]
                lr_model.fit(xx,y)
                modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep,lasso_penalty="l1")
                modelstat.all_stats_l2()
                model_output = modelstat.var_stats
                p = model_output['pvalue'].max()
                c = model_output['coef'].min()
            elif c < 0: # 回归系数为负
                fu_var = model_output['var'].loc[model_output['coef'] == c].values[0]
                x = x.drop(fu_var,axis=1)
                self.logging_file.info('删除系数异常' + fu_var)
                model.fit(x,y)
                select_var = self.in_model_var(x,model,'l2')
                xx = x[select_var]
                lr_model.fit(xx,y)
                modelstat = Lasso_var_stats(xx,y,lr_model,dep=dep,lasso_penalty="l1")
                modelstat.all_stats_l2()
                model_output = modelstat.var_stats
                c = model_output['coef'].min()
                p = model_output['pvalue'].max()
        return (model_output,lr_model,select_var) #最后返回模型各个参数、模型、变量名


class Lasso_var_stats:
    """
    about:LR模型统计
    vars_vif:计算变量的vif值
    vars_contribute:计算变量的贡献度，且倒序排列
    vars_pvalue:计算变量的P值
    vars_corr:计算变量的相关系数
    all_stats:整合上面的统计指标
    """
    def __init__(self,xx,yy,lr_model,lasso_penalty,dep='y',alpha=1,solver='saga',max_iter=100,random_state=1):
        """
        :param xx
        :param yy
        :param lr_model model,建好的模型
        :param dep str,y
        :param alpha: int,lasso 中的惩罚系数，计算P值是调用statsmodels中的lasso，需要重新拟合
        :param penalty:正则项规划
        :param solver
        :param max_iter
        :param random_state
        :param n_jobs
        
        """
        self.xx = xx
        self.yy = yy
        self.dep = dep
        self.lr_model = lr_model
        self.alpha = alpha
        self.penalty = lasso_penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
            
    def model_coef(self): #和下面的一样
        weight = pd.DataFrame(self.lr_model.coef_).T
        weight['index'] = weight.index
        intercept = self.lr_model.intercept_
        var_namelist = self.xx.T
        var_namelist['varname'] = var_namelist.index
        varname = var_namelist['varname'].reset_index()
        varname['index'] = varname.index
        model_tmp = pd.merge(varname,weight,how='inner',on=['index'])
        model_output = model_tmp.drop('index',1)
        model_output.columns = ['var','coef']
        model_output = model_output.loc[abs(model_output['coef']) > 0]
        model_output['intercept'] = intercept[0]
        self.model_output = model_output.set_index(['var'],drop=False)
        modelvar = list(self.model_output['var'])
        myvar = [x for x in self.xx.columns if x in modelvar]
        train_y = pd.DataFrame(self.yy)
        train_y.columns = [self.dep]
        self.mydat = pd.merge(train_y,self.xx[myvar],left_index=True,right_index=True)
        self.vars_data = self.mydat[myvar]
        self.newmodel = LogisticRegression(C=self.alpha,penalty=self.penalty,solver=self.solver,max_iter=self.max_iter,
                                           random_state=self.random_state)
        self.newmodel.fit(self.vars_data,train_y)
        
    def vars_vif(self):
        features = ('+').join(self.vars_data.columns) #将变量连接
        X = dmatrix(features,self.vars_data,return_type='dataframe')#分数据
        vif = pd.DataFrame()
        vif['VIF_Factor'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])] #计算方差膨胀因子
        #方差膨胀因子（Variance Inflation Factor，VIF）：容忍度的倒数，VIF越大，显示共线性越严重。经验判断方法表明：当0＜VIF＜10，
        #不存在多重共线性；当10≤VIF＜100，存在较强的多重共线性；当VIF≥100，存在严重多重共线性
        vif['features'] = X.columns
        self.df_vif = vif.set_index(['features'])
    
        
    def vars_pvalue(self):
        features = ('+').join(self.vars_data.columns)
        dep_ = self.dep + ' ~'
        y,X = dmatrices(dep_ + features, self.mydat,return_type='dataframe')
        logit = sm.Logit(y,X).fit_regularized(alpha=1.0/ self.alpha)
        self.df_pvalue = pd.DataFrame(logit.pvalues.iloc[1:],columns=['pvalue'])
        
        
    def vars_corr(self):#求相关系数
        self.df_corr = pd.DataFrame(np.corrcoef(self.vars_data,rowvar=0),columns=self.vars_data.columns,index=self.vars_data.columns)
        
    def all_stats(self): # 这个和下面的基本一样
        self.model_coef()
        self.vars_vif()
        self.vars_pvalue()
        self.vars_corr()
        df_corr_trans = pd.DataFrame(self.df_corr,columns= self.vars_data.columns)
        self.var_stats = (((self.model_output.merge(self.df_pvalue,left_index=True,right_index=True)).merge(self.df_vif,left_index=True,right_index=True)).merge(df_corr_trans,
                            left_index = True,right_index=True))
        self.var_stats = self.var_stats.reset_index(drop=True)
        
    def model_coef_l2(self):
        weight = pd.DataFrame(self.lr_model.coef_).T #得到回归系数
        weight['index'] = weight.index
        intercept = self.lr_model.intercept_ #得到截距
        var_namelist = self.xx.T
        var_namelist['varname'] = var_namelist.index
        var_namelist = var_namelist['varname'].reset_index()
        varname = pd.DataFrame(var_namelist['varname']).reset_index()
        varname['index'] = varname.index
        model_tmp = pd.merge(varname,weight,how='inner',on=['index'])
        model_output = model_tmp.drop('index',1)
        model_output.columns = ['var','coef']
        model_output = model_output.loc[abs(model_output['coef']) > 0]
        model_output['intercept'] = intercept[0] #增加回归系数
        self.model_output = model_output.set_index(['var'],drop=False) #增加index
        modelvar = list(self.model_output['var'])
        myvar = [x for x in self.xx.columns if x in modelvar] #得到符合要求的变量名
        train_y = pd.DataFrame(self.yy)
        train_y.columns = [self.dep]
        self.mydat = pd.merge(train_y,self.xx[myvar],left_index=True,right_index=True) #得到符合要求变量的数据
        self.vars_data = self.mydat[myvar]
        self.newmodel = self.lr_model
        
    def vars_pvalues_l2(self):
        features = ('+').join(self.vars_data.columns)
        dep_ = self.dep + ' ~'
        y,X = dmatrices(dep_ + features,self.mydat,return_type='dataframe') #分割出X，y
        logit = sm.Logit(y,X).fit() #执行逻辑回归,另一个包
        self.df_pvalue = pd.DataFrame(logit.pvalues.iloc[1:],columns=['pvalue']) #这里求P值
        self.summary = logit.summary() #这里是统计描述所有信息
        
    def all_stats_l2(self): #调用所有的方法
        self.model_coef_l2() #用l2方法得到符合要求的变量的数据
        self.vars_vif() #检验多重共线性问题
        self.vars_pvalues_l2() #这个也是建模，用的是statsmodels.api.Logit
        self.vars_corr() #求相关系数
        df_corr_trans = pd.DataFrame(self.df_corr,columns=self.vars_data.columns)
        self.var_stats = (((self.model_output.merge(self.df_pvalue,
                           left_index=True,right_index=True)).merge(self.df_vif,left_index=True,right_index=True)).merge(df_corr_trans,
                        left_index=True,right_index=True)) #连接上述的所有表
        self.var_stats = self.var_stats.reset_index(drop=True)


class Calc_psi:
    """
    about:
    var_psi:计算单变量的psi（包括模型分组)
    vars_psi:计算多个变量的psi
    """
    def __init__(self,data_actual=None,data_expect=None):
        """
        :param data_actual:DataFrame 实际占比，即外推样本分组后的变量
        :param data_expect:DataFrame 预期占比，即建模样本分组后的变量
        """
        self.data_actual = data_actual
        self.data_expect = data_expect
        
    def var_psi(self,series_actual,series_expect):
        """
        about:psi compute
        :param series_actual:Series,实际样本分组后的变量（或者样本分组）
        :param series_expect:Series,预测样本分组后的变量（或者样本分组）
        psi计算：sum((实际占比-预期占比)*In(实际占比/预期占比))
        一般认为psi小于0.1时候模型稳定性很高，0.1-0.25一般，大于0.25模型稳定性差
        :return
        """
        series_actual_counts = pd.DataFrame(series_actual.value_counts(sort=False,normalize=True))
        #normalize 是占比,sort=False是按index排列，
        series_actual_counts.columns = ['per_1'] #实际每组的占比
        series_expect_counts = pd.DataFrame(series_expect.value_counts(sort=False,normalize=True))
        series_expect_counts.columns = ['per_2'] #预期的每组的占比
        series_counts = series_actual_counts.merge(series_expect_counts,how='right',left_index=True,right_index=True) #进行连表
        series_counts['per_diff'] = series_counts['per_1'] - series_counts['per_2'] #求差
        series_counts['per_in_ratio'] = (series_counts['per_1'] / series_counts['per_2']).apply(lambda x: log(x)) #求比值再取log
        psi = (series_counts['per_diff'] * series_counts['per_in_ratio']).sum() #差值*上面的结果，再求和
        return psi
    
    def vars_psi(self): #对所有的变量全求psi
        col_psi_dict = {}
        for col in self.data_expect.columns:
            psi = self.var_psi(self.data_actual[col],self.data_expect[col])
            col_psi_dict[col] = psi
        
        return pd.Series(col_psi_dict)



####模型可视化#################
class ModelVisualition:
    
    def __init__(self,prob_y,pred_y,y,labels= [0,1],k=20):
        self.prob_y = prob_y
        self.pred_y = pred_y
        self.y = y
        self.k = k
        self.labels = labels

    def plot_roc_curve(self,title,save_file):
        fpr,tpr,_ = roc_curve(self.y,self.prob_y)
        c_stats  = auc(fpr,tpr)
        plt.plot([0,1],[0,1],'r--')
        plt.plot(fpr,tpr,label = "ROC curve")
        s = "AUC = {:.4f} ".format(c_stats)
        plt.text(0.6,0.2,s,bbox = dict(facecolor="red",alpha=0.5))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('{}_ROC curve={}'.format(title,s)) ## 绘制ROC曲线
        plt.legend(loc = "best")
        plt.savefig(save_file)
        plt.show()
    
    def plot_model_ks(self,title,save_file):
        # 检查y是否是二元变量
        y_type = type_of_target(self.y)
        if y_type not in ['binary']:
            raise ValueError('y必须是二元变量')
        # 合并y与y_hat,并按prob_y对数据进行降序排列
        datasets = pd.concat([self.y, pd.Series(self.prob_y, name='prob_y', index=self.y.index)], axis=1)
        datasets.columns = ["y", "prob_y"]
        datasets = datasets.sort_values(by="prob_y", axis=0, ascending=True)
        P = sum(self.y)
        Nrows = datasets.shape[0]
        N = Nrows - P
        n = float(Nrows) /self.k
        # 重新建立索引
        datasets.index = np.arange(Nrows)
        ks_df = pd.DataFrame()
        rlt = {

            "tile": str(0),
            "Ptot" : 0,
            "Ntot": 0}
        ks_df = ks_df.append(pd.Series(rlt),ignore_index=True)
        for i in range(self.k):
            lo = i*n
            up = (i+1)*n
            tile = datasets.ix[lo:(up-1),:]
            Ptot = sum(tile['y'])
            Ntot = n-Ptot
            rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                "Ntot":Ntot}
            ks_df = ks_df.append(pd.Series(rlt),ignore_index=True)
    ## 计算各子集中正负例比例以及累积比例
        ks_df['PerP'] = ks_df['Ptot'] /P
        ks_df['PerN'] = ks_df['Ntot'] /N
        ks_df['PerP_cum'] = ks_df['PerP'].cumsum()
        ks_df['PerN_cum'] = ks_df['PerN'].cumsum()
        ks_df['ks'] = ks_df['PerN_cum'] - ks_df['PerP_cum']
        ks_value = ks_df['ks'].max()
        self.s = "{}_KS value is {:.4f}".format(title, ks_value)
        # 整理得出ks统计表
        ks_results = ks_df.ix[1:, :]
        ks_results = ks_results[['tile', 'Ntot', 'Ptot', 'PerN', 'PerP', 'PerN_cum', 'PerP_cum', 'ks']]
        ks_results.columns = ['子集', '负例数', '正例数', '负例比例', '正例比例', '累积负例比例', '累积正例比例', 'ks']
        # 获取ks值所在的数据点
        self.ks_point = ks_results.ix[:, ['子集', 'ks']]
        self.ks_point = self.ks_point.ix[self.ks_point['ks'] == self.ks_point['ks'].max(), :]
        # 绘制KS曲线
        ks_ax = self._ks_plot(ks_df=ks_df, ks_label='ks', good_label='PerN_cum', bad_label='PerP_cum',save_file=save_file)
        return ks_results, ks_ax

    def _ks_plot(self,ks_df, ks_label, good_label, bad_label,save_file):
        """
        middle function for ks_stats, plot k-s curve
        """
        ks_df['tile'] = ks_df['tile'].astype(np.int32)
        plt.plot(ks_df['tile'], ks_df[ks_label], "r-.", label="ks_curve", lw=1.2)
        plt.plot(ks_df['tile'], ks_df[good_label], "g-.", label="good", lw=1.2)
        plt.plot(ks_df['tile'], ks_df[bad_label], "m-.", label="bad", lw=1.2)
        # plt.plot(point[0], point[1], 'o', markerfacecolor="red",
        # markeredgecolor='k', markersize=6)
        plt.legend(loc=0)
        plt.plot([0, self.k], [0, 1], linestyle='--', lw=0.8, color='k', label='Luck')
        plt.xlabel("decilis")  # 等份子集
        plt.title(self.s)  # KS曲线图
        plt.savefig(save_file)
        plt.show()


    def plot_confusion_matrix(self,title,save_file,normalize=False,cmap = plt.cm.Blues):
        cm = confusion_matrix(self.y,self.pred_y,labels = self.labels)
        plt.imshow(cm,interpolation="nearest",cmap=cmap) ## 在指定的轴上展示图像
        plt.colorbar() # 增加色柱
        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks,self.labels,rotation=45) # 设置坐标轴标签
        plt.yticks(tick_marks,self.labels)
        if normalize:
            print("标准化混淆矩阵")
            cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        else:
            print("非标准化混淆矩阵")
            pass
        thresh = cm.max() / 2
        for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,cm[i,j],fontsize=12,horizontalalignment="center",
                     color="white"  if cm[i,j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title("{}_confusion matrix".format(title))
        plt.savefig(save_file)
        plt.show()

    def lift_lorenz(self,title,save_file):
        y_type = type_of_target(self.y)
        if y_type not in ['binary']:
            raise ValueError('y必须是二元变量')
        # 合并y与y_hat,并按prob_y对数据进行降序排列
        datasets = pd.concat([self.y, pd.Series(self.prob_y, name='prob_y', index=self.y.index)], axis=1)
        datasets.columns = ["y", "prob_y"]
        datasets = datasets.sort_values(by="prob_y", axis=0, ascending=False)
        # 计算正案例数和行数,以及等分子集的行数n
        P = sum(self.y)
        Nrows = datasets.shape[0]
        n = float(Nrows) / self.k
        # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
        datasets.index = np.arange(Nrows)
        lift_df = pd.DataFrame()
        rlt = {
            "tile": str(0),
            "Ptot": 0,
        }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
        for i in range(self.k):
            lo = i * n
            up = (i + 1) * n
            tile = datasets.ix[lo:(up - 1), :]
            Ptot = sum(tile['y'])
            rlt = {
                "tile": str(i + 1),
                "Ptot": Ptot,
            }
            lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
        # 计算正例比例&累积正例比例
        lift_df['PerP'] = lift_df['Ptot'] / P
        lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
        # 计算随机正例数、正例率以及累积随机正例率
        lift_df['randP'] = float(P) / self.k
        lift_df['PerRandP'] = lift_df['randP'] / P
        lift_df.ix[0, :] = 0
        lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
        lift_ax = self.lift_Chart(lift_df,title=title,save_file=save_file)
        lorenz_ax = self.lorenz_cruve(lift_df,title=title,save_file=save_file)
        return lift_ax, lorenz_ax

    def lift_Chart(self,df,title,save_file):
        """
        middle function for lift_lorenz, plot lift Chart
        """
        # 绘图变量
        PerP = df['PerP'][1:]
        PerRandP = df['PerRandP'][1:]
        # 绘图参数
        fig, ax = plt.subplots()
        index = np.arange(self.k + 1)[1:]
        bar_width = 0.35
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        plt.bar(index, PerP, bar_width,
                alpha=opacity,
                color='b',
                error_kw=error_config,
                label='Per_p')  # 正例比例
        plt.bar(index + bar_width, PerRandP, bar_width,
                alpha=opacity,
                color='r',
                error_kw=error_config,
                label='random_P')  # 随机比例
        plt.xlabel('Group')
        plt.ylabel('Percent')
        plt.title('{}_lift_Chart'.format(title))
        plt.xticks(index + bar_width / 2, tuple(index))
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()

    def lorenz_cruve(self,df,title,save_file):
        """
        middle function for lift_lorenz, plot lorenz cruve
        """
        # 准备绘图所需变量
        PerP_cum = df['PerP_cum']
        PerRandP_cum = df['PerRandP_cum']
        decilies = df['tile']
        # 绘制洛伦茨曲线
        plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')  # lorenz曲线
        plt.plot(decilies, PerRandP_cum, 'k-.', label='random')  # 随机
        plt.legend()
        plt.xlabel("decilis")  # 等份子集
        plt.title("{}_lorenz_cruve".format(title), fontsize=10)  # 洛伦茨曲线
        plt.savefig(save_file)
        plt.show()

    def cross_verify(self,x,y,estimators,fold,save_file,scoring="roc_auc"):
        cv_result = cross_val_score(estimator=estimators,X=x,y = y,cv =fold,n_job=-1,scoring=scoring)
        print("CV的最大AUC为{}".format(cv_result.max()))
        print("CV的最小AUC为{}".format(cv_result.min()))
        print("CV的平均AUC为{}".format(cv_result.mean()))
        plt.figure(figsize=(6,4))
        plt.title("交叉验证的评价指标分布图")
        plt.boxplot(cv_result,patch_artist=True,showmeans = True,
                    boxprops = {'color':'black','facecolor':'yellow'},
                    meanprops = {'marker':'D','markfacecolor':'tomato'},
                    flierprops ={'marker':'o','markerfacecolor':'red','color':'black'},
                    medianprops = {'linestyle':'--','color':'orange'})
        plt.savefig(self.save_file)
        plt.show()

    def plot_learning_curve(self,estimator,x,y,save_file,cv=None,
                            train_size=np.linspace(0.1,1.0,5),plt_size=None):

        """
        :param estimator: 画学习曲线的基模型
        :param x: 自变量的数据集
        :param y: 因变量的数据集
        :param cv: 交叉验证的策略
        :param train_size: 训练集划分的策略
        :param plt_size: 画图尺寸
        :return: 学习曲线
        """
        train_size,train_scores,test_scores = learning_curve(estimator=estimator,
                                            X =x,y=y,cv=cv,n_jobs =-1,train_sizes=train_size)
        train_scores_mean = np.mean(train_scores,axis=1)
        train_scores_std = np.std(train_scores,axis=1)
        test_scores_mean = np.mean(test_scores,axis=1)
        test_scores_std = np.std(test_scores,axis=1)
        plt.figure(figsize=plt_size)
        plt.xlabel('Training-example')
        plt.ylabel('score')
        plt.fill_between(train_size,train_scores_mean-train_scores_std,
                         train_scores_mean+train_scores_std,alpha = 0.1,color= 'r')
        plt.fill_between(train_size,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,
                         alpha = 0.1,color = 'g')
        plt.plot(train_size,train_scores_mean,'o-',color = 'r',label = "Training-score")
        plt.plot(train_size,test_scores_mean,'o-',color = 'g',label ="cross-val-score")
        plt.legend(loc='best')
        plt.savefig(save_file)
        plt.show()



### 变量和训练集上分数分布情况
class ModelScore:
    def __init__(self,df1,df2,target="y",k=10,bascore=600,PDO=30):
        self.bascore = bascore
        self.PDO = PDO
        self.df1 = df1
        self.df2 = df2
        self.k = k
        self.target = target

    def model_res(self):
        x = self.df1.drop(self.target,1)
        y =  self.df1[self.target]
        x_train,y_train= x,y
        logit_model = sm.Logit(y_train,sm.add_constant(x_train))
        logit_res = logit_model.fit()
        return logit_res

    def model_var_score(self):
        model_res = self.model_res()
        params = model_res.params
        print('params'.params)
        x = self.df2.drop(self.target,1)
        x = sm.add_constant(x)
        keys_list = []
        woe_list = []
        score_list = []
        A = self.bascore
        B = self.PDO/log(2)
        # 获取变量分值
        for item in list(x):
            if item == "const":
                score_v = A - B*params[item]
                keys_list.append(item)
                woe_list.append('base')
                score_list.append(score_v)
            else:
                for i in list(set(x[item])):
                    score_v = -1*B*params[item]*i
                    keys_list.append(item)
                    woe_list.append(i)
                    score_list.append(score_v)
        score_var = pd.DataFrame({'key':keys_list,'woe':woe_list,'score':score_list})
        x['prob'] = model_res.predict(x)
        x['score'] = A + B*np.log((1-x['prob'])/x['prob'])
        x['y'] = self.df2['y']
        score = x[['prob','score','y']]
        return score,score_var
    def model_cut_or_quct_score(self,df,method='cut'):
        """

        :param df: 包含预测概率prob 和 实际y
        :param method: 主要有等距切分cut和等频切分qcut
        :param k: 切分组数
        :return: 切分后分数分组的情况
        """
        df['score'] = df['prob'].map(lambda x: np.log((1-x)/x)*self.PDO/log(2)+self.bascore)
        if method == "cut":
            df['distance'] = pd.cut(df['score'],self.k,precision=0)
        if method == "qut":
            df['distance'] = pd.qcut(df['score'],self.k,precision=0)
        count = df.groupby(by='distance')[['y']].count()
        sun = df.groupby(by='distance')[['y']].sum()
        new_df = pd.merge(count,sun,left_index=True,right_index=True)
        new_df.rename(columns ={'y_x':'total','y_y':'bad'},inplace=True)
        new_df['good'] = new_df['total']- new_df['bad']
        new_df['risk'] = new_df['bad'] / new_df['total']
        new_df['total%'] = new_df['total']/new_df['total'].sum()
        new_df['good%'] = new_df['good'] / new_df['good'].sum()
        new_df['bad%'] = new_df['bad'] / new_df['bad'].sum()
        new_df['total_cum'] = new_df['total'].cumsum()
        new_df['good_cum'] = new_df['good'].cumsum()
        new_df['bad_cum'] = new_df['bad'].cumsum()
        new_df['total_cum%'] = new_df['total_cum'] / new_df['total_cum'].sum()
        new_df['good_cum%'] = new_df['good_cum'] / new_df['good_cum'].sum()
        new_df['bad_cum%'] = new_df['bad_cum'] / new_df['bad_cum'].sum()
        new_df['ks'] = np.abs(new_df['good_cum%'] - new_df['bad_cum%'])
        ## lift 累计坏/累计总/平均逾期率
        new_df['lift'] = (new_df['bad_cum']/new_df['total_cum'])/(new_df['bad'].sum()/new_df['total'].sum())
        new_df.index = new_df.index.astype(str)
        new_df.loc['total',['total','bad','good','total%','bad%','good%']] = new_df.apply(lambda x:x.sum(),axis=0)
        new_df.loc['total','ks'] = new_df['ks'].max()
        return new_df



## 模型上线后变量稳定性分析
class ModelVariableAnalysis:
    def __init__(self,score_result,df,var,id_col,score_col,bins,method):
        """

        :param score_result: 评分score的明细表 包含区间 用户数 用户占比 得分
        :param df:
        :param var: 上线样本变量得分 包含用户的id 变量的value 变量的score
        :param id_col: df中用户的id字段名
        :param score_col: df的得分字段名
        :param bins: 变量划分的区间
        return: 变量的稳定性分析表
        """
        model_var_group = score_result.loc[score_result.col == var,
        ['bin','total','totalrate','score']].reset_index(drop=True).\
            rename(columns={'total':'建模用户数','totalrate':'建模用户占比','score':'得分'})
        if method == "cut":
            df['bin'] = pd.cut(df[score_col],bins=bins,precision=0)
        if method == "qcut":
            df['bin'] = pd.qcut(df[score_col],bin=bins,precision=0)
        online_var_group = df.groupby('bin', as_index=False)[id_col].count() \
            .assign(pct=lambda x: x[id_col] / x[id_col].sum()) \
            .rename(columns={id_col: '线上用户数',
                             'pct': '线上用户占比'})
        var_stable_df = pd.merge(model_var_group,online_var_group,on='bin',how='inner')
        var_stable_df = var_stable_df.iloc[:,[0,3,1,2,4,5]]
        var_stable_df['得分'] = var_stable_df['得分'].astype('int64')
        var_stable_df['建模样本权重'] = np.abs(var_stable_df['得分']* var_stable_df['建模样本用户占比'])
        var_stable_df['线上样本权重'] = np.abs(var_stable_df['得分']*var_stable_df['线上用户占比'])
        var_stable_df['权重差距'] = var_stable_df['线上样本权重'] - var_stable_df['建模样本权重']
        return var_stable_df














































