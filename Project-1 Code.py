#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import scipy.io
import math
import geneNewData
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    myID='2289' #your ID here
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    ############ Task 1 ############
    feature1_avg_arr_0=[]
    feature1_avg_arr_1=[]
    feature2_std_arr_0=[]
    feature2_std_arr_1=[]
    for i in range(len(train0)):
             feature1_avg_arr_0.append(numpy.average(train0[i]))
             feature2_std_arr_0.append(numpy.std(train0[i]))
    for i in range(len(train1)):
             feature1_avg_arr_1.append(numpy.average(train1[i]))
             feature2_std_arr_1.append(numpy.std(train1[i]))
    
    ############ Task 2 ############
    parameters=[]
    parameters.append(numpy.mean(feature1_avg_arr_0))
    parameters.append(numpy.var(feature1_avg_arr_0))
    parameters.append(numpy.mean(feature2_std_arr_0))
    parameters.append(numpy.var(feature2_std_arr_0))
    
    parameters.append(numpy.mean(feature1_avg_arr_1))
    parameters.append(numpy.var(feature1_avg_arr_1))
    parameters.append(numpy.mean(feature2_std_arr_1))
    parameters.append(numpy.var(feature2_std_arr_1))
    print(parameters)
    
    
    ############ Task 3 ############
    #As per given note
    feature_tst_0=[]
    feature_tst_1=[]
    for i in range(len(test0)):
             feature_tst_0.append([numpy.average(test0[i]),numpy.std(test0[i])])
    for i in range(len(test1)):
             feature_tst_1.append([numpy.average(test1[i]),numpy.std(test1[i])])
    
    prior0=prior1= 0.5
    prob_0 = 0
    prob_1=0
    
    pred_list0=[]
    for i in range(len(feature_tst_0)):
        prob_0=prior0*prob(feature_tst_0[i][0],parameters[0],parameters[1])*prob(feature_tst_0[i][1],parameters[2],parameters[3])
        prob_1=prior1*prob(feature_tst_0[i][0],parameters[4],parameters[5])*prob(feature_tst_0[i][1],parameters[6],parameters[7])
        
        if prob_0>prob_1:
            pred_list0.append(1)
        else:
            pred_list0.append(0)
    
    pred_list1=[]
    for i in range(len(feature_tst_1)):
        prob_0=prior0*prob(feature_tst_1[i][0],parameters[0],parameters[1])*prob(feature_tst_1[i][1],parameters[2],parameters[3])
        prob_1=prior1*prob(feature_tst_1[i][0],parameters[4],parameters[5])*prob(feature_tst_1[i][1],parameters[6],parameters[7])
        
        if prob_0>prob_1:
            pred_list1.append(0)
        else:
            pred_list1.append(1)
            
    ############ Task 4 ############

    accuracy0=sum(pred_list0)/len(feature_tst_0)
    accuracy1=sum(pred_list1)/len(feature_tst_1)
    print(accuracy0,accuracy1)
    
    
    
    
def prob(val1,val2,val3):
        p1=1/(numpy.sqrt(2*3.14*val3))
        exponent_term = numpy.exp((-1/2)*(numpy.power((val1-val2),2)/val3))
        return p1*exponent_term
    

    
if __name__ == '__main__':
    main()


# In[ ]:




