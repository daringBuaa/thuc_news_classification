# thuc_news_classification
利用朴素贝叶斯的原理对thuc_news分类
由于样本量过大，只选取了每类2000条数据进行分类，运行结果如下
模型score ： 0.846285714286
             precision    recall  f1-score   support

         体育       0.91      0.93      0.92       738
         娱乐       0.84      0.76      0.80       744
         家居       0.82      0.82      0.82       749
         彩票       0.92      0.96      0.94       761
         房产       0.83      0.90      0.87       733
         教育       0.83      0.91      0.87       747
         时尚       0.89      0.84      0.87       758
         时政       0.81      0.81      0.81       745
         星座       0.83      0.97      0.89       749
         游戏       0.91      0.87      0.89       755
         社会       0.85      0.60      0.71       773
         科技       0.86      0.77      0.81       751
         股票       0.73      0.81      0.77       748
         财经       0.84      0.91      0.87       749

avg / total       0.85      0.85      0.84     10500
