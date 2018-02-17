

# 		            Lab1 Report

​																Number:15307130224

​																		Name: 佘国榛



### Note

1. I use the octave in the first part of lab, and use the python(version=3.5) in the second part of lab.I will clarify it in the following report.

2. The raw code` test.py` can run and generate graph, but the code snippet in the report can't.

3. The doc of code can be found in documentation.html

   ​

   ​

### 1.1 Adding a Bias Variable

​	In this part, we should take the bias into consideration, compared with the previous method. 


- #### Methodsshuffle	We can add a new parameter$w_0$ in $w^T$, which can take the bias into consideration. To facilitate the caculation, we can add a new column into the previous matrix **A**, without the side effect of other part of the function.

- #### Critical codes(Octave)

  ```matlab
  function [model] = leastSquaresBias(X,y)

  % Take bias into consideration
  testCal = [(linspace(1,1,size(X,1)))', X]
  w = (testCal'*testCal)\testCal'*y

  model.w = w;
  model.predict = @(model,X) predict(model, [(linspace(1,1,size(X,1)))', X])

  end
  ```

  ##### Algorithm skims

  1. The way to add a new line in matlab 

     ```
     testCal = [(linspace(1,1,size(X,1)))', X]
     ```

  2. The way to wrapper the predict function 

     ```
     model.predict = @(model,X) predict(model, [(linspace(1,1,size(X,1)))', X])
     ```

     With this functional programming tricks, we can preserve the interface of the leastSquaresBias


- #### Experimental analysis

  - ##### Error

    training Error:3553.3

    test Error:3393.9

    ​


-   ##### Graph

    Raw:

     ![RawImage](graph/RawImage.png)

    ![SingleParRegression](graph/SingleParRegression.png

  ​

  ​	New:

  ​	 ![SingleParRegression](graph/SingleParRegression.png)

  ​

- ##### Analysis

  We can tell from the graph and data that the trimmed model fits the data with high accuracy.Further more，we can find that **the slope do not change with the change of model**, in other word, we just move the line parrallel. This can be proved thereoticaly via the independency of matrix caculation.

  ​	

  From the new test error and training error, we can verify our assumption that the inserted parameter(bias) build a better model.









### 2.1 Polynomial Basis

In this part, we will talk about the polynomial basis, which will build a model with more features, build we should also realize the fact that the "overfitting" will occur with the complexity of the polynomial basis. This will be showed in the following part:

- #### Algorithm skims

  Code Style : OOP(A naive attemption)

  I use the OOP style of code aimed to generetae reusage code, The main class is the following:

  [CommonMethod]()

  [DataSet]()

  [PolyNomialRegression]()

  [RidgeRegression]()

  [TestPolyNomial](l)

  [TestRidge]()

  Further information(Like nested function) can be found in the [documentation](poly.html) in the same directory.

  ​

- #### Critical Section of leastSquaresBasis

  - Code

```python
    def generateMatrix(x_matrix, deg):
        """
        可用输入和最高次数生成计算矩阵
        :param x_matrix: 输入矩阵，
        :param deg: 多项式次数
        :return: 多项式拟合后的矩阵

        """
        ret = []
        for xline in x_matrix:
            row = []
            try:
                for x in xline:
                    for index in range(1, deg + 1):
                        row.append(x ** index)

                ret.append(row)
            except:
                ##如果只有单feature,不可迭代
                return np.array([[x ** i for i in range(0, deg + 1)] for x in x_matrix])

        return ret
    
    
   def leastSquaresBias(self, x_vec, y_vec, deg):
        """
        求出最优解
        :param x_vec: 输入向量
        :param y_vec: 输出向量
        :param deg: 维数
        :return: w:最优解
        """
        generatedmatrix = CommonMethod.generateMatrix(x_vec, deg)

        w = np.dot(np.linalg.inv(generatedmatrix.T @ generatedmatrix), generatedmatrix.T) @ y_vec

        return w
```

-    Explain

     I use the function $w^t = (A^TA)^{-1}A^Ty $  to generate the model, in this function. I also use another sub function`generateMatrix`,which can generate the $A$ in the fomula according to the data input.

-    #### Experimental analysis

     - ##### note

       I draw the graph for deg range from 0 to 20,aimed to get a more genral conclusion

     - ##### data analysis

       I use the `TestPolyNomial` class to test the generate model with the change of degree

       And I got the data illustrated in the table

     | DEG  | Training Error | Test Error    |
     | ---- | -------------- | ------------- |
     | 0    | 15480.519782   | 14390.763     |
     | 1    | 3551.34587066  | 3393.86909807 |
     | 2    | 2167.99194338  | 2480.72539869 |
     | 3    | 252.046100863  | 242.804943781 |
     | 4    | 251.4615531    | 242.126426754 |
     | 5    | 251.143461824  | 239.544855064 |
     | 6    | 248.582814359  | 246.005401837 |
     | 7    | 247.01104707   | 242.887593376 |
     | 11   | 235.021591096  | 255.794440925 |
     | 15   | 219.535408153  | 417.989650185 |
     | 20   | 193.88452453   | 641.835357593 |

     - ##### Graph

      ![PolynomialRegression](graph/PolynomialRegression.png)

     - ##### Analysis

       From the chart above, we can see that the training data error is declined with the increasing of degree, but the test data error is slight diffrent, it hit the buttom at some point, and then bounce back, this phenomenon can verify the "overfitting".

       From the graph above, we can tell that the simulating curve fit the training better as the degree is increasing, and the overfitting phenomonon can be witness in the graph.

       ​

       ### 2.2 Preprocessing – Data standardization

       - #### Algorithm skims

         - To read the data in txt, I make a class `Dataset`, which can load the data into his own member,
         - And to divide the data and cross validation, I use a method `RidgeRegression.divide`
         - To get the result of standardization, I use a common method(for reusability) to generate 6 parameter (x_means, y_mean, stds_x, std_y, ret_matrix.T, ret_y)
         - $\theta = (A^TA+\lambda I)^{-1}A^Ty$

       - #### Critical Section of Code

         ##### standardization

         ```python
             def standardize(matrix):
                 """

                 :param matrix: 需要标准化的平均值
                 :return: (x平均值(数组)，y平均值，x标准差(数组)， y标准差，标准化之后的x（矩阵），标准化之后的y（数组）)
                 """
                 matrix_tr = matrix.T

                 feature_number = len(matrix_tr)

                 x_means = []
                 stds_x = []

                 ret_matrix = []

                 for i in range(0, feature_number - 1):
                     mean = np.mean(matrix_tr[i])
                     x_means.append(mean)

                     std_x = np.std(matrix_tr[i])
                     stds_x.append(std_x)

                     ##坑死人啊
                     if std_x == 0:
                         std_x = 1

                     ret_matrix.append(list(map(lambda x: (x - mean) / std_x, matrix_tr[i])))
                     y = matrix_tr[feature_number - 1].T
                      y_mean = np.mean(y)

                      std_y = np.std(y)

                      ret_y = list(map(lambda x: (x-y_mean), y))

                      return (x_means, y_mean, stds_x, std_y, np.array(ret_matrix).T, ret_y)
         ```



​       
​    
​        
​    
         ```python
             def getSolution(self):
                 """
                 获取bias，theta
                 :return:
                 """
                 I = np.eye(self.data_set.features_count)
    
                 step_1 = np.dot(self.standardize_training_matrix.T, self.standardize_training_matrix)
                 step_2 = self.lambda_current * I + step_1
                 step_3 = np.linalg.inv(step_2)
                 self.w = step_3 @ self.standardize_training_matrix.T @ self.training_y
                 self.bias = self.training_y_mean - self.training_x_means @ self.w
                 return self.w




-   #### Experimental analysis

    - #### How model change with $\lambda$ 

       ![Theta_with_Omega](graph/Theta_with_Omega.png)

      In this part, we can see that the each parameter in model seems to be undermined as the $\lambda$ increase. And the Infinite situation will be zero for all the parameter.

      - Code

        #### note:

        The code just a small part of the all class, to generate the graph, you should run the source file.

        ```python
            def testW(self):
                """
                检测w随lambda的变化情况
                :return:
                """
                self.draw_pic = []

                for name in self.data_set.features_name:
                    self.draw_pic.append({
                        "name": name,
                        "w": []
                    })
        ```


                w_range = np.linspace(-1,5,50)
                training_errors = []
                test_errors = []
                for index in w_range:
                    regress = RidgeRegression(self.data_set, 10**index)
                    regress.shuffle()
                    regress.divide(50, 97)
                    regress.run()



                    training_errors.append(regress.training_error)
                    test_errors.append(regress.test_error)


                    for index, ha in enumerate(self.draw_pic):
                        ha["w"].append(regress.w[index])


                plt.figure()

                for index, ll in enumerate(self.draw_pic):
                    plt.plot(w_range, ll["w"], label=ll['name'])
    
                plt.legend(loc="upper right")
                plt.savefig("graph/Theta_with_Omega.png")
                plt.show()


                plt.figure()

                plt.xlabel("Theta^K")
                plt.plot(w_range, training_errors, label="training")
                plt.plot(w_range, test_errors, label="test")
    
                plt.legend(loc="upper right")
                plt.savefig("graph/Find_Theta.png")
    
                plt.show()
               
        ```
    
    - #### Test error and training error
    
      - Code
    
        random function(use random number to shuffle the matrix)
    
        ```python
            def shuffle(self):
                """
                随即打乱矩阵
                :return:
                """
                index =[]
                matrix = self.data_set.data_matrix[:]  ##深拷贝
                for i in range(0, len(matrix)):
                    index.append(i)
    
                # random.shuffle(index)


                for (i,j) in enumerate(index):
                    self.data_set.data_matrix[i] = matrix[j]
        ```
    
        function
    
        ```
            def testW(self):
                """
                检测w随lambda的变化情况
                :return:
                """
                self.draw_pic = []
    
                for name in self.data_set.features_name:
                    self.draw_pic.append({
                        "name": name,
                        "w": []
                    })


                w_range = np.linspace(-1,5,50)
                training_errors = []
                test_errors = []
                for index in w_range:
                    regress = RidgeRegression(self.data_set, 10**index)
                    regress.shuffle()
                    regress.divide(50, 97)
                    regress.run()



                    training_errors.append(regress.training_error)
                    test_errors.append(regress.test_error)


                    for index, ha in enumerate(self.draw_pic):
                        ha["w"].append(regress.w[index])


                plt.figure()

                for index, ll in enumerate(self.draw_pic):
                    plt.plot(w_range, ll["w"], label=ll['name'])
    
                plt.legend(loc="upper right")
                plt.savefig("graph/Theta_with_Omega.png")
                plt.show()


                plt.figure()

                plt.xlabel("Theta^K")
                plt.plot(w_range, training_errors, label="training")
                plt.plot(w_range, test_errors, label="test")
    
                plt.legend(loc="upper right")
                plt.savefig("graph/Find_Theta.png")
    
                plt.show()


​    
      - training_error      test_error

      - 0.445334853518 0.620342741928
        0.304861398402 21.4553351088
        0.262982056507 41.6357619217
        0.287857548324 35.0617982762
        0.409290947046 2.03267778646
        0.342065360759 20.161183167
        0.445336251203 0.620126866669
        0.342095576644 20.0085956863
        0.445385618458 0.617538987274
        0.304918818892 21.2890781326
        0.263039629756 41.2437726428
        0.287909564773 34.5672920658
        0.409342484428 2.00461035413
        0.342119223261 19.9444584365
        0.445424056412 0.616495223665
        0.304961952765 21.2246789724
        0.263082866197 41.0928624344
        0.28794892706 34.3764185432
        0.409381399761 1.99387361096
        0.342159840439 19.8608657569
        0.4454898176 0.615173671141
        0.305035323209 21.1408917796
        0.263156383605 40.8973881425
        0.288016239388 34.1288506894
        0.409447838332 1.98002924127
        0.342229120427 19.7524667048
        0.296998033209 29.6356938218
        0.323172668626 21.3354455945
        0.441927505019 1.41614656674
        0.376802392456 13.6573123932
        0.493415479447 0.609503848392
        0.354575902683 14.0656468879
        0.980164540667 1.01385715002
        0.977634359717 1.02687141808
        0.978518979461 0.914610160671
        0.982376289684 0.995455016344
        0.981136871693 0.988159302928
        0.990132152815 0.992415805723
        0.984940361566 1.00963451478
        0.983017224404 1.01848325116
        0.983688205288 0.933538785568
        0.993782895377 0.995021793153
        0.996768159795 0.997515127751
        0.995057715961 1.00255303181
        0.99442510362 1.0047480988
        0.994644857695 0.977067718447
        0.995606150965 0.998818268873
        0.995300397607 0.996140449136
        0.997558845348 0.998122977192
    
      - graph
    
          ![Cross_Validation](graph/Cross_Validation.png)
    
      - 	Training Error       Test Error
    
        0.495040114028 1.09436167374
        0.495043986647 1.09355071068
        0.495050751125 1.09248206357
        0.495062543655 1.09107657416
        0.495083048333 1.08923276438
        0.495118580439 1.08682199667
        0.495179879921 1.08368366064
        0.495285022263 1.07962132725
        0.495464018105 1.07440151593
        0.495765821505 1.06775765554
        0.496268483713 1.05940284366
        0.497092844675 1.0490556977
        0.498419093273 1.03648312486
        0.500503383208 1.02156103379
        0.503688457766 1.00434782102
        0.508398918554 0.985156095621
        0.515110937201 0.964598447833
        0.524291414442 0.943579180728
        0.536314706089 0.923212850236
        0.55138224128   0.904674842845
        0.569480712063 0.889021352755
        0.590405493644 0.877037444207
        0.613844905358 0.869163575789
        0.639482833634 0.86551172621
        0.667058279802 0.865936037355
        0.696339024641 0.870102661966
        0.727017591352 0.877523316758
        0.758591713755 0.887559673512
        0.790311085531 0.899436697208
        0.821238233904 0.912298370706
        0.85040412122   0.925305241524
        0.876986844927 0.937740266235
        0.90043913259   0.949083125355
        0.920529235413 0.95903281478
        0.937305105237 0.96748402472
        0.951016168211 0.974477505519
        0.962027172942 0.980145150104
        0.970745914402 0.984663179795
        0.977572924628 0.988218616386
        0.982872157196 0.990988682649
        0.986957729948 0.993130275662
        0.990091205366 0.994776216542
        0.992484865499 0.996035515007
        0.994307805882 0.996995685188
        0.995692879701 0.99772586845
        0.996743405262 0.998280052436
        0.997539122831 0.9987000274
        0.998141226125 0.99901793374
        0.998596476149 0.999258370777
        0.998940490959 0.999440098612


      - ### Cross Validation

        - Code

          divide function

          ```python
              def divide(self, start, end):
                  """
                  分离训练集和测试集
                  :param num: 训练集个数
                  :return:
                  """
                  self.test_matrix = self.data_set.data_matrix[start:end]
                  A1 = list(self.data_set.data_matrix[0:start])[:]
                  A2 = list(self.data_set.data_matrix[end:len(self.data_set.data_matrix)])[:]
                  self.training_matrix = np.array(A1 + A2)[:]
          ```
    
          crossValidation function:
    
        ```python
            def crossValidation(self, k):
                """
                K分类交叉检验
                :param k: 参数
                :return:
                """
    
                average_training_errors = []
                average_test_errors = []
    
                w_range = np.linspace(-1,5,50)
                for index in w_range:
    
                    regress = RidgeRegression(self.data_set, 10**index)
                    regress.shuffle()
    
                    test_size = int(len(self.data_set.data_matrix)/k)  ##测试集大小
    
                    training_error_sum = 0
                    test_error_sum = 0
    
                    for i in range(0, k):
                        regress.divide(i * test_size, (i+1) * (test_size))
                        regress.run()
    
                        training_error_sum += regress.training_error
                        test_error_sum += regress.test_error
    
                    regress.average_training_error = training_error_sum / k
                    regress.average_test_error = test_error_sum / k
    
                    average_training_errors.append(regress.average_training_error)
                    average_test_errors.append(regress.average_test_error)


                    print("theta={0}".format(10**index))
                    print("average_training{0}".format(regress.average_training_error))
                    print("average_test{0}".format(regress.average_test_error))
    
                plt.figure()
    
                plt.plot(w_range, average_training_errors, label="training")
                plt.plot(w_range, average_test_errors, label="test")
    
                plt.legend(loc="upper right")
                plt.savefig("graph/Cross_Validation.png")
    
                plt.show()
        ```
    
        ​
    
      - Graph
    
         ![Find_Theta](graph/Find_Theta.png)
    
        ​

-   #### Analysis

    If we choose the first 50 data as training group, and the remaining data as the test group. We will get the figure as same as the FIgure1. 

    But the important thing is that the second graph is not consistent with the expected graph. At the first glance, it seems weird, but from another respect, I think my answer do make sense. From the perspective of trend of error——test error will hit a bottom(which means the optimal answer), and then bounce off(means deviation arising from overly regularization), and training error will increase consistently. This graph just hit the point.

    Now I can tell from the picture that when minimized the test error, the corresponding x value will be the optimal $\lambda = 97.1$


### Discussion of proposed method

1. I am still confused about how to handle the bias in regularization model. I think there are a bunch of  methods to add bias, but the experiment shows that they differ with each other sharply. So my question is : Is there any best practice in regularization model, or you should tell by the actual situation.
2. Second question is about how to do the **cross validation**, we find that a lot of methods in cross validation(I choose the K cross validation)

