import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class CommonMethod:
    """
    常用方法类
    """
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
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


class DataSet:
    """
    数据集，提供matlab文件和txt文件读入
    """
    def __init__(self, filepath):
        """
        初始化读入文件
        :param filepath: 文件名
        """
        self.filepath = filepath
        suffix = self.filepath.split(".")[-1]
        print(suffix)
        if suffix == 'mat':
            self.readFromMat()
        if suffix == 'txt':
            self.readFromTxt()
        else:
            print("Unkown")

    def readFromTxt(self):
        """
        从txt读入数据
        """

        with open(self.filepath) as fp:
            names = fp.readline().strip('\n').split("\t")

            self.features_name = names[0:-1]
            self.result_name = names[-1]
            self.features_count = len(self.features_name)

            str_data_list = fp.readlines()
            self.data_matrix = np.array(list(map(lambda line: list(map(float, line.strip('\n').split("\t"))),
                                                 str_data_list)))

    def readFromMat(self):
        """
        从matlab文件读入数据
        """
        self.raw_data = sio.loadmat(self.filepath)
        self.X = self.raw_data['X']
        self.y = self.raw_data['y']
        self.Xtest = self.raw_data['Xtest']
        self.ytest = self.raw_data['ytest']
        self.amount = len(self.y)


class PolyNomialRegression():
    """
    多项式回归类
    """
    def __init__(self, data_set, deg):
        """

        :param data_set: 数据集
        :param deg: 多项式次数
        """
        # super.__init__(self)
        self.x = data_set.X[:, 0]
        self.y = data_set.y[:, 0]
        self.xtest = data_set.Xtest[:, 0]
        self.ytest = data_set.ytest[:, 0]
        self.amount = data_set.amount
        self.deg = deg

    def run(self):
        """
        启动回归算法，由Test类调用
        """
        # print("deg={0}".format(self.deg))
        self.w = self.leastSquaresBias(self.x, self.y, self.deg)

        self.validateTrainingSet()
        self.validateTestSet()

        print(self.deg, self.trainingError, self.testError)

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

    def validateTrainingSet(self):
        """
        计算训练集误差
        """
        generatedmatrix = CommonMethod.generateMatrix(self.x, self.deg)
        self.trainingError = self.calculateError(generatedmatrix, self.w, self.y)

    def validateTestSet(self):
        """
        计算测试集误差
        """
        generatedmatrix = CommonMethod.generateMatrix(self.xtest, self.deg)
        self.testError = self.calculateError(generatedmatrix, self.w, self.ytest)

    @staticmethod
    def calculateError(matrix, w, y):
        """
        静态方法，计算测试error
        :param matrix: 测试集（矩阵）
        :param w: 返回值
        :param y: 预测值（矩阵）
        :return: error
        """
        return (y - matrix @ w).T @ (y - matrix @ w) / len(matrix)

    def drawScatter(self):
        """
        画散点图
        :return:
        """
        plt.scatter(self.x, self.y, s=0.1, c='blue')

    def drawCurve(self):
        """
        画曲线
        :return:
        """
        x = np.arange(min(self.x), max(self.x), 0.1)

        y = CommonMethod.generateMatrix(x, self.deg) @ self.w

        plt.plot(x, y, color='red')


class TestPolyNomial:
    """
    测试多项式类
    """
    def __init__(self, deg):
        """

        :param deg: 系数
        """
        self.deg = deg
        self.testErrors = np.array([])
        self.trainingErrors = np.array([])

    def test(self, data_set):
        """
        启动测试
        :param data_set:输入数据
        :return:
        """
        plt.figure(figsize=(6, 5))
        for index in range(0, self.deg + 1):
            regress = PolyNomialRegression(data_set, index)
            regress.run()
            plt.subplot(5, 5, index + 1)

            self.testErrors = np.append(self.testErrors, regress.testError)
            self.trainingErrors = np.append(self.trainingErrors, regress.trainingError)

            regress.drawScatter()
            regress.drawCurve()

            textCoordinateX = min(regress.x)
            textCoordinateY = max(regress.y)

            plt.text(x=textCoordinateX, y=textCoordinateX, s="N={0}".format(index), fontdict={'fontsize': 10})
            plt.text(x=textCoordinateX, y=textCoordinateY * 0.8, s="tr:{0}".format(int(regress.trainingError)),
                     fontdict={'fontsize':
                                   10})
            plt.text(x=textCoordinateX, y=textCoordinateY * 0.65, s="te:{0}".format(int(regress.testError)),
                     fontdict={'fontsize': 10})

        plt.savefig("graph/PolynomialRegression.png")
        plt.show()

        plt.close()
        self.analysis()

    def analysis(self):
        """
        分析结果与画图
        :return:
        """
        print(np.arange(1, self.deg + 1, 1))
        print(self.testErrors)

        plt.figure()
        plt.title("Test Error vs training Error")
        plt.xlabel('deg')
        plt.ylabel('error')
        plt.plot(np.arange(0, self.deg + 1, 1), self.testErrors, marker='o', label='test')
        plt.plot(np.arange(0, self.deg + 1, 1), self.trainingErrors, marker='x', label='training')
        plt.legend(loc="upper right")

        plt.savefig("graph/ErrorviaDeg.png")
        plt.show()
        plt.close()


class TestRidge:
    def __init__(self, data_set):
        """
        岭回归测试类
        :param data_set: 测试数据
        """
        self.data_set = data_set

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
            # regress.shuffle()

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


            # print("theta={0}".format(10**index))
            # print("average_training{0}".format(regress.average_training_error))
            # print("average_test{0}".format(regress.average_test_error))
            print(regress.average_training_error, regress.average_test_error)

        plt.figure()

        plt.plot(w_range, average_training_errors, label="training")
        plt.plot(w_range, average_test_errors, label="test")

        plt.legend(loc="upper right")
        plt.savefig("graph/Cross_Validation.png")

        plt.show()





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
            # regress.shuffle()
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







class RidgeRegression():
    """
    岭回归类
    """
    def __init__(self, data_set, lambda_current):
        # super.__init__(self)
        """
       初始化
        :param data_set: 数据集
        :param lambda_current: 当前lambda
        """

        self.data_set = data_set
        self.feature_number = self.data_set.features_count
        self.lambda_current = lambda_current




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




    def preprocessing(self):
        """
        预处理，获取标准化之后的数据集
        :return:
        """
        (self.training_x_means, self.training_y_mean, self.training_stds_x, self.training_std_y, self.standardize_training_matrix, self.training_y) = CommonMethod.standardize(self.training_matrix)
        (self.test_x_means, self.test_y_means, self.test_std_x, self.test_std_y, self.standardize_test_matrix, self.test_y) = CommonMethod.standardize(self.test_matrix)

    def run(self):
        """
        运行
        :return:
        """
        self.preprocessing()
        self.getSolution()
        self.test()

    def test(self):
        """
        测试，由运行调用
        :return:
        """
        self.validateTrainingSet()
        self.validateTestSet()

        # print("test error {0}".format(self.test_error))
        # print("training error {0}".format(self.training_error))

        print(self.training_error, self.test_error)


    def validateTestSet(self):
        """
        验证测试集合
        :return:
        """
        step1 = self.standardize_test_matrix @ self.w
        step2 = step1 - self.test_y
        self.test_error = np.dot(step2, step2)/np.dot(self.test_y, self.test_y)



    def validateTrainingSet(self):
        """
        验证训练集
        :return:
        """
        step1 = self.standardize_training_matrix @ self.w
        step2 = step1 - self.training_y
        self.training_error = np.dot(step2.T, step2)/np.dot(self.training_y,self.training_y)


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


if __name__ == '__main__':

    # """
    # 1.1
    # """
    # data_set_1 = DataSet("./data/basisData.mat")
    #
    # test1 = TestPolyNomial(20)
    # test1.test(data_set_1)


    data_set_2 = DataSet("./data/prostate.data.txt")

    """
    2.1
    """
    test2 = TestRidge(data_set_2)
    test2.crossValidation(5)

    """
    2.2
    """
    # test2 = TestRidge(data_set_2)
    # test2.testW()