from visdom import Visdom
import numpy as np


class Display(object):
    """
    功能：显示训练过程中的loss变化，验证过程中的准确率变化等曲线图。
    参数：--env--面板名称localhost::8097可访问
            --state--'TRAIN' 'VAL' 'T-V' 'ValTrain' 决定创建的窗口样式和数量
            --window_tag--窗口名称
    """

    def __init__(self, envir, state, window_tag):

        self.viz = Visdom(env=envir)

        if state == 'TRAIN':
            self.train_display = TrainDisplay(self.viz, window_tag)
        else:
            raise Exception("Error: Visdom state error!")

    def __call__(self, phase, X, Y):
        """
        时间：2019.04.23
        功能：更新曲线
        参数：--phase--'TRAIN' 'VAL'训练或验证过程
        """
        if phase == 'TRAIN':
            self.train_display(X, Y)  # X：迭代次数 Y：Loss
        else:
            raise Exception("Error: Visdom phase error!")


class TrainDisplay(object):
    """
    时间：2019.04.23
    功能：训练过程中损失曲线
    """

    def __init__(self, envir, window_tag):
        self.env = envir
        self.train_line = self.env.line(
            X=0.1 * np.ones(1),
            Y=0.1 * np.ones(1),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title=('Loss:' + window_tag),
            )
        )

    def __call__(self, Iteration, Loss):
        """
        时间：2019.04.24
        功能：更新曲线
        参数：--Iteration--迭代次数
                --Loss--损失值
        """
        Loss = 1 if Loss > 1 else Loss  # loss显示不超过1
        self.env.line(
            X=np.array([Iteration]),
            Y=np.array([Loss]),
            win=self.train_line,
            update='append')


class ValDisplay(object):
    """
    时间：2019.04.23
    功能：验证过程曲线绘制
    """

    def __init__(self, envir, window_tag):
        self.env = envir
        self.val_line = self.env.line(
            X=np.zeros(1),
            Y=np.zeros((1, 4)),
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title=('Accuracy:' + window_tag),
                legend=['AP', 'WP', 'NC', 'NLH'],
            )
        )

    def __call__(self, Epoch, Precision_Dict):
        """
        时间：2019.04.23
        功能：更新曲线
        参数：--Epoch--
                --Precision_Dict-- 准确率字典{'AP':0,'WP':0,'NC':0,"NLH":0}
        """
        precision = list((Precision_Dict.values()))
        self.env.line(
            X=np.array([Epoch]),
            Y=np.array([precision]),
            win=self.val_line,
            update='append')