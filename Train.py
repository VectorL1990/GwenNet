import os
import torch
import struct
import numpy as np
import pickle
import time
from PytorchNet import PolicyValueNet

class Trainer:
    def __init__(self, init_model=None):
        self.learn_rate = 0.001
        self.lr_multiplier = 1
        self.batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.015
        self.game_num = 1
        self.totalGameStatesBatch = []
        self.totalActionProbsBatch = []
        self.totalGameWinScoresBatch = []
        self.train_loop = 0


        try:
            # 检查文件是否存在
            if init_model and not os.path.exists(init_model):
                print(f"警告: 模型文件不存在: {init_model}")
                print("将使用新模型初始化")
                self.policy_value_net = PolicyValueNet()
            else:
                # 创建 PolicyValueNet 并加载模型
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print("模型加载成功")
        
        except FileNotFoundError as e:
            # 文件不存在异常
            print(f"错误: {e}")
            print("将使用新模型初始化")
            self.policy_value_net = PolicyValueNet()
        
        except Exception as e:
            # 其他加载问题
            print(f"加载模型时发生未知错误: {e}")
            print("将使用新模型初始化")
            self.policy_value_net = PolicyValueNet()

        #try:
        #    self.policy_value_net = PolicyValueNet()
        #    self.policy_value_net.load_model(model_file = init_model)
        #except:
        #    print("can not find previous model.pt")
        #if init_model:
        #    try:
        #        self.policy_value_net = PolicyValueNet(model_file = init_model)
        #    except:
        #        print("can not find previous model.pt")
                #self.policy_value_net = PolicyValueNet()
        #else:
        #    self.policy_value_net = PolicyValueNet()

    def safe_explained_variance(self, y_true, y_pred):
        """
        安全计算解释方差，处理目标值方差为零的情况
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        var_true = np.var(y_true)
        var_residual = np.var(y_true - y_pred)
        
        # 处理目标值方差为零的特殊情况
        if var_true < 1e-9:  # 近似为零
            # 如果预测值也是常数，则完美拟合
            if np.var(y_pred) < 1e-9 and abs(np.mean(y_true) - np.mean(y_pred)) < 1e-6:
                return 1.0
            # 否则无法计算有意义的解释方差
            return 0.0
        
        return 1 - var_residual / var_true

    def policy_update(self):
        #self.gameStatesBatch = np.array(self.gameStatesBatch).astype('float32')
        #print(f"after np array game state batch size is {len(self.gameStatesBatch)}")

        #self.gameActionProbsBatch = np.array(self.gameActionProbsBatch).astype('float32')

        #self.gameWinScoresBatch = np.array(self.gameWinScoresBatch).astype('float32')

        #print(self.gameWinScoresBatch)
        old_probs, old_sohv = self.policy_value_net.policy_value(self.gameStatesBatch)
        #print(old_v.flatten())

        for i in range(self.epochs):
            pl, sohl, entropy = self.policy_value_net.train_step(
                self.gameStatesBatch,
                self.gameActionProbsBatch,
                self.gameWinScoresBatch,
                self.learn_rate * self.lr_multiplier
            )

            new_probs, new_sohv = self.policy_value_net.policy_value(self.gameStatesBatch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:
                break

        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        scores_cpu = self.gameWinScoresBatch.cpu().numpy()

        #explained_var_old = self.safe_explained_variance(scores_cpu, old_v.flatten())
        #explained_var_new = self.safe_explained_variance(scores_cpu, new_v.flatten())

        
        explained_var_soh_old = (1 -
                             np.var(scores_cpu - old_sohv.flatten()) /
                             np.var(scores_cpu))
        explained_var_soh_new = (1 -
                             np.var(scores_cpu - new_sohv.flatten()) /
                             np.var(scores_cpu))
        
        
        print(f"cur train loop: {self.train_loop} ==========")
        print(("kl:{:.5f},\n"
               "lr_multiplier:{:.3f},\n"
               "policy loss: {}, \n"
               "spatial loss:{},\n"
               "entropy:{}, \n"
               "explained_var_spatial_old:{:.9f}, \n"
               "explained_var_spatial_new:{:.9f}, \n"
               ).format(kl,
                        self.lr_multiplier,
                        pl,
                        sohl,
                        entropy,
                        explained_var_soh_old,
                        explained_var_soh_new))
        print("loop end ====================================")

        self.train_loop += 1
        return pl, sohl, entropy

    def run(self):
        try:
            startSectionList = []
            stateCodingFileSize = os.path.getsize("StateCoding.bin")
            with open("StateCoding.bin", 'rb') as stateCodingsFile:
                #gameNb = struct.unpack('i', stateCodingsFile.read(4))[0]
                channelNb = struct.unpack('i', stateCodingsFile.read(4))[0]
                height = struct.unpack('i', stateCodingsFile.read(4))[0]
                width = struct.unpack('i', stateCodingsFile.read(4))[0]
                stateGameNb = 0
                zeroStart = 0
                oneStart = 0
                while stateCodingsFile.tell() < stateCodingFileSize:
                    stepNb_bytes = stateCodingsFile.read(4)
                    if not stepNb_bytes:  # 已到达文件末尾
                        break
                    stateGameNb += 1
                    stepNb = struct.unpack('i', stepNb_bytes)[0]
                    stepStates = []
                    
                    for j in range(stepNb):
                        #print(f"now comes to step {j}")
                        stateCoding = np.frombuffer(stateCodingsFile.read(channelNb*height*width*4), dtype=np.int32)
                        stateCoding = stateCoding.reshape((25, 14, 4)).astype(np.float32)
                        if j == 1:
                            if stateCoding[23][0][0] == 1:
                                zeroStart += 1
                                startSectionList.append(1)
                            elif stateCoding[23][0][0] == -1:
                                oneStart += 1
                                startSectionList.append(-1)

                        #if stateCoding[19][0][0] == -1:
                        #    stateCoding[18] = -stateCoding[18]
                        stepStates.append(stateCoding)
                    stepStates = list(stepStates)[:]
                    self.totalGameStatesBatch.extend(stepStates)
                print(f"zero start: {zeroStart}")
                print(f"one start: {oneStart}")
                print(f"game nb is state: {stateGameNb}")
                print(f"size of gameStatesBatch is: {len(self.totalGameStatesBatch)}")


            actionProbsFileSize = os.path.getsize("ActionProbs.bin")
            with open("ActionProbs.bin", 'rb') as actionProbsFile:
                actionGameNb = 0
                while actionProbsFile.tell() < actionProbsFileSize:
                    stepNb_bytes = actionProbsFile.read(4)
                    if not stepNb_bytes:  # 已到达文件末尾
                        break
                    actionGameNb += 1
                    stepNb = struct.unpack('i', stepNb_bytes)[0]
                    stepActionProbs = []

                    for j in range(stepNb):
                        #print(f"now comes to step {j}")
                        actionNbs = struct.unpack('i', actionProbsFile.read(4))[0]
                        actionProbs = np.zeros(20000, dtype=np.float32)
                        for k in range(actionNbs):
                            actionId = struct.unpack('i', actionProbsFile.read(4))[0]
                            actionProb = struct.unpack('f', actionProbsFile.read(4))[0]
                            actionProbs[actionId] = actionProb
                        stepActionProbs.append(actionProbs)
                    stepActionProbs = list(stepActionProbs)[:]
                    self.totalActionProbsBatch.extend(stepActionProbs)
                print(f"game nb is actionprobs: {actionGameNb}")

            testNb = 0
            sectionZeroWinNb = 0
            sectionOneWinNb = 0
            zeroStartWin = 0
            zeroStartLose = 0
            oneStartWin = 0
            oneStartLose = 0
            drawNb = 0
            winScoreFileSize = os.path.getsize("WinScores.bin")
            with open("WinScores.bin", 'rb') as winScoresFile:
                winScoreGameNb = 0
                testWinScoreNb = 0
                while winScoresFile.tell() < winScoreFileSize:
                    stepNb_bytes = winScoresFile.read(4)
                    if not stepNb_bytes:  # 已到达文件末尾
                        break
                    winScoreGameNb += 1
                    stepNb = struct.unpack('i', stepNb_bytes)[0]
                    winScores = []

                    #print("sdfsdfsdfsdfsdfsdfsdf")
                    for j in range(stepNb):
                        winScore = struct.unpack('f', winScoresFile.read(4))[0]
                        if j == 1:
                            if startSectionList[testNb] == 1:
                                if winScore == 1:
                                    sectionZeroWinNb += 1
                                    zeroStartWin += 1
                                elif winScore == -1:
                                    sectionOneWinNb += 1
                                    zeroStartLose += 1
                                else:
                                    drawNb += 1
                            else:
                                if winScore == 1:
                                    sectionOneWinNb += 1
                                    oneStartWin += 1
                                elif winScore == -1:
                                    sectionZeroWinNb += 1
                                    oneStartLose += 1
                                else:
                                    drawNb += 1
                        winScores.append(winScore)
                        testWinScoreNb += 1

                    winScores = list(winScores)[:]
                    self.totalGameWinScoresBatch.extend(winScores)
                    testNb += 1
                print(f"win score game nb is actionprobs: {winScoreGameNb}")
                print(f"size of winScores is: {len(self.totalGameWinScoresBatch)}")
                print(f"0 win: {sectionZeroWinNb}")
                print(f"1 win: {sectionOneWinNb}")
                print(f"draw: {drawNb}")
                print(f"0 start win {zeroStartWin}")
                print(f"0 start lose {zeroStartWin}")
                print(f"1 start win {oneStartWin}")
                print(f"1 start lose {oneStartLose}")

            #states_array = np.array(self.totalGameStatesBatch)
            #probs_array = np.array(self.totalActionProbsBatch)
            #wins_array = np.array(self.totalGameWinScoresBatch)

            states_array = np.stack(self.totalGameStatesBatch).astype(np.float32)
            probs_array = np.stack(self.totalActionProbsBatch).astype(np.float32)
            wins_array = np.array(self.totalGameWinScoresBatch).astype(np.float32)

            total_samples = len(states_array)
            for i in range(5000):
                random_indices = np.random.choice(total_samples, size=512, replace=True)
                batch_states = states_array[random_indices]
                batch_probs = probs_array[random_indices]
                batch_scores = wins_array[random_indices]
                #self.gameStatesBatch = states_array[random_indices]
                #self.gameActionProbsBatch = probs_array[random_indices]
                #self.gameWinScoresBatch = wins_array[random_indices]

                self.gameStatesBatch = torch.as_tensor(batch_states, dtype=torch.float32).to('cuda')
                self.gameActionProbsBatch = torch.as_tensor(batch_probs, dtype=torch.float32).to('cuda')
                self.gameWinScoresBatch = torch.as_tensor(batch_scores, dtype=torch.float32).to('cuda')

                pl, sohl, entropy = self.policy_update()
                self.policy_value_net.save_model("model.pt")

        except KeyboardInterrupt:
            print('\n\rquit')

trainer = Trainer(init_model='model.pt')
trainer.run()
