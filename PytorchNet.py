import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast

class ResBlock(nn.Module):
	def __init__(self, num_filters=256):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3,3), stride=(1,1), padding=1)
		self.conv1_bn = nn.BatchNorm2d(num_filters, )
		self.conv1_act = nn.ReLU()
		self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3,3), stride=(1,1), padding=1)
		self.conv2_bn = nn.BatchNorm2d(num_filters, )
		self.conv2_act = nn.ReLU()

	def forward(self, x):
		y = self.conv1(x)
		y = self.conv1_bn(y)
		y = self.conv1_act(y)
		y = self.conv2(y)
		y = self.conv2_bn(y)
		y = x + y
		return self.conv2_act(y)

class Net(nn.Module):
	#def __init__(self, in_features_num = 200, num_channels=256, num_res_blocks=7):
	def __init__(self, in_features_num = 25, num_channels=256, num_res_blocks=7):
		# in_features_num represents feature descriptions of the board, which is 
		super().__init__()

		# resnet for features extraction
		self.policy_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
		self.spatial_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
		self.hp_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
		self.defence_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
		self.curcd_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
		self.cura_res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

		self.policy_spatial_onehot_preprocess = nn.Sequential(
			nn.Conv2d(in_channels=20, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.policy_spatial_onehot_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(20*14*4, 20000),
			nn.LogSoftmax(dim=1)
		)


		self.spatial_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=20, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.spatial_onehot_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(20*14*4, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Tanh()
		)

		self.hp_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.defence_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.curcd_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.cura_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.hp_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(1*14*4, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)

		self.defence_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(1*14*4, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)

		self.curcd_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(1*14*4, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)

		self.cura_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(1*14*4, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)





	

	def forward(self, x):
		spatial_onehot_channel_idx = [0,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
		hp_channel_idx = [1]
		defence_channel_idx = [2]
		cur_cd_channel_idx = [4]
		cur_available_channel_idx = [6]
		global_channel_idx = [23, 24]

		spatial_onehot_input = x[:, spatial_onehot_channel_idx, :, :]
		hp_input = x[:, hp_channel_idx, :, :] / 10.0
		defence_input = x[:, defence_channel_idx, :, :] / 5.0
		cur_cd_input = x[:, cur_cd_channel_idx, :, :] / 2.0
		##################################### problem
		cur_cd_input[:, 0, :, :] = 0.0
		cur_available_input = x[:, cur_available_channel_idx, :, :] / 2.0
		global_input = x[:, global_channel_idx, :, :]
		global_input[:, 1, :, :] = global_input[:, 1, :, :] / 10.0

		spatial_onehot_output = self.spatial_preprocess_block(spatial_onehot_input)
		for layer in self.spatial_res_blocks:
			spatial_onehot_output = layer(spatial_onehot_output)

		'''
		hp_output = self.hp_preprocess_block(hp_input)
		for layer in self.hp_res_blocks:
			hp_output = layer(hp_output)

		defence_output = self.defence_preprocess_block(defence_input)
		for layer in self.defence_res_blocks:
			defence_output = layer(defence_output)

		cur_cd_output = self.curcd_preprocess_block(cur_cd_input)
		for layer in self.curcd_res_blocks:
			cur_cd_output = layer(cur_cd_output)

		cur_available_output = self.cura_preprocess_block(cur_available_input)
		for layer in self.cura_res_blocks:
			cur_available_output = layer(cur_available_output)
		'''


		onehot_value = self.spatial_onehot_block(spatial_onehot_output)
		'''
		hp_value = self.hp_block(hp_output)
		defence_value = self.defence_block(defence_output)
		cur_cd_value = self.curcd_block(cur_cd_output)
		cur_available_value = self.cura_block(cur_available_output)
		'''

		
		policy_output = self.policy_spatial_onehot_preprocess(spatial_onehot_input)
		for layer in self.policy_res_blocks:
			policy_output = layer(policy_output)
		policy = self.policy_spatial_onehot_block(policy_output)



		#return policy, all_feature_value
		return policy, onehot_value

class PolicyValueNet:
	def __init__(self, model_file = None, use_gpu = True, device = 'cuda'):
		self.use_gpu = use_gpu
		self.l2_const = 2e-3
		self.device = device
		self.policy_value_net = Net().to(self.device)
		self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
		if model_file:
			try:
				self.load_model(model_file)
			except Exception as e:
				print(f"初始化时加载模型失败: {e}")

	def load_model(self, model_file):
		if not os.path.exists(model_file):
			raise FileNotFoundError(f"PytorchNet can not find: {model_file}")

		try:
			traced_model = torch.jit.load(model_file, map_location=self.device)
			self.policy_value_net.load_state_dict(traced_model.state_dict())
			print(f"successfully load TorchScript model: {model_file}")
			return
		except Exception as e:
			print(f"fail to load TorchScript model: {e}")


		try:
			state_dict = torch.load(model_file, map_location=self.device)

			# 处理可能的键不匹配
			new_state_dict = {}
			for k, v in state_dict.items():
				name = k[7:] if k.startswith("module.") else k
				new_state_dict[name] = v
			
			self.policy_value_net.load_state_dict(new_state_dict)
			print(f"成功加载状态字典模型: {model_file}")
		except Exception as e:
			print(f"加载模型失败: {e}")
			raise RuntimeError("无法加载模型文件")

	def policy_value(self, state_batch):
		self.policy_value_net.eval()
		state_batch = torch.tensor(state_batch).to(self.device)
		log_act_probs, spatial_onehot_v = self.policy_value_net(state_batch)
		log_act_probs = log_act_probs.cpu()
		spatial_onehot_v = spatial_onehot_v.cpu()
		act_probs = np.exp(log_act_probs.detach().numpy())
		return act_probs, spatial_onehot_v.detach().numpy()

	def save_model(self, model_file):
		example_input = torch.randn(1, 25, 14, 4).to(next(self.policy_value_net.parameters()).device)
		self.policy_value_net.eval()
		traced_model = torch.jit.trace(self.policy_value_net, example_input)
		traced_model.save(model_file)
		torch.save(self.policy_value_net.state_dict(), model_file.replace(".pt", "_state_dict.pt"))
		#torch.save(self.policy_value_net.state_dict(), model_file)

	def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
		self.policy_value_net.train()

		state_batch = torch.tensor(state_batch, dtype = torch.float32).to(self.device)
		mcts_probs = torch.tensor(mcts_probs, dtype = torch.float32).to(self.device)
		winner_batch = torch.tensor(winner_batch, dtype = torch.float32).to(self.device)

		self.optimizer.zero_grad()
		for params in self.optimizer.param_groups:
			params['lr'] = lr

		log_act_probs, spatial_onehot_v = self.policy_value_net(state_batch)
		spatial_onehot_v = torch.reshape(spatial_onehot_v, shape=[-1])

		spatial_onehot_v_loss = F.mse_loss(input=spatial_onehot_v, target=winner_batch)

		policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim = 1))

		spatial_onehot_v_loss.backward()
		policy_loss.backward()

		self.optimizer.step()

		with torch.no_grad():
			entropy = -torch.mean(
				torch.sum(torch.exp(log_act_probs) * log_act_probs, dim = 1)
			)

		return policy_loss.detach().cpu().numpy(), spatial_onehot_v_loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


    