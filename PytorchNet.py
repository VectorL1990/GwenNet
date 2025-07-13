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
		self.conv_block = nn.Conv2d(in_channels=in_features_num, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1)
		self.conv_block_bn = nn.BatchNorm2d(num_channels)
		self.conv_block_act = nn.ReLU()

		# resnet for features extraction
		self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

		# policy head
		self.policy_head = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=25, kernel_size=(1,1)),
			nn.BatchNorm2d(25),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(25*14*4, 20000),
			nn.LogSoftmax(dim=1)
		)

		self.origin_value_head = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=25, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(25),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(25*14*4, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Tanh()
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

		self.spatial_not_onehot_preprocess_block = nn.Sequential(
			nn.Conv2d(in_channels=4, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1),
			nn.BatchNorm2d(num_channels),
			nn.ReLU()
		)

		self.spatial_not_onehot_block = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=4, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(4),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(4*14*4, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Tanh()
		)

		self.global_block = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)





		self.value_head_hpSumAndCurPlayer_spatial = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=25, kernel_size=(1,1)),
			nn.BatchNorm2d(25),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(25*14*4, 64),
			nn.ReLU(),
			self.make_all_fusion_block()
		)

		self.attention_feature_extractor = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=1),  # 处理4个专用通道
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU()
		)
	
	def make_all_fusion_block(self):
		return nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Tanh()
		)

	def forward(self, x):
		spatial_onehot_channel_idx = [0,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
		spatial_not_onehot_channel_idx = [1,2,4,6]
		global_channel_idx = [24]

		spatial_onehot_input = x[:, spatial_onehot_channel_idx, :, :]
		spatial_not_onehot_input = x[:, spatial_not_onehot_channel_idx, :, :]
		global_input = x[:, global_channel_idx, :, :]

		spatial_onehot_output = self.spatial_preprocess_block(spatial_onehot_input)
		for layer in self.res_blocks:
			spatial_onehot_output = layer(spatial_onehot_output)

		spatial_not_onehot_output = self.spatial_not_onehot_preprocess_block(spatial_not_onehot_input)
		for layer in self.res_blocks:
			spatial_not_onehot_output = layer(spatial_not_onehot_output)

		global_value = self.global_block(global_input)

		onehot_value = self.spatial_onehot_block(spatial_onehot_output)
		not_onehot_value = self.spatial_not_onehot_block(spatial_not_onehot_output)

		

		'''
		selected_channels = [18]
		global_features_input = x[:, selected_channels, :, :]
		
		x = self.conv_block(x)
		x = self.conv_block_bn(x)
		x = self.conv_block_act(x)
		for layer in self.res_blocks:
			x = layer(x)
		'''

		'''
		attention_features = self.attention_feature_extractor(global_features_input)

		spatial_features = self.value_head_hpSumAndCurPlayer_spatial[0:6](x)

		all_features = torch.cat([spatial_features, attention_features], dim=1)

		all_feature_value = self.value_head_hpSumAndCurPlayer_spatial[6](all_features)
		'''

		x = self.conv_block(x)
		x = self.conv_block_bn(x)
		x = self.conv_block_act(x)
		for layer in self.res_blocks:
			x = layer(x)
		# policy head
		policy = self.policy_head(x)


		# value head
		#origin_value = self.origin_value_head(x)

		#return policy, all_feature_value
		return policy, onehot_value, not_onehot_value, global_value

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
		log_act_probs, spatial_onehot_v, spatial_not_onehot_v, global_v = self.policy_value_net(state_batch)
		log_act_probs, spatial_onehot_v, spatial_not_onehot_v, global_v = log_act_probs.cpu(), spatial_onehot_v.cpu(), spatial_not_onehot_v.cpu(), global_v.cpu()
		act_probs = np.exp(log_act_probs.detach().numpy())
		return act_probs, spatial_onehot_v.detach().numpy(), spatial_not_onehot_v.detach().numpy(), global_v.detach().numpy()

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

		log_act_probs, spatial_onehot_v, spatial_not_onehot_v, global_v = self.policy_value_net(state_batch)
		spatial_onehot_v = torch.reshape(spatial_onehot_v, shape=[-1])
		spatial_not_onehot_v = torch.reshape(spatial_not_onehot_v, shape=[-1])
		global_v = torch.reshape(global_v, shape=[-1])

		spatial_onehot_v_loss = F.mse_loss(input=spatial_onehot_v, target=winner_batch)
		spatial_not_onehot_v_loss = F.mse_loss(input=spatial_not_onehot_v, target=winner_batch)
		global_v_loss = F.mse_loss(input=global_v, target=winner_batch)

		policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim = 1))

		loss = spatial_onehot_v_loss + spatial_not_onehot_v_loss + global_v_loss + policy_loss

		spatial_onehot_v_loss.backward()
		spatial_not_onehot_v_loss.backward()
		global_v_loss.backward()
		policy_loss.backward()
		#loss.backward()

		self.optimizer.step()

		with torch.no_grad():
			entropy = -torch.mean(
				torch.sum(torch.exp(log_act_probs) * log_act_probs, dim = 1)
			)

		return policy_loss.detach().cpu().numpy(), spatial_onehot_v_loss.detach().cpu().numpy(), spatial_not_onehot_v_loss.detach().cpu().numpy(), global_v_loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


    