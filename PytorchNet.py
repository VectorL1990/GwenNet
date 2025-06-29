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
	def __init__(self, in_features_num = 21, num_channels=256, num_res_blocks=7):
		# in_features_num represents feature descriptions of the board, which is 
		super().__init__()
		self.conv_block = nn.Conv2d(in_channels=in_features_num, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1)
		self.conv_block_bn = nn.BatchNorm2d(num_channels)
		self.conv_block_act = nn.ReLU()

		# resnet for features extraction
		self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

		self.global_feature_extractor = nn.Sequential(
			nn.Linear(1, 16),
			nn.ReLU(),
			nn.Linear(16, 32)
		)

		# policy head
		self.policy_head = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=21, kernel_size=(1,1)),
			nn.BatchNorm2d(21),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(21*14*4, 20000),
			nn.LogSoftmax(dim=1)
		)

		self.origin_value_head = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=21, kernel_size=(1,1), stride=(1,1)),
			nn.BatchNorm2d(21),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(21*14*4, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Tanh()
		)

		self.value_head_hpSumAndCurPlayer_spatial = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=21, kernel_size=(1,1)),
			nn.BatchNorm2d(21),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(21*14*4, 64),
			nn.ReLU(),
			self.make_all_fusion_block()
		)

		self.value_head_hpSumAndCurPlayer_global = nn.Sequential(
			#nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=(1,1)),
			#nn.BatchNorm2d(128),
			#nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(21, 64),
			nn.ReLU(),
			self.make_global_fusion_block(64)
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

	def make_curplayer_feature_fusion_block(self, input_size):
		return nn.Sequential(
			nn.Linear(input_size + 32, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Tanh()
		)

	def make_global_fusion_block(self, input_size):
		return nn.Sequential(
			nn.Linear(input_size + 32 + 32 + 32 + 32, 128),  # 128是全局特征大小
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
		selected_channels = [20]
		global_features_input = x[:, selected_channels, :, :]
		curplayer = x[:, 17, 0, 0].unsqueeze(1)
		sectionZeroHp = x[:, 18, 0, 0].unsqueeze(1)
		sectionOneHp = x[:, 19, 0, 0].unsqueeze(1)
		hpDiff = x[:, 20, 0, 0].unsqueeze(1)

		x = self.conv_block(x)
		x = self.conv_block_bn(x)
		x = self.conv_block_act(x)
		for layer in self.res_blocks:
			x = layer(x)

		#curplayer_features = self.global_feature_extractor(curplayer)
		#hpDiffFeatures = self.global_feature_extractor(hpDiff)
		#sectionZeroFeatures = self.global_feature_extractor(sectionZeroHp)
		#sectionOneFeatures = self.global_feature_extractor(sectionOneHp)
		attention_features = self.attention_feature_extractor(global_features_input)

		spatial_features = self.value_head_hpSumAndCurPlayer_spatial[0:6](x)
		#global_features = self.value_head_hpSumAndCurPlayer_global[0:4](x)
		

		#fused_features = torch.cat([global_features, curplayer_features], dim=1)
		#fused_features = torch.cat([fused_features, hpDiffFeatures], dim=1)
		#fused_features = torch.cat([fused_features, sectionZeroFeatures], dim=1)
		#fused_features = torch.cat([fused_features, sectionOneFeatures], dim=1)

		#global_feature_all = self.value_head_hpSumAndCurPlayer_global[4](fused_features)

		all_features = torch.cat([spatial_features, attention_features], dim=1)

		all_feature_value = self.value_head_hpSumAndCurPlayer_spatial[6](all_features)

		# policy head
		policy = self.policy_head(x)


		# value head
		#origin_value = self.origin_value_head(x)

		return policy, all_feature_value

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
		log_act_probs, value = self.policy_value_net(state_batch)
		log_act_probs, value = log_act_probs.cpu(), value.cpu()
		act_probs = np.exp(log_act_probs.detach().numpy())
		return act_probs, value.detach().numpy()

	def save_model(self, model_file):
		example_input = torch.randn(1, 21, 14, 4).to(next(self.policy_value_net.parameters()).device)
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

		log_act_probs, value = self.policy_value_net(state_batch)
		value = torch.reshape(value, shape=[-1])

		value_loss = F.mse_loss(input=value, target=winner_batch)

		policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim = 1))

		loss = value_loss + policy_loss

		loss.backward()

		self.optimizer.step()

		with torch.no_grad():
			entropy = -torch.mean(
				torch.sum(torch.exp(log_act_probs) * log_act_probs, dim = 1)
			)

		return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


    