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
	
class GlobalFeatureExtractor(nn.Module):
    def __init__(self, num_input_channels, num_features=128):
        super().__init__()
        # 使用通道注意力聚焦重要特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_input_channels, num_input_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(num_input_channels // 16, num_input_channels, 1),
            nn.Sigmoid()
        )
        
        # HP总和特征提取
        self.hp_extractor = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 14 * 4, num_features)
        )
        
    def forward(self, x):
        # 通道注意力加权
        channel_weights = self.channel_attention(x)
        weighted_x = x * channel_weights
        
        # 提取全局HP特征
        global_features = self.hp_extractor(weighted_x)
        return global_features

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

		self.global_feature_extractor = GlobalFeatureExtractor(num_channels)

		self.cur_player_extractor = nn.Sequential(
			nn.Linear(1, 16),
			nn.ReLU(),
			nn.Linear(16,32)
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

		self.value_head_hp_feature = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(256*14*4, 256),
			nn.ReLU(),
			self.make_hpsum_fusion_block(256),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Tanh()
		)

		self.value_head_curplayer_feature = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=(1,1)),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(128, 64),
			nn.ReLU(),
			self.make_curplayer_feature_fusion_block(64)
		)

		self.value_head_2 = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=(1,1)),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Tanh()
		)

		self.value_head_3 = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=21, kernel_size=(1,1)),
			nn.BatchNorm2d(21),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(21*14*4, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Tanh()
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

	def make_hpsum_fusion_block(self, input_size):
		return nn.Sequential(
			# 特征拼接层
			nn.Linear(input_size + 128, input_size),  # 128是全局特征大小
			nn.ReLU(),
			# 特征交互层
			nn.Linear(input_size, input_size),
			nn.ReLU(),
			# 归一化
			nn.LayerNorm(input_size)
		)

	def forward(self, x):
		curplayer = x[:, 17, 0, 0].unsqueeze(1)

		x = self.conv_block(x)
		x = self.conv_block_bn(x)
		x = self.conv_block_act(x)
		for layer in self.res_blocks:
			x = layer(x)

		global_features = self.global_feature_extractor(x)

		curplayer_features = self.cur_player_extractor(curplayer)

		curplayer_spatial_features = self.value_head_curplayer_feature[0:7](x)

		fused_curplayer_features = torch.cat([curplayer_spatial_features, curplayer_features], dim=1)

		curplayer_feature_value = self.value_head_curplayer_feature[7](fused_curplayer_features)

		# policy head
		policy = self.policy_head(x)


		# value head
		'''
		spatial_features = self.value_head[0:6](x)
		fusion_features = torch.cat([spatial_features, global_features], dim=1)
		fusion_features = self.value_head[6](fusion_features)
		value = self.value_head[7:](fusion_features)
		'''

		#value = self.value_head_2(x)
		#value = self.value_head_3(x)

		return policy, curplayer_feature_value

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


    