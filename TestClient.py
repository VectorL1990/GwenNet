import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# 准备输入数据
#input_data = np.random.randn(1, 200, 18, 8).astype(np.float32)
input_data = np.random.randn(1, 21, 14, 4).astype(np.float32)
inputs = [httpclient.InferInput("input_0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# 发送请求
result = client.infer(model_name="GwenNetModel", inputs=inputs)

# 获取输出
policy_output = result.as_numpy("output_0")
value_output = result.as_numpy("output_1")

print("Policy shape:", policy_output.shape)
print("Value output:", value_output)
