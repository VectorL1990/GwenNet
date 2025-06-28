import tritonclient.http as httpclient
import numpy as np
import os

client = httpclient.InferenceServerClient(url="localhost:8000")

#testCoding = np.random.randn(1, 21, 14, 4).astype(np.float32)

#def load_test_data():
stateCodingFileSize = os.path.getsize("SectionZeroWeakTestCaseStateCoding.bin")
with open("SectionZeroWeakTestCaseStateCoding.bin", 'rb') as stateCodingsFile:
    while stateCodingsFile.tell() < stateCodingFileSize:
        stateCoding = np.frombuffer(stateCodingsFile.read(21*14*4*4), dtype=np.int32)
        stateCoding = stateCoding.reshape((1, 21, 14, 4)).astype(np.float32)
        testCoding = stateCoding
        testCoding[0][17] = -testCoding[0][17]

# 准备输入数据
#input_data = np.random.randn(1, 21, 14, 4).astype(np.float32)
#load_test_data()
print(testCoding[0][17])
input_data = testCoding
inputs = [httpclient.InferInput("input_0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# 发送请求
result = client.infer(model_name="GwenNetModel", inputs=inputs)

# 获取输出
policy_output = result.as_numpy("output_0")
value_output = result.as_numpy("output_1")

print("Policy shape:", policy_output.shape)
print("Value output:", value_output)
