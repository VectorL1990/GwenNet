import tritonclient.http as httpclient
import numpy as np
import os
import struct
import copy

client = httpclient.InferenceServerClient(url="localhost:8000")

#testCoding = np.random.randn(1, 21, 14, 4).astype(np.float32)


#def load_test_data():
sectionZeroWeakStateCodingFileSize = os.path.getsize("SectionZeroWeakTestCaseStateCoding.bin")
with open("SectionZeroWeakTestCaseStateCoding.bin", 'rb') as stateCodingsFile:
    while stateCodingsFile.tell() < sectionZeroWeakStateCodingFileSize:
        stateCoding_0 = np.frombuffer(stateCodingsFile.read(21*14*4*4), dtype=np.int32)
        stateCoding_0 = stateCoding_0.reshape((1, 21, 14, 4)).astype(np.float32)
        sectionZeroWeakSectionZeroActionTestCoding = stateCoding_0
        sectionZeroWeakSectionZeroActionTestCoding[0][17] = 1

        stateCoding_1 = copy.deepcopy(stateCoding_0)
        #stateCoding_1 = stateCoding_1.reshape((1, 21, 14, 4)).astype(np.float32)
        sectionZeroWeakSectionOneActionTestCoding = stateCoding_1
        sectionZeroWeakSectionOneActionTestCoding[0][17] = -1
        sectionZeroWeakSectionOneActionTestCoding[0][20] = -sectionZeroWeakSectionOneActionTestCoding[0][20]


sectionOneWeakStateCodingFileSize = os.path.getsize("SectionOneWeakTestCaseStateCoding.bin")
with open("SectionOneWeakTestCaseStateCoding.bin", 'rb') as stateCodingsFile:
    while stateCodingsFile.tell() < sectionOneWeakStateCodingFileSize:
        stateCoding_2 = np.frombuffer(stateCodingsFile.read(21*14*4*4), dtype=np.int32)
        stateCoding_2 = stateCoding_2.reshape((1, 21, 14, 4)).astype(np.float32)
        sectionOneWeakSectionZeroActionTestCoding = stateCoding_2
        sectionOneWeakSectionZeroActionTestCoding[0][17] = 1

        stateCoding_3 = copy.deepcopy(stateCoding_2)
        #stateCoding_3 = stateCoding_3.reshape((1, 21, 14, 4)).astype(np.float32)
        sectionOneWeakSectionOneActionTestCoding = stateCoding_3
        sectionOneWeakSectionOneActionTestCoding[0][17] = -1
        sectionOneWeakSectionOneActionTestCoding[0][20] = -sectionOneWeakSectionOneActionTestCoding[0][20]

# 准备输入数据
#input_data = np.random.randn(1, 21, 14, 4).astype(np.float32)
#load_test_data()
print("section Zero Weak Section Zero Action TestCoding")
print(sectionZeroWeakSectionZeroActionTestCoding[0][20])
print(sectionZeroWeakSectionZeroActionTestCoding[0][17])
input_data_0 = sectionZeroWeakSectionZeroActionTestCoding
inputs_0 = [httpclient.InferInput("input_0", input_data_0.shape, "FP32")]
inputs_0[0].set_data_from_numpy(input_data_0)

# 发送请求
result_0 = client.infer(model_name="GwenNetModel", inputs=inputs_0)

# 获取输出
policy_output_0 = result_0.as_numpy("output_0")
value_output_0 = result_0.as_numpy("output_1")

print("Policy shape:", policy_output_0.shape)
print("Value output:", value_output_0)


print("section Zero Weak Section One Action TestCoding")
print(sectionZeroWeakSectionOneActionTestCoding[0][20])
print(sectionZeroWeakSectionOneActionTestCoding[0][17])
input_data_1 = sectionZeroWeakSectionOneActionTestCoding
inputs_1 = [httpclient.InferInput("input_0", input_data_1.shape, "FP32")]
inputs_1[0].set_data_from_numpy(input_data_1)

# 发送请求
result_1 = client.infer(model_name="GwenNetModel", inputs=inputs_1)

# 获取输出
policy_output_1 = result_1.as_numpy("output_0")
value_output_1 = result_1.as_numpy("output_1")

print("Policy shape:", policy_output_1.shape)
print("Value output:", value_output_1)



print("section One Weak Section Zero Action TestCoding")
print(sectionOneWeakSectionZeroActionTestCoding[0][20])
print(sectionOneWeakSectionZeroActionTestCoding[0][17])
input_data_2 = sectionOneWeakSectionZeroActionTestCoding
inputs_2 = [httpclient.InferInput("input_0", input_data_2.shape, "FP32")]
inputs_2[0].set_data_from_numpy(input_data_2)

# 发送请求
result_2 = client.infer(model_name="GwenNetModel", inputs=inputs_2)

# 获取输出
policy_output_2 = result_2.as_numpy("output_0")
value_output_2 = result_2.as_numpy("output_1")

print("Policy shape:", policy_output_2.shape)
print("Value output:", value_output_2)



print("section One Weak Section One Action TestCoding")
print(sectionOneWeakSectionOneActionTestCoding[0][20])
print(sectionOneWeakSectionOneActionTestCoding[0][17])
input_data_3 = sectionOneWeakSectionOneActionTestCoding
inputs_3 = [httpclient.InferInput("input_0", input_data_3.shape, "FP32")]
inputs_3[0].set_data_from_numpy(input_data_3)

# 发送请求
result_3 = client.infer(model_name="GwenNetModel", inputs=inputs_3)

# 获取输出
policy_output_3 = result_3.as_numpy("output_0")
value_output_3 = result_3.as_numpy("output_1")

print("Policy shape:", policy_output_3.shape)
print("Value output:", value_output_3)
