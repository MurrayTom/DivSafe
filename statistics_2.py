import json


def read_json_tolist(filename=""):
    with open(filename, 'r') as file:
        data = json.load(file)
    print(len(data))

    return data

data = read_json_tolist(filename="/root/EasyJailbreak/outputs_statistics/jailbreak/mistral-7b-instruct.json")
print(len(data))
print("------------------------------")

num_1, num_2 = 0, 0
for item in data:
    if item["wrong_class"]!=None:
        if "1" in item["wrong_class"] or "first" in item["wrong_class"]:
            num_1+=1
        elif "2" in item["wrong_class"] or "second" in item["wrong_class"]:
            num_2+=1
        else:
            print(item["wrong_class"])

print(num_1, num_2)
