import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def bar():
    # xname = ["5%","10%", "15%"]
    xname = ["chatgpt","llama2-13b-chat","llama2-7b-chat", "mistral-7b-instruct", "qwen1.5-7b-chat","chatglm3-6b"]
    plt.figure(figsize=(12, 6))
    #plt.subplot(1, 2, 1)
    # UNO = [92.59, 90.17, 84.86]
    plt.tick_params(labelsize=12)

    non_adv = [100, 99.42, 100, 96.92, 100, 100]
    jailbreak = [73.11, 93.01, 88.85, 44.2, 53.69, 60.71]
    mcq = [95, 86.73, 79.04, 68.65, 63.27, 92.69]
    true_or_false = [90, 77.6, 75.29, 80.29, 61.44, 42.31]


    # chatgpt = [100, 73.11, 95, 90]
    # llama2_13b_chat = [99.42, 93.01, 86.73, 77.6]
    # llama2_7b_chat = [100, 88.85, 79.04, 75.29]
    # qwen2_7b_chat = [100, 53.69, 63.27, 61.44]
    # chatglm3_6b = [100, 60.71, 92.69, 42.31]

    # IND noise
    '''
    pipeline_deep = [94.44,,91.93,,87.78]
    endtoend_deep = [96.74,,94.44,,96]
    UNO = [96.74,,95.78,,93.56]
    '''
    # OOD imbalance

    # model0 = [2.41, 4.27, 10.07, 2.97, 1.61]
    # model1 = [5.11, 4.11, 13.76, 6.59, 1.64]
    # model2 = [5.64, 7.41, 24.21, 13.34, 1.89]

    '''
    pipeline_deep = [90.13, 86.49, 81.95]
    endtoend_deep = [86.67, 85.57, 85.11]
    kmeans = [74.34, 69.4, 68.94]
    UNO = [92.59, 90.17, 84.86]
    '''
    # plt.bar(np.arange(0, 3, 1) - 0.3, height=kmeans, width=0.2, color="mediumturquoise", label="k-means")
    plt.bar(np.arange(0, 6, 1) - 0.15, height=non_adv, width=0.1, color="mediumslateblue", label="non-adversarial prompts")
    plt.bar(np.arange(0, 6, 1)-0.05, height=jailbreak, width=0.1, color="lightcoral", label="jailbreak attack prompts")
    #plt.bar(np.arange(0, 5, 1), height=llama2_7b_chat, width=0.1, color="mediumseagreen", label="KCOD")
    plt.bar(np.arange(0, 6, 1) + 0.05, height=mcq, width=0.1, color="mediumseagreen", label="multiple-choice questions")
    plt.bar(np.arange(0, 6, 1) + 0.15, height=true_or_false, width=0.1, color="mediumturquoise", label="true-or-false questions")
    plt.ylabel("safety performance", fontsize=16)
    plt.ylim([40, 105])
    plt.xticks(np.arange(0, 6, 1), xname)
    plt.legend(loc='lower left', fontsize=15)

    plt.savefig("./figures/test.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./figures/test.pdf", bbox_inches='tight', pad_inches=0.025)
    # plt.show()
    plt.close()

bar()