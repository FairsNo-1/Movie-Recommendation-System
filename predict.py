import paddle
import pickle
import numpy as np


# 定义根据用户给定的电影推荐相似且符合他兴趣的电影
def recommend_mov_for_usr_v1(usr_id, mov_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    # 获取特定的电影和用户的信息
    usr_feat_re = usr_feats[str(usr_id)]
    mov_feat_re = mov_feats[str(mov_id)]

    # 定义相似度矩阵（前者为用户与电影之间的，后者是电影之间的）
    cos_sims_um = []
    cos_sims_mm = []

    # with dygraph.guard():
    paddle.disable_static()
    # 索引电影特征，计算和输入用户ID的特征的相似度
    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        usr_feat_re = paddle.to_tensor(usr_feat_re)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(usr_feat_re, mov_feat)
        cos_sims_um.append(sim.numpy()[0])

    # 对相似度排序
    index = np.argsort(cos_sims_um)[-top_k:]

    # 读取电影文件里的数据，根据电影ID索引到电影信息
    mov_info = {}
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item

    print("当前的用户是：")
    print("usr_id:", usr_id)
    print("你输入的电影相关信息：")
    print(mov_info[str(mov_id)])
    print("根据该电影推荐你可能喜欢的电影是：")

    # 加入随机选择因素，确保每次推荐的都不一样
    res = []
    while len(res) < 30:  # res数组存储用户与电影相似度矩阵中从大到小前30个对象
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    # 计算用户给定电影与用户感兴趣的电影之间的相似度
    for id in res:
        mov_feat_0 = mov_feats[str(id)]
        paddle.disable_static()
        mov_feat_0 = paddle.to_tensor(mov_feat_0)
        mov_feat_re = paddle.to_tensor(mov_feat_re)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(mov_feat_re, mov_feat_0)
        cos_sims_mm.append(sim.numpy()[0])

    # 对相似度排序
    index_0 = np.argsort(cos_sims_mm)[-top_k:]

    # 加入随机选择因素，确保每次推荐的都不一样
    rec = []
    while len(rec) < pick_num:
        val_0 = np.random.choice(len(index_0), 1)[0]
        idx_0 = index_0[val_0]
        mov_id = list(mov_feats.keys())[idx_0]
        if mov_id not in rec:
            rec.append(mov_id)

    for id in rec:
        print("mov_id:", id, mov_info[str(id)])


# 只根据提供的电影推荐相似的电影
def recommend_mov_for_usr_v2(mov_id, top_k, pick_num, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k
    # 读取电影特征
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    mov_feat_re = mov_feats[str(mov_id)]

    # 定义相似度矩阵
    cos_sims = []

    # 读取电影文件里的数据，根据电影ID索引到电影信息
    mov_info = {}
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item

    print("你输入的电影相关信息：")
    print(mov_info[str(mov_id)])
    print("根据该电影推荐你可能喜欢的电影是：")

    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        mov_feat_re = paddle.to_tensor(mov_feat_re)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(mov_feat_re, mov_feat)
        cos_sims.append(sim.numpy()[0])

    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    # 随机推荐
    res = []
    while len(res) < pick_num: # 保存相似度矩阵中从大到小前pick_num个对象
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    for id in res:
        print("mov_id:", id, mov_info[str(id)])


def main():
    movie_data_path = "./work/ml-1m/movies.dat"
    print("模式一：根据提供的电影预测相似同时符合特定用户兴趣的电影(需要提供用户ID);")
    print("模式二：只根据提供的电影预测相似的电影(无需提供用户ID).")
    cod = input("输入Y使用模式一预测，否则使用模式二预测：")
    if cod == "Y":
        top_k, pick_num = 50, 6
        usr_id = input("请输入用户ID:")
        mov_id = input("请输入电影ID:")
        recommend_mov_for_usr_v1(usr_id, mov_id, top_k, pick_num,
                                'usr_feat.pkl', 'mov_feat.pkl', movie_data_path)
    else:
        top_k, pick_num = 30, 6
        mov_id = input("请输入电影ID:")
        recommend_mov_for_usr_v2(mov_id, top_k, pick_num, 'mov_feat.pkl', movie_data_path)


if __name__ == "__main__":
    main()