# huggingface-cli login --token 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 --add-to-git-credential
# 首先去自己账号里面 new 一个dataset.  这里面我创立玩数据就是这个https://huggingface.co/datasets/zhangbo2008/video_data 链接能打开.
#那么下面就可以往上面当网盘传数据了.
from huggingface_hub import HfApi
api = HfApi()


rep="zhangbo2008/wav2lip2"
rt='model'



api.create_repo(repo_id =rep,exist_ok=True,repo_type=rt) #======注意要写好类型
api.upload_folder(
    folder_path="./",
    repo_id=rep,
    repo_type=rt,   # 下面这里也要配置好类型.
    
)
#============添加model card 方便别人使用!!!!!!!
from huggingface_hub import ModelCard

content = """
wav2lip直接使用. 1.py 2.py 3.py 4.py
"""

card = ModelCard(content)
card.push_to_hub(repo_id=rep,repo_type=rt)