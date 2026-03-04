from evalscope import run_task, TaskConfig
from custom import MyCustomModel


save_path = "4000"
DEVICE = "cuda:4"
out_path = "/workspace/EVAL/" + save_path
# 创建模型实例
custom_model = MyCustomModel(
    model_name='MDModel',
    gsm8k_path= save_path,
    _device = DEVICE,
    model_args={'test': 'test'}
)

# 配置评测任务
task_config = TaskConfig(
    model=custom_model,
    datasets=['gsm8k'],
    dataset_args={
        "gsm8k": { 
            "few_shot_num": 4,
            "few_shot_random": False,
            'filters': {'remove_until': '</think>'}
            }
    },
    work_dir = out_path,
    timeout=60000, 
    stream=True, 
    limit=None,
)

# 运行评测
results = run_task(task_cfg=task_config)
