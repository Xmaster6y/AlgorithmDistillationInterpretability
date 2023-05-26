import wandb
import sys 


api = wandb.Api()
# if user provides a model name, use that, otherwise use the default
model_name = sys.argv[1] if len(sys.argv) > 1 else 'dark_room_test__1000__1684713463:v25'

artifact = api.artifact(f'algorithm-distillation-interp/Algorithm_distillation_intepretability/{model_name}')
artifact.download(root='models/')
