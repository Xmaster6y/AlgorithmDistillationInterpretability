import wandb
api = wandb.Api()
artifact = api.artifact('algorithm-distillation-interp/Algorithm_distillation_intepretability/testdebug__1__1683169313:v0')
artifact.download()