if __name__ == "__main__":
	from gan_task import GANTask
	from config import config
	from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
	import tensorflow as tf
	print("Can see tensorflow version "  + str(tf.__version__))
	print("Started working")
	scheduler = GPUTaskScheduler(config=config, gpu_task_class=GANTask)
	print("Has set up GPUTaskScheduler")
	scheduler.start()