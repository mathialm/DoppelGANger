if __name__ == "__main__":
	from gan_task import GANTask
	from config import config
	from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
	print("Started working")
	scheduler = GPUTaskScheduler(config=config, gpu_task_class=GANTask)
	print("Has set up GPUTaskScheduler")
	scheduler.start()