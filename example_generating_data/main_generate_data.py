if __name__ == "__main__":
    from gan_generate_data_task import GANGenerateDataTask
    from config_generate_data import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

    import tensorflow as tf
    import sys

    print("Main Python version " + str(sys.version))
    print("Can see tensorflow version " + str(tf.__version__))
    print("Started working")
    scheduler = GPUTaskScheduler(
        config=config, gpu_task_class=GANGenerateDataTask)
    print("Generator has set up GPUTaskScheduler")
    scheduler.start()