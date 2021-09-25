from common_config import config


def main():
    if config.model == 'GDGAN':
        from GDGAN.gdgan import GDGAN
        from GDGAN.gdgan_config import validate_args
        config.y1_indices = [int(i) for i in config.y1_indices.split(',')]
        config.y2_indices = [int(i) for i in config.y2_indices.split(',')]
        gan = GDGAN(config)
    print("finally!!")
    gan.run()


if __name__ == '__main__':
    main()
