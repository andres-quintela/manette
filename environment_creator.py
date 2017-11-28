class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        if args.emulator_name == 'ALE' :
        """ Creates an object from which new environments can be created.
            Param : args. """
        if args.game == 'tetris' :
            from tetris_emulator import TetrisEmulator
            self.num_actions = 5
            self.create_environment = lambda i: TetrisEmulator(i, args)
        else :
            from atari_emulator import AtariEmulator
            from ale_python_interface import ALEInterface
            filename = args.rom_path + "/" + args.game + ".bin"
            ale_int = ALEInterface()
            ale_int.loadROM(str.encode(filename))
            self.num_actions = len(ale_int.getMinimalActionSet())
            self.create_environment = lambda i: AtariEmulator(i, args)

            from atari_emulator import AtariEmulator
            from ale_python_interface import ALEInterface
            filename = args.rom_path + "/" + args.game + ".bin"
            ale_int = ALEInterface()
            ale_int.loadROM(str.encode(filename))
            self.num_actions = len(ale_int.getMinimalActionSet())
            self.create_environment = lambda i: AtariEmulator(i, args)

        else :

            from gym_emulator import GymEmulator
            import gym
            self.create_environment = lambda i: GymEmulator(i, args)
            env_test = gym.make(args.game)
            self.num_actions = env_test.action_space.n
