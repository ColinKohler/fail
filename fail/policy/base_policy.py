from fail.utils.module_attr_mixin import ModuleAttrMixin
from fail.utils.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    def __init__(self, robot_state_dim, world_state_dim, action_dim, num_robot_state, num_world_state, num_action_steps):
        super().__init__()

        self.robot_state_dim = robot_state_dim
        self.world_state_dim = world_state_dim
        self.action_dim = action_dim
        self.num_robot_state = num_robot_state
        self.num_world_state = num_world_state
        self.num_action_steps = num_action_steps

        self.normalizer = LinearNormalizer()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
