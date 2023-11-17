from fail.utils.module_attr_mixin import ModuleAttrMixin
from fail.utils.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    def __init__(self, action_dim, seq_len, z_dim):
        super().__init__()

        self.action_dim = action_dim
        self.seq_len = seq_len
        self.z_dim = z_dim

        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
