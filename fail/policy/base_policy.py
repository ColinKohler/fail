from fail.utils.module_attr_mixin import ModuleAttrMixin


class BasePolicy(ModuleAttrMixin):
    def __init__(self, action_dim, seq_len, z_dim):
        super().__init__()

        self.action_dim = action_dim
        self.seq_len = seq_len
        self.z_dim = z_dim
