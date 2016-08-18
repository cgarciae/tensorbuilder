import tflearn

builders_blacklist = []

def patch_classes(Builder, BuilderTree, Applicative):
    Builder.register_map_method(tflearn.layers.conv.max_pool_2d, "tflearn.layers")
    Builder.register_map_method(tflearn.embedding, "tflearn.layers.embedding_ops")
