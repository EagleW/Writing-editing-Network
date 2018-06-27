class CommonConfig(object):
    cell = "GRU"
    nlayers = 1
    batch_size = 20
    dataparallel = False
    dropout = 0
    epochs = 20
    bidirectional = True
    max_grad_norm = 10
    min_freq = 5
    num_exams = 3
    log_interval = 1000
    predict_right_after = 3
    patience = 5

class SmallDataset(CommonConfig):
    relative_data_path = '/data/small-json/train.dat'
    relative_dev_path = '/data/small-json/dev.dat'
    relative_test_path = '/data/small-json/test.dat'
    relative_gen_path = '/data/small-json/fake%d.dat'

class LargeDataset(CommonConfig):
    relative_data_path = '/data/large-json/train.dat'
    relative_dev_path = '/data/large-json/dev.dat'
    relative_test_path = '/data/large-json/test.dat'
    relative_gen_path = '/data/large-json/fake%d.dat'

""" Learning Rate Ablation Experiments """
class SmallConfig1(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.001
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.001"

class SmallConfig2(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0005
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0005"

class SmallConfig3(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.00025
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.00025"

class SmallConfig4(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0001"

class SmallConfig5(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0000625
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0000625"

""" Pretrained Word Embeddings Ablation Experiments """
class SmallConfig6(SmallDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = False
    experiment_name = "lr-0.0001-WE-512"

class SmallConfig7(SmallDataset):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = False
    experiment_name = "lr-0.0001-WE-300"

""" Learning Rate Ablation Experiments """
class LargeConfig1(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.001
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.001"

class LargeConfig2(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0005
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0005"

class LargeConfig3(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.00025
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.00025"

class LargeConfig4(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0001"

class LargeConfig5(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0000625
    pretrained = None
    use_topics = False
    experiment_name = "lr-0.0000625"

""" Pretrained Word Embeddings Ablation Experiments """
class LargeConfig6(LargeDataset):
    emsize = 512
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete-512.vec'
    use_topics = False
    experiment_name = "lr-0.0001-WE-512"

class LargeConfig7(LargeDataset):
    emsize = 300
    context_dim = 128
    lr = 0.0001
    pretrained = 'embeddings/complete.vec'
    use_topics = False
    experiment_name = "lr-0.0001-WE-300"

configuration = {"s1": SmallConfig1(),
                 "s2": SmallConfig2(),
                 "s3": SmallConfig3(),
                 "s4": SmallConfig4(),
                 "s5": SmallConfig5(),
                 "s6": SmallConfig6(),
                 "s7": SmallConfig7(),
                 "l1": LargeConfig1(),
                 "l2": LargeConfig2(),
                 "l3": LargeConfig3(),
                 "l4": LargeConfig4(),
                 "l5": LargeConfig5(),
                 "l6": LargeConfig6(),
                 "l7": LargeConfig7()}

def get_conf(name):
    return configuration[name]