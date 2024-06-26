def multinews_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_qy3/share_1565115/jonxie/model_base/bart-base")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_qy3/share_1565115/jonxie/data_base/multi_news")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 777)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 16384)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.chunk_len = getattr(args, 'chunk_len', 512)
    args.gen_max_len = getattr(args, "gen_max_len", 400)
    args.gen_min_len = getattr(args, "gen_min_len", 150)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 2.0)
    args.eval_interval = getattr(args, "eval_interval", 1000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "bart (simcas-multinews)")
    args.desc = getattr(args, 'desc', 'TODO')
    args.is_wandb = getattr(args, 'is_wandb', False)


def wcep_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 1)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_cq2/share_1567347/share_info/model_base/bart-large")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_cq2/share_1567347/jonxie/workspace/data_base/WCEP-10")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 970913)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 6144)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.gen_max_len = getattr(args, "gen_max_len", 40)
    args.gen_min_len = getattr(args, "gen_min_len", 15)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 2.0)
    args.eval_interval = getattr(args, "eval_interval", 400)
    args.save_interval = getattr(args, "save_interval", 40000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "bart (simcas wcep)")
    args.desc = getattr(args, 'desc', '')


def arxiv_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_cq2/share_1567347/share_info/model_base/bart-large")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_cq2/share_1567347/jonxie/workspace/data_base/arxiv-summarization")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 5555)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 6144)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.gen_max_len = getattr(args, "gen_max_len", 300)
    args.gen_min_len = getattr(args, "gen_min_len", 50)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 5.0)
    args.eval_interval = getattr(args, "eval_interval", 2000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "bart (simcas arxiv)")
    args.desc = getattr(args, 'desc', '')


def govreport_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_qy3/share_1565115/jonxie/model_base/bart-base")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_qy3/share_1565115/jonxie/data_base/govreport-summarization")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 777)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 16384)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.chunk_len = getattr(args, 'chunk_len', 512)
    args.gen_max_len = getattr(args, "gen_max_len", 740)
    args.gen_min_len = getattr(args, "gen_min_len", 50)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 2.0)
    args.eval_interval = getattr(args, "eval_interval", 1000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "")
    args.desc = getattr(args, 'desc', 'TODO')
    args.is_wandb = getattr(args, 'is_wandb', False)


def pubmed_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_cq2/share_1567347/share_info/model_base/bart-large")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_cq2/share_1567347/jonxie/workspace/data_base/pubmed-summarization")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 555)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 6144)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.gen_max_len = getattr(args, "gen_max_len", 400)
    args.gen_min_len = getattr(args, "gen_min_len", 40)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 4.0)
    args.eval_interval = getattr(args, "eval_interval", 2000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "bart (simcas pubmed)")
    args.desc = getattr(args, 'desc', '')


def summscreen_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_qy3/share_1565115/jonxie/model_base/bart-base")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_qy3/share_1565115/jonxie/data_base/govreport-summarization")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 777)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 16384)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.chunk_len = getattr(args, 'chunk_len', 512)
    args.gen_max_len = getattr(args, "gen_max_len", 740)
    args.gen_min_len = getattr(args, "gen_min_len", 50)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 2.0)
    args.eval_interval = getattr(args, "eval_interval", 1000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "")
    args.desc = getattr(args, 'desc', 'TODO')
    args.is_wandb = getattr(args, 'is_wandb', False)


def nrtv_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, "report_freq", 50)
    args.accumulate_step = getattr(args, "accumulate_step", 16)
    args.pretrained = getattr(args, "pretrained", None)
    args.model_type = getattr(args, "model_type", "/apdcephfs_qy3/share_1565115/jonxie/model_base/bart-base")
    args.dataset_name = getattr(args, "dataset_name", "/apdcephfs_qy3/share_1565115/jonxie/data_base/govreport-summarization")
    args.warmup_steps = getattr(args, "warmup_steps", 1600)
    args.normalize = getattr(args, "normalize", True)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 777)
    args.max_lr = getattr(args, "max_lr", 4e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "")
    args.max_len = getattr(args, "max_len", 128)
    args.smooth = getattr(args, "smooth", 0.1)
    args.total_len = getattr(args, "total_len", 1024)
    args.do_sample = getattr(args, "do_sample", True)
    args.max_input_len = getattr(args, 'max_input_len', 16384)
    args.max_output_len = getattr(args, 'max_output_len', 1024)
    args.chunk_len = getattr(args, 'chunk_len', 512)
    args.gen_max_len = getattr(args, "gen_max_len", 20)
    args.gen_min_len = getattr(args, "gen_min_len", 1)
    args.adding = getattr(args, "adding", 0)
    args.length_penalty = getattr(args, "length_penalty", 2.0)
    args.eval_interval = getattr(args, "eval_interval", 1000)
    args.save_interval = getattr(args, "save_interval", 10000)
    args.num_beams = getattr(args, "num_beams", 4)
    args.project_name = getattr(args, "project_name", "")
    args.desc = getattr(args, 'desc', 'TODO')
    args.is_wandb = getattr(args, 'is_wandb', False)
   