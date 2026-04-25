import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--vocab_size', default=50000, type=int)
    parser.add_argument('--gpu', default='0', help='gpu id')
    
    # 模型训练保存基础目录
    dir_prefix = 'saved/'
    # 不同模型保存目录
    model_dir = dir_prefix + 'train_with_no_pro_doc/'
    # 训练轮数
    num_epochs = 3
    # 训练批次大小
    batch_size = 64
    # 是否使用项目特定信息
    use_project_info = False
    # 是否使用文档特定信息
    use_docstring = False

    '''
    以下为可配置项，其他参数不建议修改！！！
    '''
    # ======================== 可配置项 ========================

    # ========== 训练超参数 ==========
    # 训练批次大小
    parser.add_argument('--batch_size', default=batch_size, type=int)
    # 验证批次大小
    parser.add_argument('--eval_batch_size', default=128, type=int)
    # 日志目录
    parser.add_argument('--logdir', default=model_dir+'log/', help="log directory")
    # 训练轮数
    parser.add_argument('--num_epochs', default=num_epochs, type=int)
    # 保存轮数
    parser.add_argument('--save_epochs', default=3, type=int)
    # 验证目录
    parser.add_argument('--evaldir', default="data_processed/", help="evaluation dir")

    # 是否使用项目特定信息
    parser.add_argument('--pro', default=use_project_info, type=bool,
                        help="whether to use project-specific info")
    # 是否使用文档特定信息
    parser.add_argument('--use_docstring', default=use_docstring, type=bool,
                        help="whether to use docstring information")
    
    # ========== 测试超参数 ==========
    # 测试批次大小
    parser.add_argument('--test_batch_size', default=128, type=int)
    # 测试目录
    parser.add_argument('--testdir', default="data_processed/", help="test result dir")
    # 结果目录
    parser.add_argument('--res_log', default=model_dir+'res.txt', help="result dir")

    # ========== 检查点文件路径 ==========
    parser.add_argument('--ckpt', default=model_dir, help="checkpoint file path")

    # ======================== 可配置项 ========================

    # ========== 数据路径 ==========
    # 词表文件路径
    parser.add_argument('--sub_word_vocab_file', default='data_processing/sub_token_w2id.txt',
                        help="vocabulary file path")
    parser.add_argument('--doc_vocab_file', default='data_processing/doc_w2id.txt',
                        help="vocabulary file path")
    # 训练数据路径
    parser.add_argument('--data_path', default='data_processed/',
                        help="data path")

    '''
    以下为不建议修改项！！！
    '''
    # ====================== 以下配置项不建议修改 ======================
    # 学习率相关
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    # 预热步数
    parser.add_argument('--warmup_steps', default=4000, type=int)
    # ========== 上下文大小 ==========
    # 主体上下文大小
    parser.add_argument('--body_context_size', default=55, type=int,
                        help="body_context_size")
    # 项目上下文大小
    parser.add_argument('--project_context_size', default=60, type=int,
                        help="project_context_size")
    # 文档上下文大小
    parser.add_argument('--doc_context_size', default=10, type=int,
                        help="doc_context_size")
    # 目标名称上下文大小
    parser.add_argument('--tgt_name_size', default=5, type=int,
                        help="tgt_name_size")

    # ========== 模型超参数 ==========
    # 模型维度
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    # 前馈层维度
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    # 编码器/解码器块数
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    # 注意力头数
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    # dropout率
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    # label smoothing率
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")
