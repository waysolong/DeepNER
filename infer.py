import time
import os
import json
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
from src.utils.evaluator import crf_evaluation, span_evaluation, mrc_evaluation
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.preprocess.processor import NERProcessor, convert_examples_to_features
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, dev_examples=None):
    with open(os.path.join(opt.mid_data_dir, f'{opt.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    if opt.task_type == 'crf':
        model = build_model('crf', opt.bert_dir, num_tags=len(ent2id),
                            dropout_prob=opt.dropout_prob)
    elif opt.task_type == 'mrc':
        model = build_model('mrc', opt.bert_dir,
                            dropout_prob=opt.dropout_prob,
                            use_type_embed=opt.use_type_embed,
                            loss_type=opt.loss_type)
    else:
        model = build_model('span', opt.bert_dir, num_tags=len(ent2id)+1,
                            dropout_prob=opt.dropout_prob,
                            loss_type=opt.loss_type)


    if dev_examples is not None:
        
        dev_features, _ = convert_examples_to_features(opt.task_type, dev_examples,
                                                                       opt.max_seq_len, opt.bert_dir, ent2id,"infer")

        dev_dataset = NERDataset(opt.task_type, dev_features, 'dev', use_type_embed=False)

        dev_loader = DataLoader(dev_dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        dev_info = (dev_loader,dev_examples[0])

        model_path_list = get_model_path_list(opt.output_dir)

        metric_str = ''

        max_f1 = 0.
        max_f1_step = 0

        max_f1_path = ''

        for idx, model_path in enumerate(model_path_list):

            tmp_step = model_path.split('/')[-2].split('-')[-1]


            model, device = load_model_and_parallel(model, '0',
                                                    ckpt_path=model_path)

            if opt.task_type == 'crf':
                crf_evaluation(model, dev_info, device, ent2id)
            elif opt.task_type == 'mrc':
                mrc_evaluation(model, dev_info, device)
            else:
                res = span_evaluation(model, dev_info, device, ent2id,mode="one")
                print(res)


def training(opt):

    dev_examples = ["6g*12袋  国家医保目录（乙类）  1.收缩子宫：新生化颗粒使DNA含量和子宫利用葡萄糖能力增加，促进子宫蛋白质合成及子宫增生，以促进子宫收缩，从而起到止血并排出瘀血的目的。实验室研究表明，新生化颗粒能明显增加大鼠离体子宫的收缩张力、收缩频率和收缩振幅，且呈剂量依赖性关系。冲洗药液后，子宫活动仍可恢复到正常状态。2.镇痛：实验室研究表明，新生化颗粒能明显减少大鼠扭体次数。3.抗血小板凝聚及抗血栓作用：新生化颗粒能抑制血小板聚集促进剂(H-SHT)产生。血液流变学表明，新生化颗粒通过降低血浆纤维蛋白原浓度，增加血小板细胞表面电荷，促进细胞解聚，降低血液粘度，达到抗血栓形成的作用。从而使瘀血不易凝固而利于排出。4.造血和抗贫血作用：新生化颗粒能促进血红蛋白(Hb)和红细胞(RBC)的生成。对造血干细胞(CFU&mdash;S)增值有显著的刺激作用，并能促进红系细胞分化。粒单细胞(CFU&mdash;D)、红系(BFU&mdash;E)祖细胞的产率均有明显升高作用。新生化颗粒同时还能抑制补体(c3b)与红细胞膜结合，降低补体溶血功能。5.改善微循环：增加子宫毛细血管流量，促进子宫修复。6.抗炎：新生化颗粒有很好的抗炎抑菌作用。体外试验表明，新生化颗粒对痢疾杆菌、大肠杆菌、绿脓杆菌、变形杆菌和金黄色葡萄球菌均有很好的抑菌作用。  亚宝药业大同制药有限公司  本品为黄棕色至黄褐色的颗粒；味甘、微苦。  活血、祛瘀、止痛。用于产后恶露不行，小腹疼痛，也可试用于上节育环后引起的阴道流血，月经过多 尚不明确。  6g*12袋/盒。  热水冲服，一次2袋，一日2-3次。  用于产后恶露不行，少腹疼痛，也可用于上节育环后引起的阴道流血，月经过多  尚不明确。  尚不明确"]
    dev_examples = ["阿莫西林胶囊消炎止痛，不要吃苹果等，容易犯困，促进血液循环，从而起到止血并排出瘀血的目的"]
    dev_examples = ["本品为胶囊剂，每次一粒，味甘微苦，阿莫西林胶囊消炎止痛，不要吃苹果等，容易犯困，促进血液循环，从而起到止血并排出瘀血的目的，孕妇不能吃"]

    train_base(opt, dev_examples)

if __name__ == '__main__':
    start_time = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')
    parser.add_argument('--output_dir', default='out/roberta_wwm_ls_ce_span',
                            help='the output dir for model checkpoints')
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--task_type', default='span')
    parser.add_argument('--bert_type', default='checkpoint-2448')
    parser.add_argument('--bert_dir', default='bert/torch_roberta_wwm')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--mid_data_dir', default='./data/mid_data',
                            help='the mid data dir')
    parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
    parser.add_argument('--loss_type', default='ls_ce',
                            help='loss type for span bert')
    args = parser.parse_args()

    assert args.mode in ['eval', 'stack'], 'mode mismatch'
    assert args.task_type in ['crf', 'span', 'mrc']

    args.output_dir = os.path.join(args.output_dir, args.bert_type)

    set_seed(args.seed)


    # if args.task_type == 'mrc':
    #     if args.use_type_embed:
    #         args.output_dir += f'_embed'
    #     args.output_dir += f'_{args.loss_type}'

    # args.output_dir += f'_{args.task_type}'

    args.output_dir.replace('/', '\\')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'{args.mode} {args.task_type} in max_seq_len {args.max_seq_len}')

    if args.mode == 'eval':
        training(args)

    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
