from dataclasses import dataclass
from beir.retrieval.evaluation import EvaluateRetrieval
from helpers.helper import *


@dataclass
class Evaluator:
    task: str
    golden_qrels: dict
    data_dir: str
    encoder_suffix: str = ''
    evaluator : EvaluateRetrieval = EvaluateRetrieval()

    def evaluate(self, result_path_generator, add_task_prefix=False):
        return_dict = dict()
        for encoder in encoders:
            result_path = result_path_generator(encoder)

            try:
                input_file = os.path.join(self.data_dir,
                                          encoder + self.encoder_suffix, result_path)
                input_df = get_records(input_file)

            except Exception as e:
                print(f"Error: {e}")
                print(f'Encoder {encoder} does not exist.')
                continue

            qid_pids_dict = {}

            for record in input_df:
                qid = str(record['qid'])
                if add_task_prefix:
                    qid = self.task + '-' + qid
                pid = record['pid']
                score = record['score']

                if 'prop' in result_path:
                    passage_id = '-'.join(pid.split('-')[:-2])
                else:
                    passage_id = '-'.join(pid.split('-')[:-1])

                if qid not in qid_pids_dict:
                    qid_pids_dict[qid] = {passage_id: score}
                else:
                    if passage_id not in qid_pids_dict[qid] or qid_pids_dict[qid][passage_id] < score:
                        qid_pids_dict[qid][passage_id] = score

            topks = [1, 5, 10, 20]

            tmp_result = self.evaluator.evaluate(self.golden_qrels, qid_pids_dict, topks)
            return_dict[encoder] = {
                'NDCG@5': tmp_result[0]['NDCG@5'],
                'NDCG@20': tmp_result[0]['NDCG@20'],
                'Recall@5': tmp_result[2]['Recall@5'],
                'Recall@20': tmp_result[2]['Recall@20'],
            }

        self.aggregate_result(return_dict)

        return return_dict
    
    def evaluate_bm25(self, bm25_file):
        return_dict = dict()
        input_file = os.path.join(self.data_dir, bm25_file)
        input_df = get_records(input_file)

        qid_pids_dict = {}

        for record in input_df:
            qid = str(record['qid'])
            pid = record['pid']
            score = record['score']
            passage_id = '-'.join(pid.split('-')[:-1])

            if qid not in qid_pids_dict:
                qid_pids_dict[qid] = {passage_id: score}
            else:
                if passage_id not in qid_pids_dict[qid] or qid_pids_dict[qid][passage_id] < score:
                    qid_pids_dict[qid][passage_id] = score

        topks = [1, 5, 10, 20]

        tmp_result = self.evaluator.evaluate(self.golden_qrels, qid_pids_dict, topks)
        return_dict = {
            'NDCG@5': tmp_result[0]['NDCG@5'],
            'NDCG@20': tmp_result[0]['NDCG@20'],
            'Recall@5': tmp_result[2]['Recall@5'],
            'Recall@20': tmp_result[2]['Recall@20'],
        }

        return return_dict

    def aggregate_result(self, _dict):
        """
        dict : {
            key (ance) : (
                {
                    "NDCG@1":
                    ...
                },
                {
                    "MAP@1":
                    ...
                },
                {
                    "Recall@5":
                    ...
                },
                {
                    "P@1"
                    ...
                }
            )
        }
        """
        ret_dict = defaultdict(list)
        for encoder in encoders:
            try:
                ret_dict['NDCG@5'].append(_dict[encoder]['NDCG@5'])
                ret_dict['NDCG@20'].append(_dict[encoder]['NDCG@20'])
                ret_dict['Recall@5'].append(_dict[encoder]['Recall@5'])
                ret_dict['Recall@20'].append(_dict[encoder]['Recall@20'])
            except Exception as e:
                print(e)
    
        mean_dict = dict()
        for k, v in ret_dict.items():
            mean_dict[k] = np.mean(v)

        _dict['avg'] = mean_dict

    @classmethod
    def display_metrics(cls, list_of_dict, metric_keys=['NDCG@5', 'NDCG@20', 'Recall@5', 'Recall@20']):
        for _key in metric_keys:
            for encoder in encoders:
                try:
                    print(_key)
                    print(encoder)
                    print("|".join([str(percent_round(_dict[encoder][_key])) for _dict in list_of_dict]))
                except Exception as e:
                    print(e)

            print("avg")
            print("|".join([str(percent_round(_dict['avg'][_key])) for _dict in list_of_dict]))
        

            print('==============')

    def latex_generation(cls, list_of_list_of_dict, metric_keys=['NDCG@5', 'NDCG@20']):
        """
        [[dict]] inner is for different setups, outer is for different dataset
        {'ance': {'NDCG@5': 0.41538,
        'NDCG@20': 0.45296,
        'Recall@5': 0.51172,
        'Recall@20': 0.63051}}
        """
        lines = []
        setups = [
            '$s_{q\text{-}d}$',
            '$s_{q\text{-}p}$',
            '$s_{s\text{-}p}$',
            '\shorttitle',
        ]
        for encoder in encoders:
            for setup_idx in range(4):
                tmp_arr_5 = []
                tmp_arr_20 = []
                for list_of_dict in list_of_list_of_dict:
                    tmp_arr_5.append(list_of_dict[setup_idx][encoder]['NDCG@5'])
                    tmp_arr_20.append(list_of_dict[setup_idx][encoder]['NDCG@20'])
                mean_5 = np.mean(tmp_arr_5)
                mean_20 = np.mean(tmp_arr_20)
                line = f'{encoder} & {setups[setup_idx]} '
                for (ndcg5, ndcg20) in zip(tmp_arr_5, tmp_arr_20):
                    line += f'& {percent_round(ndcg5)} & {percent_round(ndcg20)}'
                line += f'& {percent_round(mean_5)} & {percent_round(mean_20)}'
                lines.append(line)

        return lines