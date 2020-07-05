class Workflow(pycnfg.Producer):
    def __hash__(self):
        # Not need for producer
        return hashlib.md5(str(self.params).encode('utf-8')).hexdigest()


    # =============================================== add/pop ============================================================
    def add_dataset(self, dataset, dataset_id):
        """Add dataset to workflow internal storage.

        Args:
            dataset (): .
            dataset_id (str): .

        """
        # [alternative]
        # dataset = self.data_check(dataset)
        self.datasets.update({dataset_id: dataset})
        return

    def pop_dataset(self, dataset_ids):
        """Pop data from wotkflow data storage.

        Args:
            dataset_ids (str, iterable): ids to pop.
        Return:
            popped data dict.
        """
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]
        return {dataset_id: self.datasets.pop(dataset_id, None)
                for dataset_id in dataset_ids}

    def add_pipeline(self, pipeline, pipeline_id):
        """Add pipeline to workflow internal storage.

        Args:
            pipeline (): .
            pipeline_id (str): .

        """
        self.pipelines.update({pipeline_id: pipeline})
        return

    def pop_pipeline(self, pipe_ids):
        """Pop pipeline from wotkflow pipeline storage.

        Args:
            pipe_ids (str, iterable): ids to pop.
        Return:
            popped pipelines dict.
        """
        if isinstance(pipe_ids, str):
            pipe_ids = [pipe_ids]
        return {pipe_id: self.pipelines.pop(pipe_id, None)
                for pipe_id in pipe_ids}

# [deprecated] hard-code type always better.
#     def _check_arg(self, arg, func=None):
#         """Check if argument is id or object.
#
#         Example
#         pipeline_id = self._check_arg(pipeline)
#         dataset_id = self._check_arg(dataset)
#         """
#         if isinstance(arg, str):
#             return arg
#         else:
#             assert False, 'Argument should be str'
#             # TODO[beta]: пока оставлю так,пусть дата и пайплайн всегда через config задаются, потом можно расширить
#             # fit() принимает pipeline_id вместо  pipeline, но read_conf резолвит pipeline
#             # вообще у даты и пайплайн особый статус, но с другими параметрами должна быть синхронность
#             # Лучше так: можно и id  и напрямую пайплайн, дату, это будет логично.
#             # тогда не будет отличатся от других. Только там внутри есть хранилища зависимые от айдишников.
#             # надо дефолтный айди тогда создавать!
#             # pipeline should contain pipeline.pipeine
#
#             # Generate arbitrary.
#             # id = str(int(time.time()))
#             # Add to storage under id.
#             # func(arg, id)
#             # return id

    # =============================================== load ==========================================================
    # [deprecated] there special class to load pipeline and set
    # def load(self, file):
    #     """Load fitted model on disk/string.

    #     Note:
    #         Better use only the same version of sklearn.

    #     """
    #     self.logger.info("\u25CF LOAD MODEL")
    #     pipeline = joblib.load(file)
    #     self.logger.info('Load fitted model from file:\n    {}'.format(file))

    #     # alternative
    #     # with open(f"{self.project_path}/sump.model", 'rb') as f:
    #     #     self.estimator = pickle.load(f)

    # [deprecated] explicit param to resolve in hp_grid
    # def _set_hps(self, pipeline, data, kwargs):
    #     hps = pipeline.get_params().update(self._get_zero_position(kwargs))
    #     hps = self._resolve_hps(hps, data, kwargs)
    #     pipeline.set_params(**hps)
    #     return pipeline

    # def _resolve_hps(self, hps, data, kwargs):
    #     for hp_name in hps:
    #         # step_name = step[0]
    #         # step_hp = {key: p[key] for key in p.keys() if step_name + '__' in key}
    #         val = hps[hp_name]
    #         if self._is_data_hp(val):
    #             key = val.split('__')[-1]
    #             hps[hp_name] = self.get_from_data_(data, key)
    #         elif isinstance(val, dict):
    #             # dict case
    #             for k,v in val.items():
    #                 if self._is_data_hp(v):
    #                     key = v.split('__')[-1]
    #                     val[k] = self.get_from_data_(data, key)
    #         elif hasattr(type(val), '__iter__') and\
    #                 hasattr(type(val), '__getitem__'):
    #             # sequence case
    #             for k, v in enumerate(val):
    #                 if self._is_data_hp(v):
    #                     key = v.split('__')[-1]
    #                     val[k] = self.get_from_data_(data, key)
    #     return hps
    # def _is_data_hp(self, val):
    #     return isinstance(val, str) and val.startswith('data__')



