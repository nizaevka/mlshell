class MockOptimizer():
    pass
# [deprecated] unfinished, only single case. find better logic
# Default (if not hp_grid).
# mock_pipeline = pipeline
# params = pipeline.get_params()
# for hp_name in hp_grid:  # 'a__b__c'
#     # Copy whole original step, contain hp_name last parameter
#     # and recover upstream name structure (envelop all except last).
#     # If only one subname, copy whole pipeline.
#     lis = hp_name.split('__')  # ['a', 'b', 'c']
#     step_name = '__'.join(lis[:-1])  # ['a', 'b']
#
#     if step_name:
#         for subname in lis[-2::-1]:  # ['b', 'a']
#             mock_pipeline = sklearn.pipeline.Pipeline(
#                 steps=[(subname, params[step_name])])
#     else:
#         mock_pipeline = pipeline
    def runs_compliance(self, runs, runs_th_, best_index):
        """"Combine GS results to csv dump."""
        # runs.csv compliance
        # add param
        default_th = self.pipeline.get_params()['estimate__apply_threshold__threshold']
        runs['param_estimate__apply_threshold__threshold'] = np.full(len(runs['params']), fill_value=default_th)
        runs_th_['param_estimate__apply_threshold__threshold'] =\
            runs_th_.pop('param_threshold', np.full(len(runs_th_['params']),
                                                    fill_value=default_th))
        # update runs.params with param
        for run in runs['params']:
            run['estimate__apply_threshold__threshold'] = default_th
        for run in runs_th_['params']:
            run.update(runs['params'][best_index])
            run['estimate__apply_threshold__threshold'] = run.pop('threshold', default_th)
        # add all cv_th_ runs as separate rows with optimizer.best_params_ default values
        runs_df = pd.DataFrame(runs)
        runs_th_df = pd.DataFrame(runs_th_)
        sync_columns = [column for column in runs_th_df.columns if not column.endswith('time')]

        runs_df = runs_df.append(runs_th_df.loc[:, sync_columns], ignore_index=True)
        # replace Nan with optimizer.best_params_
        runs_df.fillna(value=runs_df.iloc[best_index], inplace=True)
        return runs_df

    # [deprecated] params only contains modifiers, not need full take full params.
    for run in optimizer.cv_results_['params']:
        # extended = self.pipeline.get_params().update(run)
        # run.update(extended)
        # Remove mock estimator.
        for key in run:
            if key.startswith('estimate_del'):
                del run[key]
    for attr, val in {'estimator': self.pipeline,
                      'best_estimator_': self.pipeline,
                      'cv_results': optimizer.cv_results}.items():