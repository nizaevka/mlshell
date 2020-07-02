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
