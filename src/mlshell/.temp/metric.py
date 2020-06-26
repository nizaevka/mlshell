#TODO: [deprecated]
class ExtendedScorer(object):
    def __init__(self, scorer):
        # Scorer to extend.
        self.scorer = scorer

        # [deprecated] now not memorize state, if no pass_csutom step, use default.
        # Last kwargs state to use in score for second stage optimizers.
        # self.cache_custom_kwargs = {}
        # TODO: here is actually problem when multiple pipeline are used.
        #   it is better to inheret pass_custom step for next level
        #   so the attribute error will never rise.

        self.init_kwargs = self.scorer._kwargs


    def __call__(self, estimator, x, y, **kwargs):
        """Read custom_kwargs from current pipeline, pass to scorer.

        Note:
            In gs self object copy, we can dynamically get param only from estimator.

            Use initial kwargs for score if:
                pipeline not contain steps
                no `pass_custom`
                kwargs empty {}.

        """
        # Use initial kwargs for score if pipeline not contain steps.
        self.scorer._kwargs.update(self.init_kwargs)
        if hasattr(estimator, 'steps'):
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kwargs:
                    self.scorer._kwargs.update(estimator.steps[0][1].kwargs)
        # [deprecated] need tests.
        # except AttributeError:
        #     # ThresholdClassifier object has no attribute 'steps'.

        #     # [deprecated] Now use init kwargs in score,
        #     #   not last if no step or `pass_custom`.
        #     # self.scorer._kwargs.update(self.cache_custom_kwargs)
        #     pass

        return self.scorer(estimator, x, y, **kwargs)