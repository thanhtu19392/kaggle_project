from sklearn.base import BaseEstimator, TransformerMixin


class _TransparentMixin(object):

    def _get_delegate_params(self):
        return self.delegate.get_params()

    def get_params(self, deep=True):
        parent = super(_TransparentMixin, self)
        if hasattr(parent, 'get_params'):
            out = parent.get_params(deep=False)
        else:
            out = {'delegate': self.delegate}
        if deep:
            out.update(self._get_delegate_params())
        return out

    def _set_delegate_params(self, **kwargs):
        self.delegate.set_params(**kwargs)

    def set_params(self, **kwargs):
        parent = super(_TransparentMixin, self)
        if hasattr(parent, 'get_params'):
            self_params = parent.get_params(deep=False)
            self_update = {k: v for k, v in kwargs.items() if k in self_params}
            parent.set_params(**self_update)
        else:
            self_params = {'delegate': self.delegate}
        delegate_update = {k: v for k, v in kwargs.items() if k not in self_params}
        self._set_delegate_params(**delegate_update)
        return self

    def getname(self):
        if hasattr(self.delegate, 'getname'):
            return self.delegate.getname()
        return type(self.delegate).__name__.lower()


class _Delegate(_TransparentMixin, BaseEstimator, TransformerMixin):

    def __init__(self, delegate):
        self.delegate = delegate

    def fit(self, X, y=None):
        self.delegate.fit(X, y)
        return self

    def transform(self, X):
        return self.delegate.transform(X)

    # Override
    def _set_delegate_params(self, **kwargs):
        if kwargs:
            raise Exception("Cannot modify freezed (fitted) object")
            # self.delegate.delegate.set_params(**kwargs)

    # Override
    def _get_delegate_params(self):
        return self.delegate.delegate.get_params(True)
