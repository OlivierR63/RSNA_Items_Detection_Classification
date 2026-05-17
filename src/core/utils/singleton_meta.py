# coding: utf-8

class SingletonMeta(type):
    """
    Metaclass for implementing the pattern Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # The instance does not exist yet, so it is created
            instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
