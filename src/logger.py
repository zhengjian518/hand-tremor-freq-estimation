# -*- coding: utf-8 -*-
import logging

class Logger():
    """Logger Class."""

    def __init__(self,logger_name=__name__,level_name='info'):
        """Init logger configuration.

        Args:
            logger_name: a string indicating logger name.
            level_name: a string indicating level name, choices including 
                        'debug','info','ẃarning','error','crtical'.
        """

        LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

        level = LEVELS.get(level_name, logging.NOTSET)
        self.LEVELNAME = level_name
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def info(self, msg):
        """Log information.

        Args:
            msg: a string presenting info message.
        """
        self._logger.info(msg)

    def debug(self, msg):
        """Log debug information.

        Args:
            msg: a string presenting debug message.
        """
        self._logger.debug(msg)

    def warn(self, msg):
        """Log warning.

        Args:
            msg: a string presenting warning message.
        """        
        self._logger.warn(msg)

