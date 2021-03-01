DEFAULT_PORT = 42523

from zprocess import ZMQClient
from fib_daqmx_scanner import DEFAULT_PORT


class Client(ZMQClient):
    """A ZMQClient for communication with runmanager"""

    def __init__(self, host=None, port=DEFAULT_PORT, timeout=60):
        ZMQClient.__init__(self)
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, command, *args, **kwargs):
        return self.get(
            self.port, self.host, data=[command, args, kwargs], timeout=self.timeout
        )

    def say_hello(self):
        """Ping the runmanager server for a response"""
        return self.request('hello')

    def get_version(self):
        """Return the version of runmanager the server is running in"""
        return self.request('__version__')

    def get_count_rate(self, npts=10):
        """Get the count rate, averaged over the next npts samples acquired. Return the
        mean and standard error in the mean"""
        return self.request('get_count_rate', npts=npts)


_default_client = Client()

get_count_rate = _default_client.get_count_rate

if __name__ == '__main__':
    # Test
    print("get count: rate", get_count_rate())
