DEFAULT_PORT = 42523

from zprocess import ZMQClient
from fib_daqmx_scanner import DEFAULT_PORT


class Client(ZMQClient):
    """A ZMQClient for communication with runmanager"""

    def __init__(
        self,
        host='localhost',
        port=DEFAULT_PORT,
        timeout=60,
        allow_insecure=False,
        shared_secret=None
    ):
        ZMQClient.__init__(
            self, allow_insecure=allow_insecure, shared_secret=shared_secret
        )
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

    def get_dwell_time(self):
        return self.request('get_dwell_time')

    def get_sample_rate(self):
        return self.request('get_sample_rate')

    def acquire(self, npts=10):
        """acquire and return the next npts points of the faraday cup current,
        target current, and count rate, as arrays."""
        return self.request('acquire', npts=npts)

    def do_scan(self):
        """Do a scan with current settings and return the image"""
        return self.request('do_scan')

    def set_range_fractional(self, xmin, xmax, ymin, ymax):
        """Set the new view range as a fraction of the maximum view range.
        That is, xmin, xmax, ymin, ymax must be between 0 and 1"""
        return self.request('set_range_fractional',  xmin, xmax, ymin, ymax)

    def set_resolution(self, nx, ny):
        """Set the number of scan points in the x and y directions"""
        return self.request('set_resolution', nx, ny)


# not general, do a different way if packaging this as an app, config file or something
from pathlib import Path
this_folder = Path(__file__).absolute().parent
shared_secret = (this_folder.parent / "zpsecret-23ee8167.key").read_text()

_default_client = Client(shared_secret=shared_secret)

get_count_rate = _default_client.get_count_rate
acquire = _default_client.acquire
do_scan = _default_client.do_scan
set_range_fractional = _default_client.set_range_fractional
set_resolution = _default_client.set_resolution
get_dwell_time = _default_client.get_dwell_time
get_sample_rate = _default_client.get_sample_rate

if __name__ == '__main__':
    # Test
    print("get count: rate", get_count_rate())
