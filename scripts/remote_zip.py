"""Read selected ZIP members over HTTP Range, without downloading whole datasets."""
import io
import re
import urllib.request
import zipfile


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        _, content_range = self._request(0, 0)
        self.size = int(content_range.rsplit('/', 1)[1])

    def _request(self, start, end):
        separator = '&' if '?' in self.url else '?'
        url = f'{self.url}{separator}phyrc_range={start}-{end}'
        request = urllib.request.Request(url, headers={
            'Range': f'bytes={start}-{end}', 'User-Agent': 'PhyRC-asset-installer/1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            content_range = response.headers.get('Content-Range', '')
            expected = f'bytes {start}-{end}/'
            if response.status != 206 or not content_range.startswith(expected):
                raise RuntimeError(f'Server does not honor HTTP Range: {content_range!r}. '
                                   'Use the manual archive instructions in README.md.')
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError('Incomplete HTTP Range response')
        return data, content_range

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        data, _ = self._request(self.position, self.position + size - 1)
        self.position += len(data)
        return data


def open_remote_zip(url):
    return zipfile.ZipFile(HTTPRangeReader(url))


if __name__ == '__main__':
    import sys
    with open_remote_zip(sys.argv[1]) as archive:
        pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else '.')
        for entry in archive.infolist():
            if pattern.search(entry.filename):
                print(entry.file_size, entry.filename)
