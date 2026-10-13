# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

class TiledJpegReader:
    """Top-left, 8-bit contiguous JPEG TIFF reader; never allocates full WSI."""

    def __init__(self, path, cache_tiles=32):
        if cache_tiles < 1:
            raise ValueError('cache_tiles must be positive')
        self.path = Path(path)
        self.cache_tiles = cache_tiles
        self.cache = OrderedDict()
        self.stream = None
        with tifffile.TiffFile(self.path) as tif:
            page = tif.pages[0]
            if not page.is_tiled or int(page.compression) != 7 or page.samplesperpixel != 3 or (page.bitspersample != 8) or (int(page.planarconfig) != 1):
                raise ValueError('Requires tiled JPEG, 8-bit, contiguous RGB level0 TIFF')
            orientation = page.tags.get('Orientation')
            if orientation is not None and int(orientation.value) != 1:
                raise ValueError('Unsupported TIFF orientation')
            self.width, self.height = (int(page.imagewidth), int(page.imagelength))
            self.tw, self.th = (int(page.tilewidth), int(page.tilelength))
            self.cols, self.rows = (math.ceil(self.width / self.tw), math.ceil(self.height / self.th))
            self.offsets, self.counts = (tuple(page.dataoffsets), tuple(page.databytecounts))
            self.tables = page.jpegtables
        if len(self.offsets) != self.cols * self.rows or len(self.counts) != self.cols * self.rows:
            raise ValueError('Unexpected TIFF JPEG tile layout')
        self.stream = self.path.open('rb')

    def _tile(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        count = self.counts[index]
        if count <= 0:
            raise ValueError(f'Empty JPEG tile {index}: {self.path}')
        self.stream.seek(self.offsets[index])
        compressed = self.stream.read(count)
        if len(compressed) != count:
            raise ValueError(f'Truncated JPEG tile {index}: {self.path}')
        image = decode_jpeg_tile(compressed, (self.tw, self.th), self.tables)
        pixels = np.asarray(image).copy()
        image.close()
        self.cache[index] = pixels
        if len(self.cache) > self.cache_tiles:
            self.cache.popitem(last=False)
        return pixels

    def crop(self, x, y, size=512):
        if not (0 <= x < self.width and 0 <= y < self.height and (size > 0)):
            raise ValueError('Invalid crop origin or size')
        result = np.full((size, size, 3), 255, dtype=np.uint8)
        right, bottom = (min(x + size, self.width), min(y + size, self.height))
        for ty in range(y // self.th, (bottom - 1) // self.th + 1):
            for tx in range(x // self.tw, (right - 1) // self.tw + 1):
                tile = self._tile(ty * self.cols + tx)
                left, top = (max(x, tx * self.tw), max(y, ty * self.th))
                end_x, end_y = (min(right, (tx + 1) * self.tw), min(bottom, (ty + 1) * self.th))
                result[top - y:end_y - y, left - x:end_x - x] = tile[top - ty * self.th:end_y - ty * self.th, left - tx * self.tw:end_x - tx * self.tw]
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.cache.clear()

    def __del__(self):
        self.close()

def preprocess_crop(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or (rgb.shape[0] != rgb.shape[1]) or (rgb.shape[0] not in (408, 409)):
        raise ValueError('Dense input must be calibrated square RGB uint8 crop of 408/409 pixels')
    pixels = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

class DenseCropDataset(Dataset):

    def __init__(self, rows, max_open_sources=2, cache_tiles=32):
        self.rows = rows
        self.max_open_sources = max_open_sources
        self.cache_tiles = cache_tiles
        self.readers = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = row['source_path']
        if path not in self.readers:
            if len(self.readers) >= self.max_open_sources:
                _, expired = self.readers.popitem(last=False)
                expired.close()
            self.readers[path] = TiledJpegReader(path, self.cache_tiles)
        self.readers.move_to_end(path)
        reader = self.readers[path]
        if (reader.width, reader.height) != (row['source_width'], row['source_height']):
            raise ValueError(f'WSI dimensions changed: {path}')
        return (preprocess_crop(reader.crop(row['x'], row['y'], row['crop_size'])), index)

    def __getstate__(self):
        return {**self.__dict__, 'readers': OrderedDict()}

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __del__(self):
        self.close()

def flips4_probabilities(model, images, amp=False):
    """Average four softmax probabilities, never average logits or features."""
    probabilities = []
    for axes in (None, (-1,), (-2,), (-2, -1)):
        transformed = images if axes is None else torch.flip(images, dims=axes)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp and images.device.type == 'cuda'):
            logits = model(transformed)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError('Expected two-class logits')
            probabilities.append(torch.softmax(logits / 1.0, dim=1))
    return torch.stack(probabilities).mean(0)[:, 0]
