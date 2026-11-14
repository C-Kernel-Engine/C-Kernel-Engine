"""Bounded uploaded-image contract for a future native multimodal session.

This module performs no filesystem/network reads and does not enable vision on
text sessions. Clients upload bytes as data URLs. Original part order and image
bytes are preserved; decoding is validation only, not model preprocessing.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import io
import warnings
from typing import Any


class VisionInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VisionLimits:
    max_images: int = 1
    max_image_bytes: int = 8 * 1024 * 1024
    max_total_image_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 16 * 1024 * 1024
    max_total_pixels: int = 16 * 1024 * 1024
    max_text_bytes: int = 64 * 1024
    max_parts: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('vision limits must be positive integers')


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime_type: str
    width: int
    height: int
    sha256: str


@dataclass(frozen=True)
class VisionInput:
    parts: tuple[TextPart | ImagePart, ...]
    image_bytes: int
    pixels: int
    text_bytes: int


def _fail(code: str, message: str):
    raise VisionInputError(code, message)


def _image(url: Any, limits: VisionLimits, bytes_left: int, pixels_left: int) -> ImagePart:
    if not isinstance(url, str):
        _fail('invalid_image', 'image_url must be a data URL string')
    # Bound encoded input before splitting or allocating decoded bytes.
    byte_limit = min(limits.max_image_bytes, bytes_left)
    if len(url) > 4 * ((byte_limit + 2) // 3) + 32:
        _fail('image_limit', 'encoded image exceeds remaining byte budget')
    header, sep, encoded = url.partition(',')
    formats = {'data:image/png;base64': ('image/png', 'PNG'),
               'data:image/jpeg;base64': ('image/jpeg', 'JPEG')}
    if not sep or header not in formats:
        _fail('unsupported_image_source', 'upload PNG/JPEG bytes as a base64 data URL; paths and remote URLs are not fetched')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        _fail('invalid_image', 'invalid base64 image')
    if not data or len(data) > byte_limit:
        _fail('image_limit', 'image exceeds remaining byte budget or is empty')
    mime, expected = formats[header]
    try:
        from PIL import Image
    except ImportError as error:
        raise VisionInputError('image_decoder_unavailable', 'Pillow is required for image validation') from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if image.format != expected:
                    _fail('invalid_image', 'declared MIME type does not match decoded image')
                if width <= 0 or height <= 0 or width * height > min(limits.max_pixels, pixels_left):
                    _fail('image_limit', 'image exceeds remaining pixel budget')
                if getattr(image, 'n_frames', 1) != 1:
                    _fail('unsupported_image', 'animated or multi-frame images are unsupported')
                image.verify()
            # verify checks container integrity; load also detects truncated pixels.
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except VisionInputError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise VisionInputError('invalid_image', 'image cannot be decoded safely') from error
    return ImagePart(data, mime, width, height, hashlib.sha256(data).hexdigest())


def prepare_vision_input(content: Any, *, limits: VisionLimits = VisionLimits()) -> VisionInput:
    """Validate one ordered message without reordering or flattening its parts.

    Supports Responses input_text/input_image and Chat Completions text/image_url
    forms. This does not tokenize, render Jinja, resize, or certify a model. HTTP
    callers must additionally bound request bodies and concurrent uploads before
    parsing JSON; these limits do not undo allocations made by a caller.
    """
    if not isinstance(content, list) or not 1 <= len(content) <= limits.max_parts:
        _fail('invalid_content', 'expected a bounded nonempty list of ordered parts')
    parts: list[TextPart | ImagePart] = []
    total_bytes = total_pixels = text_bytes = images = 0
    for part in content:
        if not isinstance(part, dict):
            _fail('invalid_content', 'content parts must be objects')
        kind = part.get('type')
        allowed = {'type', 'text'} if kind in ('text', 'input_text') else {'type', 'image_url', 'detail'}
        if set(part) - allowed:
            _fail('invalid_content', 'unsupported content fields must not be silently ignored')
        if kind in ('text', 'input_text'):
            text = part.get('text')
            if not isinstance(text, str):
                _fail('invalid_content', 'text must be a string')
            # Character count is a cheap lower bound before UTF-8 allocation.
            if len(text) > limits.max_text_bytes - text_bytes:
                _fail('text_limit', 'text exceeds byte budget')
            try:
                text_bytes += len(text.encode('utf-8'))
            except UnicodeEncodeError:
                _fail('invalid_content', 'text contains invalid Unicode')
            if text_bytes > limits.max_text_bytes:
                _fail('text_limit', 'text exceeds byte budget')
            parts.append(TextPart(text))
        elif kind in ('input_image', 'image_url'):
            images += 1
            if images > limits.max_images:
                _fail('image_limit', 'too many images for this contract')
            url = part.get('image_url')
            if isinstance(url, dict):
                if set(url) - {'url', 'detail'}:
                    _fail('invalid_image', 'unsupported image URL options')
                if 'detail' in part and 'detail' in url and part['detail'] != url['detail']:
                    _fail('invalid_image', 'conflicting detail modes')
                detail = url.get('detail', part.get('detail', 'auto'))
                url = url.get('url')
            else:
                detail = part.get('detail', 'auto')
            if detail != 'auto':
                _fail('unsupported_image', 'explicit detail modes require a model preprocessing contract')
            image = _image(url, limits, limits.max_total_image_bytes - total_bytes,
                           limits.max_total_pixels - total_pixels)
            total_bytes += len(image.data)
            total_pixels += image.width * image.height
            parts.append(image)
        else:
            _fail('unsupported_media', 'unsupported content part; PDF, audio and video require separate contracts')
    if images == 0:
        _fail('invalid_content', 'vision input requires an image')
    return VisionInput(tuple(parts), total_bytes, total_pixels, text_bytes)
