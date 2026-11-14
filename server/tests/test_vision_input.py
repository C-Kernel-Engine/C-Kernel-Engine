import base64
from dataclasses import replace
import io

from PIL import Image
import pytest

from server.vision_input import (ImagePart, TextPart, VisionInputError,
                                 VisionLimits, prepare_vision_input)


def image_part(fmt='PNG', size=(3, 2)):
    stream = io.BytesIO()
    Image.new('RGB', size, (23, 45, 67)).save(stream, format=fmt)
    mime = 'png' if fmt == 'PNG' else 'jpeg'
    return {'type': 'input_image', 'image_url': f'data:image/{mime};base64,' + base64.b64encode(stream.getvalue()).decode()}


def test_preserves_order_bytes_and_unicode():
    original = image_part()
    content = [{'type': 'text', 'text': 'before —'}, original, {'type': 'input_text', 'text': 'after'}]
    result = prepare_vision_input(content)
    assert result.parts[0] == TextPart('before —')
    assert isinstance(result.parts[1], ImagePart)
    assert result.parts[2] == TextPart('after')
    assert result.parts[1].data == base64.b64decode(original['image_url'].split(',')[1])
    assert result.pixels == 6
    assert result.text_bytes == len('before —after'.encode())
    assert content[1] == original


def test_chat_completions_jpeg_form():
    url = image_part('JPEG')['image_url']
    result = prepare_vision_input([{'type': 'image_url', 'image_url': {'url': url, 'detail': 'auto'}}])
    assert result.parts[0].mime_type == 'image/jpeg'


@pytest.mark.parametrize('url', ['/tmp/image.png', 'file:///tmp/image.png',
    'https://example.com/image.png', 'data:image/svg+xml;base64,PHN2Zy8+',
    'data:image/png;base64,!!!!', 'data:image/png;base64,', None])
def test_rejects_sources_without_fetching(url):
    with pytest.raises(VisionInputError):
        prepare_vision_input([{'type': 'input_image', 'image_url': url}])


def test_mime_mismatch_and_truncated_pixels():
    part = image_part()
    part['image_url'] = part['image_url'].replace('image/png', 'image/jpeg')
    with pytest.raises(VisionInputError, match='MIME'):
        prepare_vision_input([part])
    part = image_part('JPEG')
    raw = base64.b64decode(part['image_url'].split(',')[1])[:-10]
    part['image_url'] = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
    with pytest.raises(VisionInputError):
        prepare_vision_input([part])


def test_exact_byte_and_pixel_limits_then_one_over():
    part = image_part()
    n = len(base64.b64decode(part['image_url'].split(',')[1]))
    limits = VisionLimits(max_image_bytes=n, max_total_image_bytes=n, max_pixels=6, max_total_pixels=6)
    assert prepare_vision_input([part], limits=limits).image_bytes == n
    for key, value in [('max_image_bytes', n-1), ('max_total_image_bytes', n-1),
                       ('max_pixels', 5), ('max_total_pixels', 5)]:
        with pytest.raises(VisionInputError) as e:
            prepare_vision_input([part], limits=replace(limits, **{key: value}))
        assert e.value.code == 'image_limit'


def test_aggregate_limits_and_image_count():
    part = image_part()
    n = len(base64.b64decode(part['image_url'].split(',')[1]))
    with pytest.raises(VisionInputError, match='many images'):
        prepare_vision_input([part, part])
    limits = VisionLimits(max_images=2, max_total_image_bytes=2*n, max_total_pixels=12)
    assert prepare_vision_input([part, part], limits=limits).image_bytes == 2*n
    for key,value in [('max_total_image_bytes',2*n-1), ('max_total_pixels',11)]:
        with pytest.raises(VisionInputError):
            prepare_vision_input([part, part], limits=replace(limits, **{key:value}))


@pytest.mark.parametrize('content', [[], 'path.png', [42], [{'type':'input_file'}],
    [{'type':'input_text','text':4}], [{'type':'input_text','text':'\ud800'}]])
def test_malformed_content(content):
    with pytest.raises(VisionInputError):
        prepare_vision_input(content)


def test_text_and_parts_limits_and_explicit_detail():
    parts = [image_part(), {'type':'text','text':'é'}]
    assert prepare_vision_input(parts, limits=VisionLimits(max_text_bytes=2)).text_bytes == 2
    with pytest.raises(VisionInputError):
        prepare_vision_input(parts, limits=VisionLimits(max_text_bytes=1))
    with pytest.raises(VisionInputError):
        prepare_vision_input(parts, limits=VisionLimits(max_parts=1))
    with pytest.raises(VisionInputError):
        prepare_vision_input([dict(image_part(), detail='high')])


def test_invalid_limits():
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            VisionLimits(max_images=value)


def test_no_global_pillow_changes_or_request_leakage():
    before = Image.MAX_IMAGE_PIXELS
    first = prepare_vision_input([image_part(size=(2,2))])
    with pytest.raises(VisionInputError):
        prepare_vision_input([image_part(size=(5,5))], limits=VisionLimits(max_pixels=4))
    second = prepare_vision_input([image_part(size=(3,2))])
    assert first.parts[0].sha256 != second.parts[0].sha256
    assert Image.MAX_IMAGE_PIXELS == before


def test_limits_reject_before_pixel_decode(monkeypatch):
    part = image_part(size=(10,10))
    def unexpected(*args, **kwargs):
        raise AssertionError('pixel decode must not execute')
    monkeypatch.setattr(Image.Image, 'load', unexpected)
    with pytest.raises(VisionInputError) as error:
        prepare_vision_input([part], limits=VisionLimits(max_pixels=99))
    assert error.value.code == 'image_limit'


def test_encoded_limit_rejects_before_base64_allocation(monkeypatch):
    part = image_part()
    def unexpected(*args, **kwargs):
        raise AssertionError('base64 decode must not execute')
    monkeypatch.setattr(base64, 'b64decode', unexpected)
    with pytest.raises(VisionInputError) as error:
        prepare_vision_input([part], limits=VisionLimits(max_image_bytes=1))
    assert error.value.code == 'image_limit'


def test_animated_image_rejected():
    stream = io.BytesIO()
    frames = [Image.new('RGB', (2,2), color) for color in ('red','blue')]
    frames[0].save(stream, format='PNG', save_all=True, append_images=frames[1:])
    url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    with pytest.raises(VisionInputError, match='multi-frame'):
        prepare_vision_input([{'type':'input_image','image_url':url}])


def test_unknown_fields_not_silently_discarded():
    with pytest.raises(VisionInputError):
        prepare_vision_input([dict(image_part(), file_id='another-image')])
    with pytest.raises(VisionInputError):
        prepare_vision_input([image_part(), {'type':'text','text':'x','image':'ignored'}])


def test_valid_upload_does_not_enable_text_session(tmp_path):
    from server.runtime import resolve_runtime_vision_capability
    assert prepare_vision_input([image_part()]).pixels == 6
    assert resolve_runtime_vision_capability(tmp_path) is False
