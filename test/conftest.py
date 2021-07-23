import glob
import os.path

import pytest


def pytest_addoption(parser):
    parser.addoption("--src", action="store", help="Source folder of test images")
    parser.addoption("--ids", action="store", help="Expected ids", nargs="*", type=int)


@pytest.fixture
def valid_ids(request):
    return request.config.getoption("--ids")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    source = metafunc.config.option.src
    option_value = metafunc.config.option.src
    if 'path' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("path", glob.glob(os.path.join(source, "*.jpg")))
