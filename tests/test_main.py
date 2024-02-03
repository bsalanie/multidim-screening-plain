from multidim_screening_plain.main import main


def test_main():
    assert main("Bernard") == "Hello from main, Bernard"
