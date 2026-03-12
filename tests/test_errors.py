import pytest
from pydantic import BaseModel, Field, ValidationError
from canopee.errors import pretty_print_error

class MockModel(BaseModel):
    x: int
    y: float = Field(..., gt=0)

def test_pretty_print_error_rich(capsys):
    try:
        MockModel(x="not-int", y=-1.0)
    except ValidationError as e:
        # This will test the rich branch if installed
        pretty_print_error(e, title="Test Error")
    
    captured = capsys.readouterr()
    # If rich is installed, it writes to stderr (rich console stderr=True)
    # If not, it also writes to stderr in fallback.
    assert "x" in captured.err
    assert "y" in captured.err
    assert "Test Error" in captured.err

def test_pretty_print_error_fallback(capsys, monkeypatch):
    import sys
    # Force fallback by mocking rich to be missing
    monkeypatch.setitem(sys.modules, "rich.console", None)
    
    try:
        MockModel(x="not-int", y=-1.0)
    except ValidationError as e:
        pretty_print_error(e, title="Fallback Error")
        
    captured = capsys.readouterr()
    assert "Fallback Error" in captured.err
    assert "2 errors detected" in captured.err
    assert "x: " in captured.err
    assert "y: " in captured.err
