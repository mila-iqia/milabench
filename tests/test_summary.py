
from milabench.validation.validation import Summary


expected_output = """
Errors
======
  Bench
  -----
    1. Errors stuff happened
    2. Errors stuff happened
    3. Errors stuff happened
""".strip()

def test_summary():
    points = [
        '1. Errors stuff happened',
        '2. Errors stuff happened',
        '3. Errors stuff happened',
    ]
    report = Summary()
    
    with report.section("Errors"):
        with report.section('Bench'):
            for p in points:
                report.add(p)
            
    output = ""
    
    def get_output(data):
        nonlocal output
        output = data
    
    report.show(get_output)
    assert output.strip() == expected_output    
    
    

def test_empty_summary():
    points = [
        '1. Errors stuff happened',
        '2. Errors stuff happened',
        '3. Errors stuff happened',
    ]
    report = Summary()
    
    with report.section("Errors"):
        with report.section('Bench'):
            pass
            
    output = ""
    
    def get_output(data):
        nonlocal output
        output = data
    
    report.show(get_output)
    
    assert output.strip() == ""    