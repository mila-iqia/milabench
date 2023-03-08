
from milabench.validation.validation import Summary


expected_output = """
Errors
======
  matmult
  -------
    1. Errors stuff happened
    2. Errors stuff happened
    3. Errors stuff happened
  matsub
  ------
    1. Errors stuff happened
    2. Errors stuff happened
    3. Errors stuff happened
""".strip()

def test_summary():
    benchs = ["matmult", "matsub"]
    points = [
        '1. Errors stuff happened',
        '2. Errors stuff happened',
        '3. Errors stuff happened',
    ]
    report = Summary()
    
    with report.section("Errors"):
        for bench in benchs:
            with report.section(bench):
                for p in points:
                    report.add(p)
                
    output = ""
    
    def get_output(data):
        nonlocal output
        output = data
        
    report.show()
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